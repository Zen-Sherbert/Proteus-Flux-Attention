"""
Shared utilities for the Proteus Attention example scripts.
"""

from __future__ import annotations

import math
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import sys

import torch
import torch.nn.functional as F

_DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "tiny_ctx_excerpt.txt"
_EOS_TEXT = "<|endoftext|>"

try:
    import tiktoken  # type: ignore

    _ENCODER = tiktoken.get_encoding("gpt2")
except Exception:  # pragma: no cover - optional dependency
    _ENCODER = None


class ByteFallbackTokenizer:
    """Simple byte-level tokenizer used when `tiktoken` is unavailable."""

    n_vocab: int = 256

    def __init__(self) -> None:
        self.eos_token_id = 10  # newline acts as an EOS surrogate

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:
        return bytes(int(t) % 256 for t in tokens).decode("utf-8", errors="ignore")


class TikTokenWrapper:
    """Thin wrapper to unify the encode/decode API with the fallback tokenizer."""

    def __init__(self) -> None:
        self.n_vocab = int(getattr(_ENCODER, "n_vocab", 50257))
        self.eos_token_id = int(
            getattr(_ENCODER, "eot_token", getattr(_ENCODER, "eos_token", 50256))
        )

    def encode(self, text: str) -> List[int]:
        return _ENCODER.encode(text, disallowed_special=())  # type: ignore[operator]

    def decode(self, tokens: Sequence[int]) -> str:
        return _ENCODER.decode(tokens)  # type: ignore[operator]


def get_tokenizer() -> TikTokenWrapper | ByteFallbackTokenizer:
    if _ENCODER is None:
        return ByteFallbackTokenizer()
    return TikTokenWrapper()


class EMA:
    """Exponential moving average wrapper for smoother evaluation."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        decay = float(decay)
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie in (0, 1).")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = param.detach().clone()

    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow_param = self.shadow.get(name)
            if shadow_param is None:
                self.shadow[name] = param.detach().clone()
                continue
            shadow_param.lerp_(param.detach(), 1.0 - self.decay)

    @contextlib.contextmanager
    def swap(self, model: torch.nn.Module):
        backups: dict[str, torch.Tensor] = {}
        try:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                shadow_param = self.shadow.get(name)
                if shadow_param is None:
                    continue
                backups[name] = param.detach().to("cpu", copy=True)
                param.data.copy_(shadow_param)
            yield
        finally:
            for name, param in model.named_parameters():
                backup = backups.get(name)
                if backup is not None:
                    param.data.copy_(backup.to(device=param.device, dtype=param.dtype))


def load_corpus(path: Optional[Path] = None, *, max_chars: int = 64 * 1024 * 1024) -> str:
    target = path or _DEFAULT_DATA
    max_chars = max(1024, int(max_chars))
    if target.is_dir():
        import random

        candidates = sorted(target.glob("**/*.txt"))
        if not candidates:
            return ""
        random.shuffle(candidates)
        pieces: list[str] = []
        total = 0
        for candidate in candidates:
            try:
                chunk = candidate.read_text(encoding="utf-8")
            except Exception:
                continue
            pieces.append(chunk)
            pieces.append(_EOS_TEXT)
            total += len(chunk)
            if total >= max_chars:
                break
        corpus = "\n".join(pieces)
        if not corpus.endswith(_EOS_TEXT):
            corpus += f"\n{_EOS_TEXT}"
        return corpus[:max_chars]
    if not target.is_file():
        return (
            "In the beginning there was Proteus Attention. "
            "It discovered new mixtures of heads and tokens, "
            "unlocking astonishing context windows on modest devices."
            f"\n{_EOS_TEXT}\n"
        )
    try:
        with target.open("r", encoding="utf-8") as handle:
            chunk = handle.read(max_chars)
            if _EOS_TEXT not in chunk[-len(_EOS_TEXT) :]:
                chunk = f"{chunk.rstrip()}\n{_EOS_TEXT}\n"
            return chunk
    except Exception:
        return ""


def build_batches(
    tokens: Sequence[int],
    seq_len: int,
    batch_size: int,
    *,
    device: torch.device,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    total = len(tokens)
    if total <= seq_len:
        repeat = math.ceil((seq_len + 1) / total)
        tokens = list(tokens) * repeat
        total = len(tokens)

    import random

    while True:
        batch_inputs = torch.empty(batch_size, seq_len, dtype=torch.long)
        batch_targets = torch.empty(batch_size, seq_len, dtype=torch.long)
        for b in range(batch_size):
            start = random.randint(0, total - seq_len - 1)
            chunk = tokens[start : start + seq_len + 1]
            batch_inputs[b] = torch.tensor(chunk[:-1], dtype=torch.long)
            batch_targets[b] = torch.tensor(chunk[1:], dtype=torch.long)
        yield batch_inputs.to(device), batch_targets.to(device)


@dataclass
class TrainingConfig:
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    vocab_size: int = 50304  # GPT-2 tokenizer size
    max_seq_len: int = 4096
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_checkpoint_dir(name: str) -> Path:
    root = Path(__file__).resolve().parent / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_prompt(
    *,
    default_prompt: str,
    sample_path: Optional[Path],
    label: str,
    char_limit: int,
) -> str:
    """
    Load a prompt string from disk, truncating to ``char_limit`` characters.

    When ``sample_path`` is ``None`` and stdin is interactive we ask the user
    for an optional path; pressing ENTER keeps the default prompt.
    """

    prompt_path = sample_path
    if prompt_path is None and sys.stdin.isatty():
        try:
            response = input(f"[{label}] prompt file (ENTER for default): ").strip()
        except EOFError:
            response = ""
        if response:
            prompt_path = Path(response)

    if prompt_path is None:
        return default_prompt

    prompt_path = prompt_path.expanduser()
    char_limit = max(1, int(char_limit))
    chunk_size = 8192
    remaining = char_limit
    chunks: list[str] = []
    truncated = False
    try:
        with prompt_path.open("r", encoding="utf-8") as handle:
            while remaining > 0:
                chunk = handle.read(min(remaining, chunk_size))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining == 0 and handle.read(1):
                truncated = True
    except Exception as exc:
        print(f"[{label}] warning: unable to read prompt file ({exc}); using default prompt.")
        return default_prompt

    prompt = "".join(chunks).strip()
    if not prompt:
        print(f"[{label}] warning: prompt file was empty; using default prompt.")
        return default_prompt
    if truncated:
        print(f"[{label}] prompt truncated to {char_limit} characters from {prompt_path}.")
    return prompt


def resolve_dataset_path(
    data_path: Optional[Path],
    *,
    label: str,
) -> Optional[Path]:
    if data_path is not None:
        resolved = data_path.expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"[{label}] training data '{resolved}' does not exist.")
        return resolved
    if not sys.stdin.isatty():
        return None
    while True:
        try:
            response = input(f"[{label}] training data file (ENTER for default): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not response:
            return None
        candidate = Path(response).expanduser()
        if candidate.exists():
            return candidate
        print(f"[{label}] path '{candidate}' not found. Please try again.")


def encode_prompt_tokens(
    tokenizer,
    prompt: str,
    *,
    max_seq_len: int,
    label: str,
) -> List[int]:
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_seq_len:
        tokens = tokens[-max_seq_len:]
        print(f"[{label}] prompt truncated to last {max_seq_len} tokens to fit the model context.")
    return tokens


def split_train_val(
    tokens: Sequence[int],
    seq_len: int,
    *,
    val_fraction: float = 0.1,
) -> Tuple[List[int], List[int]]:
    seq_len = max(1, int(seq_len))
    total = len(tokens)
    if total <= seq_len * 2 or val_fraction <= 0.0:
        return list(tokens), list(tokens)
    split_point = int(total * (1.0 - val_fraction))
    min_window = seq_len + 1
    split_point = max(min_window, min(split_point, total - min_window))
    return list(tokens[:split_point]), list(tokens[split_point:])


def evaluate_language_model(
    model,
    tokens: Sequence[int],
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    iters: int = 4,
) -> Tuple[float, float]:
    if not tokens:
        return float("nan"), float("nan")
    batches = build_batches(tokens, seq_len, batch_size, device=device)
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for _ in range(max(1, iters)):
            x, y = next(batches)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, model.cfg.vocab_size if hasattr(model, "cfg") else logits.size(-1)),
                y.view(-1),
            )
            losses.append(float(loss.item()))
    if not losses:
        return float("nan"), float("nan")
    mean_loss = sum(losses) / len(losses)
    ppl = float(torch.exp(torch.tensor(mean_loss)).item())
    return mean_loss, ppl


def init_attn_counters() -> Dict[str, float]:
    return {
        "prototype_updates": 0.0,
        "memory_slots_filled": 0.0,
        "shortlist_steps": 0.0,
    }


def accumulate_attention_counters(model, counters: Dict[str, float]) -> None:
    blocks = getattr(model, "blocks", [])
    for block in blocks:
        attn = getattr(block, "self_attn", None)
        if attn is None:
            continue
        attention = getattr(attn, "attention", None)
        if attention is None:
            continue
        proto = getattr(attention, "_last_proto_stats", None)
        if proto:
            counters["prototype_updates"] += float(proto.get("updated_gates", 0) or 0)
        memory_counts = getattr(attention, "memory_counts", None)
        if memory_counts is not None:
            counters["memory_slots_filled"] += float((memory_counts > 0).sum().item())
        stats = getattr(attention, "last_head_stats", None)
        if stats and stats.get("mode") == "shortlist":
            counters["shortlist_steps"] += 1.0


def summarize_attention_counters(
    counters: Dict[str, float],
    steps: int,
) -> Dict[str, float]:
    if steps <= 0:
        steps = 1
    return {
        "prototype_updates": counters["prototype_updates"],
        "memory_slots_filled": counters["memory_slots_filled"],
        "shortlist_usage_ratio": counters["shortlist_steps"] / steps,
    }


def print_metric_block(label: str, metrics: Dict[str, float]) -> None:
    print(f"[{label}] metrics:")
    for key, value in metrics.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        print(f"  - {key}: {value:.4f}" if isinstance(value, float) else f"  - {key}: {value}")


@torch.inference_mode()
def sample_autoregressive(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_seq_len: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    min_new_tokens: int = 0,
    eos_token_id: Optional[int] = None,
    top_a_mode: str = "off",
    top_a_lambda: float = 0.5,
    top_a_tau: float = 0.7,
    top_a_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Lightweight sampler with top-p, repetition penalty, and n-gram blocking."""

    temperature = max(1e-4, float(temperature))
    top_p = float(max(0.0, min(1.0, top_p)))
    repetition_penalty = float(max(1.0, repetition_penalty))
    no_repeat_ngram_size = max(0, int(no_repeat_ngram_size))
    min_new_tokens = max(0, int(min_new_tokens))
    max_new_tokens = max(1, int(max_new_tokens))

    seq = input_ids
    generated = 0
    top_a_mode_norm = (top_a_mode or "off").lower()
    allow_top_a = top_a_mode_norm in {"blend", "filter"}
    setattr(model, "last_top_a_stats", None)
    while generated < max_new_tokens:
        seq_trim = seq[:, -max_seq_len:]
        logits = model(seq_trim)
        step_logits = logits[:, -1, :].clone()
        step_logits = step_logits / temperature

        if repetition_penalty > 1.0:
            penalty_tokens = torch.unique(seq_trim)
            step_logits[:, penalty_tokens] /= repetition_penalty

        if (
            eos_token_id is not None
            and generated < min_new_tokens
            and 0 <= eos_token_id < step_logits.size(-1)
        ):
            step_logits[:, eos_token_id] = -float("inf")

        if no_repeat_ngram_size > 0 and seq_trim.size(1) >= no_repeat_ngram_size:
            banned = _collect_ngram_bans(seq_trim[0], no_repeat_ngram_size)
            if banned:
                ban_tensor = torch.tensor(
                    sorted(banned), device=step_logits.device, dtype=torch.long
                )
                step_logits[0, ban_tensor] = -float("inf")

        consensus_vec: Optional[torch.Tensor] = None
        embed_weight: Optional[torch.Tensor] = None
        can_use_top_a = False
        if allow_top_a and seq.size(0) == 1:
            consensus_tensor = _get_last_head_consensus(model)
            if (
                consensus_tensor is not None
                and consensus_tensor.size(0) >= 1
                and consensus_tensor.size(1) >= 1
            ):
                consensus_vec = consensus_tensor[0, -1, :].detach()
                embed_weight = _resolve_embedding_weight(model)
                can_use_top_a = embed_weight is not None

        top_a_stats = None
        if seq.size(0) == 1:
            next_token, top_a_stats = _sample_with_top_a(
                step_logits[0],
                top_p,
                top_a_mode_norm if can_use_top_a else "off",
                consensus_vec,
                embed_weight,
                top_a_lambda,
                top_a_tau,
                top_a_threshold,
            ).to(step_logits.device)
        else:
            next_token = _sample_from_logits(step_logits, top_p)
        if top_a_stats is not None:
            setattr(model, "last_top_a_stats", top_a_stats)
        seq = torch.cat([seq, next_token.unsqueeze(-1)], dim=1)
        generated += 1

        if (
            eos_token_id is not None
            and int(next_token.item()) == eos_token_id
            and generated >= min_new_tokens
        ):
            break
    return seq


def _sample_from_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    probs_sum = probs.sum(dim=-1, keepdim=True)
    zero_mask = probs_sum <= 0
    if torch.any(zero_mask):
        probs = torch.where(
            zero_mask,
            torch.full_like(probs, 1.0 / probs.size(-1)),
            probs,
        )
        probs_sum = probs.sum(dim=-1, keepdim=True)
    probs = probs / probs_sum

    if top_p <= 0.0 or top_p >= 0.999:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    cutoff[..., 0] = False
    filtered = sorted_probs.masked_fill(cutoff, 0.0)
    filtered_sum = filtered.sum(dim=-1, keepdim=True)
    filtered_sum = torch.where(
        filtered_sum <= 0,
        torch.ones_like(filtered_sum),
        filtered_sum,
    )
    filtered = filtered / filtered_sum
    sampled = torch.multinomial(filtered, num_samples=1)
    return sorted_idx.gather(-1, sampled).squeeze(-1)


def _sample_with_top_a(
    logits: torch.Tensor,
    top_p: float,
    top_a_mode: str,
    consensus_vec: Optional[torch.Tensor],
    embedding_weight: Optional[torch.Tensor],
    top_a_lambda: float,
    top_a_tau: float,
    top_a_threshold: Optional[float],
) -> Tuple[torch.Tensor, Optional[dict]]:
    if logits.ndim != 1:
        token = _sample_from_logits(logits.unsqueeze(0), top_p).squeeze(0)
        return token, None
    nucleus_idx, nucleus_logits = _build_nucleus_from_logits(logits, top_p)
    if nucleus_idx.numel() == 0:
        return logits.argmax().unsqueeze(0), None
    adjusted_logits = nucleus_logits.clone()
    stats: dict[str, float | int | str] = {
        "mode": top_a_mode,
        "nucleus": int(nucleus_idx.numel()),
        "filtered": 0,
        "lambda": float(top_a_lambda),
    }
    if (
        top_a_mode != "off"
        and consensus_vec is not None
        and embedding_weight is not None
        and nucleus_idx.numel() > 0
    ):
        agreements = _compute_agreement_scores(
            consensus_vec, embedding_weight, nucleus_idx, top_a_tau
        )
        if top_a_mode == "filter" and top_a_threshold is not None:
            mask = agreements >= float(top_a_threshold)
            if not torch.any(mask):
                mask = torch.ones_like(agreements, dtype=torch.bool)
            stats["filtered"] = int((~mask).sum().item())
            nucleus_idx = nucleus_idx[mask]
            adjusted_logits = adjusted_logits[mask]
            agreements = agreements[mask]
        adjusted_logits = adjusted_logits + float(top_a_lambda) * agreements
        stats["agree_mean"] = float(agreements.mean().item())
        stats["agree_min"] = float(agreements.min().item())
        stats["agree_max"] = float(agreements.max().item())
    probs = F.softmax(adjusted_logits, dim=-1)
    choice = torch.multinomial(probs, 1).item()
    token = nucleus_idx.new_tensor(int(nucleus_idx[choice].item()), dtype=torch.long)
    return token, stats


def _build_nucleus_from_logits(
    logits: torch.Tensor,
    top_p: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cumulative <= max(top_p, 1e-6)
    if nucleus_mask.numel() > 0:
        nucleus_mask[0] = True
    nucleus_idx = sorted_idx[nucleus_mask]
    if nucleus_idx.numel() == 0:
        nucleus_idx = sorted_idx[:1]
    nucleus_logits = logits[nucleus_idx]
    return nucleus_idx, nucleus_logits


def _compute_agreement_scores(
    consensus_vec: torch.Tensor,
    embedding_weight: torch.Tensor,
    indices: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    tau = max(float(tau), 1e-5)
    device = embedding_weight.device
    consensus = F.normalize(consensus_vec.to(device), dim=-1)
    token_vecs = embedding_weight.index_select(0, indices.to(device))
    token_vecs = F.normalize(token_vecs, dim=-1)
    scores = torch.matmul(token_vecs, consensus.unsqueeze(-1)).squeeze(-1)
    return scores / tau


def _get_last_head_consensus(model) -> Optional[torch.Tensor]:
    consensus: Optional[torch.Tensor] = None
    getter = getattr(model, "last_head_consensus", None)
    if callable(getter):
        try:
            consensus = getter()
        except Exception:
            consensus = None
    if consensus is None:
        consensus = getattr(model, "last_head_consensus", None)
    return consensus


def _resolve_embedding_weight(model) -> Optional[torch.Tensor]:
    token_emb = getattr(model, "token_emb", None)
    if token_emb is not None and hasattr(token_emb, "weight"):
        return token_emb.weight
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        return lm_head.weight
    return None


def _collect_ngram_bans(sequence: torch.Tensor, ngram_size: int) -> set[int]:
    if ngram_size <= 1 or sequence.size(0) < ngram_size:
        return set()
    tokens = sequence.tolist()
    prefix = tuple(tokens[-(ngram_size - 1) :])
    bans: set[int] = set()
    for idx in range(len(tokens) - ngram_size + 1):
        history = tuple(tokens[idx : idx + ngram_size - 1])
        next_token = tokens[idx + ngram_size - 1]
        if history == prefix:
            bans.add(next_token)
    return bans


def interactive_inference_loop(
    model,
    tokenizer,
    *,
    device: torch.device,
    max_seq_len: int,
    sample_tokens: int = 128,
    label: str = "inference",
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    min_new_tokens: int = 20,
    max_new_tokens: Optional[int] = None,
    top_a_mode: str = "blend",
    top_a_lambda: float = 0.5,
    top_a_tau: float = 0.7,
    top_a_threshold: Optional[float] = None,
) -> None:
    if not sys.stdin.isatty():
        print(f"[{label}] skipping interactive inference (stdin not a TTY).")
        return
    print(f"[{label}] entering inference mode. Press Ctrl+C to exit.")
    while True:
        try:
            prompt = input("Prompt> ")
        except (KeyboardInterrupt, EOFError):
            print("\n[inference] exiting.")
            break
        if not prompt.strip():
            continue
        encoded = encode_prompt_tokens(
            tokenizer,
            prompt,
            max_seq_len=max_seq_len,
            label=label,
        )
        input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
        prompt_len = input_ids.size(1)
        eos_token_id = getattr(tokenizer, "eos_token_id", getattr(tokenizer, "eot_token", None))
        generated = sample_autoregressive(
            model,
            input_ids,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens or sample_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            eos_token_id=eos_token_id,
            top_a_mode=top_a_mode,
            top_a_lambda=top_a_lambda,
            top_a_tau=top_a_tau,
            top_a_threshold=top_a_threshold,
        )
        top_a_stats = getattr(model, "last_top_a_stats", None)
        continuation = tokenizer.decode(generated[0, prompt_len:].tolist())
        if top_a_stats:
            agree_mean = top_a_stats.get("agree_mean")
            agree_str = f"{agree_mean:.3f}" if isinstance(agree_mean, float) else "n/a"
            print(
                f"[{label}] top-a mode={top_a_stats.get('mode')} "
                f"nucleus={top_a_stats.get('nucleus')} "
                f"filtered={top_a_stats.get('filtered')} "
                f"agree_mean={agree_str}"
            )
        print(continuation)
