#!/usr/bin/env python
"""
Synthetic checks for the chunked Flux pipeline.

The helpers here exercise the sentinel/needle recall behaviour, the effect of
adding an auxiliary DNA-style importance boost, and the ordering guarantees that
keep indices remain monotonic (mirroring the RoPE position story).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Sequence, Union, Optional

import torch

# Ensure the project `src` tree is importable when running standalone.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
SAMPLES_DIR = THIS_DIR / "samples"
DEFAULT_SAMPLE_FILE = SAMPLES_DIR / "flux_sample.txt"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from proteus_attention.tools.chunked_flux import (
    _chunk_iter,
    _finalise_indices,
    _select_top_tokens,
)


@dataclass
class ChunkLog:
    """Simple container for per-chunk diagnostics."""

    start: int
    end: int
    promoted: List[int]
    scores: List[float]


class _MockModel:
    """Mimic the small interface used by `_select_top_tokens`."""

    def __init__(self) -> None:
        self._last_token_importance: torch.Tensor | None = None
        self._last_token_mask: torch.Tensor | None = None


def _run_pipeline(
    *,
    seq_len: int,
    d_model: int,
    chunk_len: int,
    per_chunk_budget: int,
    buffer_tokens: int,
    sentinel_idx: Union[int, Sequence[int]],
    base_boost: Union[float, torch.Tensor, dict[int, float]],
    dna_boost: Union[float, dict[int, float]] = 0.0,
    rng_seed: int = 123,
    adaptive_margin: Optional[float] = None,
    max_chunk_extra: int = 8,
) -> Tuple[torch.Tensor, List[ChunkLog]]:
    """Execute the chunking helpers with synthetic importance scores."""

    torch.manual_seed(rng_seed)
    model = _MockModel()
    chunk_scores: List[torch.Tensor] = []
    chunk_indices: List[torch.Tensor] = []
    logs: List[ChunkLog] = []

    if isinstance(sentinel_idx, int):
        sentinels = [sentinel_idx]
    else:
        sentinels = list(sentinel_idx)

    def _boost_for(idx: int, value: Union[float, dict[int, float]]) -> float:
        if isinstance(value, dict):
            return float(value.get(idx, 0.0))
        return float(value)

    for start, end in _chunk_iter(seq_len, chunk_len):
        current_len = end - start
        importance = torch.randn(1, current_len, dtype=torch.float32) * 0.05
        sentinel_present = False
        for idx in sentinels:
            if start <= idx < end:
                sentinel_present = True
                pos = idx - start
                importance[0, pos] += _boost_for(idx, base_boost)
                if dna_boost:
                    importance[0, pos] += _boost_for(idx, dna_boost)
        if not sentinel_present:
            # Create a couple of distractors so the sentinel has competition.
            importance[0, torch.randint(0, current_len, (2,))] += 0.25

        model._last_token_importance = importance
        model._last_token_mask = None
        budget_this_chunk = min(per_chunk_budget, current_len)
        if adaptive_margin is not None and budget_this_chunk < current_len:
            raw_scores = importance.squeeze(0).to(torch.float32)
            max_score = float(raw_scores.max().item()) if raw_scores.numel() > 0 else 0.0
            threshold = max_score - float(adaptive_margin)
            qualifying = int((raw_scores >= threshold).sum().item())
            if qualifying > budget_this_chunk:
                extra = min(max_chunk_extra, qualifying - budget_this_chunk)
                budget_this_chunk = min(current_len, budget_this_chunk + extra)
        promoted = _select_top_tokens(
            model,
            start,
            per_chunk_budget=budget_this_chunk,
        )
        if promoted is None:
            continue
        scores_tensor, indices_tensor = promoted
        chunk_scores.append(scores_tensor)
        chunk_indices.append(indices_tensor)
        logs.append(
            ChunkLog(
                start=start,
                end=end,
                promoted=indices_tensor.tolist(),
                scores=[float(s) for s in scores_tensor],
            )
        )

    if not chunk_scores:
        return torch.empty(0, dtype=torch.long), logs

    all_scores = torch.cat(chunk_scores)
    all_indices = torch.cat(chunk_indices)
    keep_indices_tensor, _ = _finalise_indices(
        all_scores, all_indices, max_tokens=buffer_tokens
    )
    return keep_indices_tensor.to(torch.long), logs


def _format_logs(logs: Iterable[ChunkLog]) -> str:
    lines = []
    for item in logs:
        lines.append(
            f"[chunk {item.start:>5}:{item.end:<5}] promoted={item.promoted} scores={[round(s, 3) for s in item.scores]}"
        )
    return "\n".join(lines)


def _load_sample_tokens(path: Path, limit: Optional[int] = None) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to read sample file {path}: {exc}") from exc
    tokens = text.replace("\n", " ").split()
    if limit is not None and limit > 0:
        tokens = tokens[:limit]
    return tokens


def _find_phrase_positions(
    tokens: Sequence[str],
    phrase: Sequence[str],
    *,
    casefold: bool,
) -> List[int]:
    if not phrase:
        return []
    if casefold:
        lowered = [tok.casefold() for tok in tokens]
        needle = [tok.casefold() for tok in phrase]
    else:
        lowered = list(tokens)
        needle = list(phrase)
    k = len(needle)
    hits: List[int] = []
    for idx in range(0, len(lowered) - k + 1):
        if lowered[idx : idx + k] == needle:
            hits.append(idx)
    return hits


def _sample_text_demo(args: argparse.Namespace) -> None:
    tokens: List[str]
    source_label: str
    tokens = []
    source_label = ""
    if args.sample_file is not None:
        sample_path = args.sample_file
        if sample_path.exists():
            tokens = _load_sample_tokens(sample_path, limit=args.sample_limit)
            source_label = str(sample_path)
        else:
            print(f"[sample] Sample file {sample_path} not found; falling back to inline text.")
    if not tokens:
        sample_text = args.sample_text or ""
        tokens = sample_text.replace("\n", " ").split()
        if args.sample_limit is not None and args.sample_limit > 0:
            tokens = tokens[: args.sample_limit]
        source_label = source_label or "inline sample text"
    if not tokens:
        print("[sample] No tokens available for sample demo; skipping.")
        return
    print(
        f"[sample] Loaded {len(tokens):,} tokens from {source_label} "
        f"(limit={args.sample_limit or 'none'})"
    )
    phrase = (args.sample_needle.split() if args.sample_needle else [])
    sentinel_indices = _find_phrase_positions(
        tokens,
        phrase,
        casefold=args.sample_casefold,
    ) if phrase else []
    if not sentinel_indices:
        # Fall back to simple anchors if no explicit phrase hit.
        pivots = [
            max(0, len(tokens) // 4),
            max(0, len(tokens) // 2),
            max(0, (3 * len(tokens)) // 4),
        ]
        sentinel_indices = sorted({idx for idx in pivots if idx < len(tokens)})
        if args.sample_needle:
            print(
                f"[sample] Phrase '{args.sample_needle}' not found; "
                "falling back to heuristic sentinel positions."
            )
    if not sentinel_indices:
        print("[sample] Unable to determine sentinel indices; skipping sample demo.")
        return
    chunk_len = min(args.chunk_len, max(1, len(tokens)))
    per_chunk_budget = min(chunk_len, max(args.per_chunk_budget, len(sentinel_indices)))
    buffer_tokens = max(args.buffer_tokens, len(sentinel_indices) * 2)
    keep_indices, _ = _run_pipeline(
        seq_len=len(tokens),
        d_model=args.d_model,
        chunk_len=chunk_len,
        per_chunk_budget=per_chunk_budget,
        buffer_tokens=buffer_tokens,
        sentinel_idx=sentinel_indices,
        base_boost=5.0,
        dna_boost=1.0,
    )
    keep_indices = keep_indices.to(torch.long)
    retained_flags = {
        idx: bool((keep_indices == idx).any()) for idx in sentinel_indices
    }
    print(
        f"[sample] Retained {sum(retained_flags.values())}/{len(sentinel_indices)} sentinel anchors "
        f"(chunk_len={chunk_len}, buffer_tokens={buffer_tokens})."
    )
    for idx in sentinel_indices:
        start = max(0, idx - 5)
        end = min(len(tokens), idx + 6)
        snippet = " ".join(tokens[start:end])
        flag = "✓" if retained_flags[idx] else "✗"
        print(f"  [{flag}] idx={idx}: {snippet}")


def _needle_cluster_test(args: argparse.Namespace) -> None:
    chunk_origin = args.chunk_len * 2
    sentinels = [chunk_origin + offset for offset in range(0, 100, 10)]
    keep_indices, _ = _run_pipeline(
        seq_len=args.seq_len,
        d_model=args.d_model,
        chunk_len=args.chunk_len,
        per_chunk_budget=max(1, args.per_chunk_budget // 2),
        buffer_tokens=max(args.buffer_tokens, len(sentinels)),
        sentinel_idx=sentinels,
        base_boost=3.0,
        dna_boost=0.0,
    )
    retained = sum(bool((keep_indices == idx).any()) for idx in sentinels)
    recall = retained / len(sentinels)
    print(
        f"[cluster] needles kept {retained}/{len(sentinels)} "
        f"(recall={recall:.2f}) with per_chunk_budget={max(1, args.per_chunk_budget // 2)}"
    )

    adaptive_margin = args.cluster_adaptive_margin
    if adaptive_margin is not None and adaptive_margin >= 0.0:
        keep_indices_adapt, _ = _run_pipeline(
            seq_len=args.seq_len,
            d_model=args.d_model,
            chunk_len=args.chunk_len,
            per_chunk_budget=max(1, args.per_chunk_budget // 2),
            buffer_tokens=max(args.buffer_tokens, len(sentinels)),
            sentinel_idx=sentinels,
            base_boost=3.0,
            dna_boost=0.0,
            adaptive_margin=adaptive_margin,
            max_chunk_extra=args.cluster_max_extra,
        )
        retained_adapt = sum(bool((keep_indices_adapt == idx).any()) for idx in sentinels)
        recall_adapt = retained_adapt / len(sentinels)
        print(
            f"[cluster-adapt] needles kept {retained_adapt}/{len(sentinels)} "
            f"(recall={recall_adapt:.2f}) margin={adaptive_margin} max_extra={args.cluster_max_extra}"
        )


def _fading_signal_test(
    args: argparse.Namespace,
    boosts: Sequence[float],
    trials: int,
    sentinel_idx: int,
) -> None:
    print("[fading] sensitivity sweep:")
    for boost in boosts:
        hits = 0
        for seed in range(trials):
            keep_indices, _ = _run_pipeline(
                seq_len=args.seq_len,
                d_model=args.d_model,
                chunk_len=args.chunk_len,
                per_chunk_budget=args.per_chunk_budget,
                buffer_tokens=args.buffer_tokens,
                sentinel_idx=sentinel_idx,
                base_boost=boost,
                dna_boost=0.0,
                rng_seed=seed,
            )
            if bool((keep_indices == sentinel_idx).any()):
                hits += 1
        rate = hits / trials
        print(f"  boost={boost:.2f} -> recall {rate:.2%} ({hits}/{trials})")


def _jigsaw_puzzle_test(args: argparse.Namespace) -> None:
    early = args.chunk_len // 2
    late = args.seq_len - args.chunk_len // 3
    sentinels = [early, late]
    keep_indices, _ = _run_pipeline(
        seq_len=args.seq_len,
        d_model=args.d_model,
        chunk_len=args.chunk_len,
        per_chunk_budget=args.per_chunk_budget,
        buffer_tokens=max(args.buffer_tokens, len(sentinels)),
        sentinel_idx=sentinels,
        base_boost=4.0,
        dna_boost=0.5,
    )
    both = all(bool((keep_indices == idx).any()) for idx in sentinels)
    print(f"[jigsaw] both clues retained? {both} | keep_indices={keep_indices.tolist()}")


def _live_fire_test(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[live-fire] skipping (transformers unavailable: {exc})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.live_fire_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
    model.eval()

    text = args.live_fire_text
    max_len = min(args.seq_len, getattr(tokenizer, "model_max_length", args.seq_len))
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", None)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    attentions = outputs.attentions[-1][0]  # (heads, T, T)
    importance = attentions.mean(dim=0).mean(dim=0)  # (T,)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = importance.numel()
    chunk_len = min(args.chunk_len, seq_len)

    model_stub = _MockModel()
    chunk_scores: List[torch.Tensor] = []
    chunk_indices: List[torch.Tensor] = []
    for start, end in _chunk_iter(seq_len, chunk_len):
        current_len = end - start
        imp_slice = importance[start:end].unsqueeze(0).to(torch.float32)
        model_stub._last_token_importance = imp_slice
        model_stub._last_token_mask = None
        promoted = _select_top_tokens(
            model_stub,
            start,
            per_chunk_budget=args.per_chunk_budget,
        )
        if promoted is None:
            continue
        scores_tensor, indices_tensor = promoted
        chunk_scores.append(scores_tensor)
        chunk_indices.append(indices_tensor)

    if not chunk_scores:
        print("[live-fire] no tokens promoted.")
        return

    keep_indices = _finalise_indices(
        torch.cat(chunk_scores),
        torch.cat(chunk_indices),
        max_tokens=min(args.buffer_tokens, seq_len),
    )
    retained_tokens = [tokens[idx] for idx in keep_indices.tolist()]
    print(f"[live-fire] retained {len(retained_tokens)} tokens:")
    print("  " + " ".join(retained_tokens))


def run_checks(args: argparse.Namespace) -> None:
    seq_len = args.seq_len
    chunk_len = args.chunk_len
    sentinel_idx = max(0, min(args.sentinel_index, max(seq_len - 1, 0)))
    if sentinel_idx != args.sentinel_index and args.verbose:
        print(
            f"[warn] adjusted sentinel index from {args.sentinel_index} to {sentinel_idx} to fit sequence length."
        )
    per_chunk_budget = args.per_chunk_budget
    buffer_tokens = args.buffer_tokens

    print("=== Needle recall (high router score) ===")
    keep_indices, logs = _run_pipeline(
        seq_len=seq_len,
        d_model=args.d_model,
        chunk_len=chunk_len,
        per_chunk_budget=per_chunk_budget,
        buffer_tokens=buffer_tokens,
        sentinel_idx=sentinel_idx,
        base_boost=5.0,
        dna_boost=0.0,
    )
    if args.verbose:
        print(_format_logs(logs))
    retained = bool((keep_indices == sentinel_idx).any())
    print(f"retained sentinel? {retained} | keep_indices={keep_indices.tolist()}")

    print("\n=== DNA teleportation hypothesis ===")
    keep_no_dna, logs_no_dna = _run_pipeline(
        seq_len=seq_len,
        d_model=args.d_model,
        chunk_len=chunk_len,
        per_chunk_budget=per_chunk_budget,
        buffer_tokens=buffer_tokens,
        sentinel_idx=sentinel_idx,
        base_boost=-0.2,
        dna_boost=0.0,
    )
    keep_with_dna, logs_with_dna = _run_pipeline(
        seq_len=seq_len,
        d_model=args.d_model,
        chunk_len=chunk_len,
        per_chunk_budget=per_chunk_budget,
        buffer_tokens=buffer_tokens,
        sentinel_idx=sentinel_idx,
        base_boost=-0.2,
        dna_boost=1.0,
    )
    if args.verbose:
        print("-- without DNA boost --")
        print(_format_logs(logs_no_dna))
        print("-- with DNA boost --")
        print(_format_logs(logs_with_dna))
    no_dna_retained = bool((keep_no_dna == sentinel_idx).any())
    dna_retained = bool((keep_with_dna == sentinel_idx).any())
    print(
        f"without DNA retained? {no_dna_retained} | with DNA retained? {dna_retained}"
    )

    print("\n=== Ordering sanity (RoPE alignment) ===")
    monotonic = bool(torch.all(keep_indices[:-1] <= keep_indices[1:])) if keep_indices.numel() > 1 else True
    print(f"keep_indices monotonic? {monotonic}")

    print("\n=== Needle cluster stress test ===")
    _needle_cluster_test(args)

    print("\n=== Fading signal sweep ===")
    _fading_signal_test(
        args,
        boosts=args.fading_boosts,
        trials=args.fading_trials,
        sentinel_idx=sentinel_idx,
    )

    print("\n=== Jigsaw puzzle synthesis ===")
    _jigsaw_puzzle_test(args)

    if args.enable_live_fire:
        print("\n=== Live fire integration ===")
        _live_fire_test(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic validation checks for the chunked Flux demo."
    )
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--buffer-tokens", type=int, default=8)
    parser.add_argument("--per-chunk-budget", type=int, default=2)
    parser.add_argument("--sentinel-index", type=int, default=5231)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--fading-boosts",
        type=float,
        nargs="+",
        default=[5.0, 2.0, 1.0, 0.5, 0.25, 0.1],
        help="Boost values to sweep in the fading-signal sensitivity test.",
    )
    parser.add_argument(
        "--fading-trials",
        type=int,
        default=20,
        help="Number of random seeds per boost level for the fading-signal test.",
    )
    parser.add_argument(
        "--enable-live-fire",
        action="store_true",
        help="Run the optional live-fire integration test using a pretrained transformer.",
    )
    parser.add_argument(
        "--live-fire-model",
        default="distilbert-base-uncased",
        help="Model name for the live-fire integration test.",
    )
    parser.add_argument(
        "--live-fire-text",
        default=(
            "The Titan vault code is 1234. "
            "Later, security confirmed the Titan vault remains sealed. "
            "Additional context discusses unrelated topics and background information."
        ),
        help="Input text used by the live-fire integration test.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help=f"Path to a text file for the sample data demo (default: {DEFAULT_SAMPLE_FILE}).",
    )
    parser.add_argument(
        "--sample-text",
        type=str,
        default=(
            "The Titan vault code is 1234. Later, security confirmed the Titan vault remains sealed. "
            "Additional context discusses unrelated topics and background information."
        ),
        help="Inline text used for the sample demo when no --sample-file is supplied.",
    )
    parser.add_argument(
        "--sample-needle",
        type=str,
        default="Titan vault code 9876",
        help="Phrase to track within the sample text (defaults to 'Titan vault code 9876').",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional limit on the number of tokens loaded from the sample text.",
    )
    parser.add_argument(
        "--sample-casefold",
        action="store_true",
        help="Enable case-insensitive matching for the sample needle phrase.",
    )
    parser.add_argument(
        "--cluster-adaptive-margin",
        type=float,
        default=0.1,
        help="Adaptive margin for the cluster test (set negative to disable).",
    )
    parser.add_argument(
        "--cluster-max-extra",
        type=int,
        default=8,
        help="Maximum additional tokens a bursty chunk may promote in the cluster test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_params = args.chunk_len * args.buffer_tokens / max(args.seq_len, 1)
    print(
        f"config: seq_len={args.seq_len:,} chunk_len={args.chunk_len:,} "
        f"buffer_tokens={args.buffer_tokens:,} per_chunk_budget={args.per_chunk_budget:,} "
        f"(keep ratio {keep_params:.3f})"
    )
    run_checks(args)
    if args.sample_file is not None or (args.sample_text and args.sample_text.strip()):
        print("\n=== Sample text demo ===")
        _sample_text_demo(args)
    if args.enable_live_fire:
        print("\n=== Live fire integration ===")
        _live_fire_test(args)


if __name__ == "__main__":
    main()
