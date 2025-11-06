#!/usr/bin/env python
# /examples/train_comparison.py
# Compare Genetic Attention (DMoAH) vs Standard Attention on a tiny char dataset.

from proteus_attention.models.dmoah import (
    ModelConfig,
    AdaptiveSparseAttentionBlock,
    AdaptiveSparseAttention,
)
import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error
import math
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add the project root to the Python path to allow running from command line.
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))

# ------------------------------------------------------------------------------
# Repo-relative imports (your layout)
# models/      -> dmoah.py (AdaptiveSparseAttentionBlock, ModelConfig)
# kernels/     -> sparse_attn.py, tinytoy.py  (not required here, but present)
# examples/    -> this script
# ------------------------------------------------------------------------------

# ------------------------------ Utilities -------------------------------------


def get_device():
    """Prefer CUDA (incl. ROCm), otherwise CPU. MPS is unsupported for Genetic Attention/DMoAH Triton kernels."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is not supported by Triton/your kernels; fall back to CPU if that's all there is.
    return torch.device("cpu")


def set_reproducible(seed: int = 1337):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # (Optional) make cuDNN deterministic; can reduce raw throughput slightly.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Prefer tensorcore math on NVIDIA; harmless elsewhere.
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    # TF32 helps speed on NVIDIA; ROCm ignores this.
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

# ------------------------------ Tokenizer & Data ------------------------------


class CharTokenizer:
    """Tiny character-level tokenizer (demo-quality)."""

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class TextDataset(Dataset):
    """Autoregressive next-char dataset with block_size contexts."""

    def __init__(self, text: str, tokenizer: CharTokenizer, block_size: int):
        self.tok = tokenizer
        self.block = block_size
        self.data = torch.tensor(self.tok.encode(text), dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.block)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# ------------------------------ Baseline Model --------------------------------


class StandardAttention(nn.Module):
    """Causal MultiheadAttention wrapper using a bulletproof causal mask."""

    def __init__(self, d_model: int, n_head: int, p_dropout: float, bias: bool):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=p_dropout,
            batch_first=True,
            bias=bias,
        )
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        # x: (B, T, C)
        B, T, _ = x.shape
        # Causal mask: 0 on/below diagonal, -inf above
        causal = torch.full((T, T), float("-inf"), device=x.device)
        causal = torch.triu(causal, diagonal=1)
        out, _ = self.attn(x, x, x, attn_mask=causal, need_weights=False)
        return self.dropout(self.proj(out))


class StandardAdaptiveSparseAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, p_dropout: float, bias: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = StandardAttention(d_model, n_head, p_dropout, bias)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=bias),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=bias),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# ------------------------------ GPT-ish Wrapper -------------------------------


class SimpleGPT(nn.Module):
    """Minimal GPT-ish stack with either Genetic (DMoAH) blocks or standard blocks."""

    def __init__(self, config, model_type: str = "dmoah"):
        super().__init__()
        assert model_type in ("dmoah", "standard")
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.n_ctx, config.d_model)

        blocks = []
        if model_type == "dmoah":
            for _ in range(config.n_layer):
                # from models/dmoah.py
                blocks.append(AdaptiveSparseAttentionBlock(config))
        else:
            for _ in range(config.n_layer):
                blocks.append(
                    StandardAdaptiveSparseAttentionBlock(
                        d_model=config.d_model,
                        n_head=getattr(config, "n_head", 8),
                        p_dropout=config.p_dropout,
                        bias=config.bias,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.token_emb(idx)                                 # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))    # (T, C)
        x = tok + pos

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                                  # (B, T, V)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.n_ctx:]
            logits = self(idx_cond)[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# ------------------------------ Training Loop ---------------------------------


def _iter_model_blocks(root: nn.Module):
    """Yield transformer blocks from possibly wrapped/compiled models."""

    if not isinstance(root, nn.Module):
        return
    stack = [root]
    seen: set[int] = set()
    while stack:
        module = stack.pop()
        if not isinstance(module, nn.Module):
            continue
        key = id(module)
        if key in seen:
            continue
        seen.add(key)
        blocks = getattr(module, "blocks", None)
        if blocks is not None:
            for block in blocks:
                yield block
        for attr in ("module", "_orig_mod"):
            inner = getattr(module, attr, None)
            if isinstance(inner, nn.Module):
                stack.append(inner)


def _iter_dmoah_layers(root: nn.Module):
    """Yield Genetic Attention (DMoAH) layers from the model tree."""

    for block in _iter_model_blocks(root):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        candidates = (
            attn,
            getattr(attn, "module", None),
            getattr(attn, "_orig_mod", None),
        )
        for candidate in candidates:
            if isinstance(candidate, AdaptiveSparseAttention):
                yield candidate
                break


class _HeadStatsAggregator:
    """Lightweight aggregator for Genetic Attention head telemetry across training batches."""

    INTEREST_KEYS = {
        "target_k",
        "max_active_density",
        "mean_gate_prob",
        "router_entropy",
        "unique_heads",
        "mean_active_per_token",
        "max_active_rows",
        "active_fraction",
        "token_keep_fraction",
    }

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._proto_totals: dict[str, float] = {}
        self._proto_counts: dict[str, int] = {}

    def update_from_model(self, model: nn.Module) -> None:
        for block in _iter_model_blocks(model):
            stats = getattr(block, "last_head_stats", None)
            if not isinstance(stats, dict):
                continue
            for key, value in stats.items():
                if key not in self.INTEREST_KEYS and key != "proto":
                    continue
                if key == "proto" and isinstance(value, dict):
                    self._update_nested(
                        value, self._proto_totals, self._proto_counts)
                    continue
                if isinstance(value, (int, float)):
                    self._totals[key] = self._totals.get(
                        key, 0.0) + float(value)
                    self._counts[key] = self._counts.get(key, 0) + 1

    @staticmethod
    def _update_nested(payload: dict[str, float], totals: dict[str, float], counts: dict[str, int]) -> None:
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        if self._totals:
            dense = [
                f"{key}={self.mean(key):.3f}"
                for key in sorted(self._totals)
                if self._counts.get(key, 0)
            ]
            if dense:
                lines.append(" | ".join(dense))
        if self._proto_totals:
            proto = [
                f"proto_{key}={self._proto_totals[key] / max(1, self._proto_counts[key]):.3f}"
                for key in sorted(self._proto_totals)
                if self._proto_counts.get(key, 0)
            ]
            if proto:
                lines.append(" | ".join(proto))
        return lines

    def mean(self, key: str) -> Optional[float]:
        if key in self._totals and self._counts.get(key):
            return self._totals[key] / self._counts[key]
        if key.startswith("proto_"):
            raw = key[len("proto_"):]
            if raw in self._proto_totals and self._proto_counts.get(raw):
                return self._proto_totals[raw] / self._proto_counts[raw]
        return None


class SparseController:
    """Adaptive controller that nudges head budgets toward a density target."""

    def __init__(
        self,
        model: nn.Module,
        *,
        target_density: float,
        tolerance: float,
        patience: int,
        min_heads: Optional[int] = None,
        max_heads: Optional[int] = None,
        dense_threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        self._model = model
        self.target_density = float(max(0.0, min(target_density, 1.0)))
        self.tolerance = max(0.0, float(tolerance))
        self.patience = max(1, int(patience))
        self.verbose = verbose
        self._above = 0
        self._below = 0
        self._global_min = min_heads if (
            min_heads is not None and min_heads > 0) else None
        self._global_max = max_heads if (
            max_heads is not None and max_heads > 0) else None
        self._dense_threshold = None if dense_threshold is None else float(
            max(0.0, min(dense_threshold, 1.0)))
        self._current_min, self._current_max = self._snapshot_bounds()
        if self._dense_threshold is not None and self.verbose:
            print(
                f"  [SparseCtrl] force_dense_threshold set to {self._dense_threshold:.3f}")
        # Apply initial overrides if requested.
        if self._global_min is not None or self._global_max is not None:
            self._apply_bounds(self._global_min, self._global_max)
        elif self._dense_threshold is not None:
            self._apply_bounds(None, None)
        if self.verbose:
            print(
                f"  [SparseCtrl] target_density={self.target_density:.3f}, "
                f"tolerance={self.tolerance:.3f}, patience={self.patience}"
            )

    def _snapshot_bounds(self) -> tuple[Optional[int], Optional[int]]:
        layers = list(_iter_dmoah_layers(self._model))
        if not layers:
            return None, None
        return layers[0].h_active_min, layers[0].h_active_max

    def _apply_bounds(self, min_heads: Optional[int], max_heads: Optional[int]) -> bool:
        layers = list(_iter_dmoah_layers(self._model))
        if not layers:
            return False
        changed = False
        for layer in layers:
            if min_heads is not None:
                new_min = max(1, min(min_heads, layer.h_total))
                if new_min != layer.h_active_min:
                    changed = True
                layer.h_active_min = new_min
            if max_heads is not None:
                new_max = max(layer.h_active_min, min(
                    max_heads, layer.h_total))
                if new_max != layer.h_active_max:
                    changed = True
                layer.h_active_max = new_max
            layer.h_active = max(layer.h_active_min, min(
                layer.h_active, layer.h_active_max))
            if self._dense_threshold is not None:
                layer._dense_threshold = float(self._dense_threshold)
        if layers:
            self._current_min = layers[0].h_active_min
            self._current_max = layers[0].h_active_max
        if changed and self.verbose:
            print(
                f"  [SparseCtrl] Head bounds -> min={self._current_min} max={self._current_max}")
        return changed

    def maybe_adjust(
        self,
        density: Optional[float],
        epoch: int,
        stats: Optional[_HeadStatsAggregator] = None,
    ) -> None:
        if density is None or math.isnan(density):
            self._above = 0
            self._below = 0
            return

        upper = self.target_density + self.tolerance
        lower = max(0.0, self.target_density - self.tolerance)
        mean_target = stats.mean("target_k") if stats is not None else None

        if density > upper:
            self._above += 1
            self._below = 0
            if self._above >= self.patience:
                self._above = 0
                self._adjust_active_heads(-1, epoch, mean_target, density)
        elif density < lower:
            self._below += 1
            self._above = 0
            if self._below >= self.patience:
                self._below = 0
                self._adjust_active_heads(+1, epoch, mean_target, density)
        else:
            self._above = 0
            self._below = 0

    def _adjust_active_heads(
        self,
        delta: int,
        epoch: int,
        mean_target_k: Optional[float],
        density: float,
    ) -> None:
        layers = list(_iter_dmoah_layers(self._model))
        if not layers:
            return
        current_min = layers[0].h_active_min
        current_max = layers[0].h_active_max
        cap = min(layer.h_total for layer in layers)
        new_max = max(1, min(current_max + delta, cap))
        if self._global_max is not None:
            new_max = min(new_max, self._global_max)
        if new_max < 1 or new_max == current_max:
            return

        if self._global_min is not None:
            new_min = max(1, min(self._global_min, new_max))
        else:
            new_min = max(1, min(current_min, new_max))
            if new_min > new_max:
                new_min = new_max

        min_param = new_min if new_min != current_min else None
        max_param = new_max if new_max != current_max else None
        if min_param is None and max_param is None:
            return

        if self._apply_bounds(min_param, max_param):
            direction = "high" if delta < 0 else "low"
            if self.verbose:
                msg = (
                    f"  [SparseCtrl] Epoch {epoch}: density {direction} ({density:.3f}) -> "
                    f"max heads {current_max} → {self._current_max}"
                )
                if mean_target_k is not None:
                    msg += f", avg target_k {mean_target_k:.2f}"
                print(msg)


@dataclass
class TrainResult:
    steps: int
    avg_loss: Optional[float]
    elapsed: float
    epochs_completed: int


def train_model(model: nn.Module,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epochs: int,
                model_name: str,
                use_amp: bool = True,
                grad_clip: float = 1.0,
                log_head_stats: bool = True,
                sparse_controller: Optional[SparseController] = None,
                max_steps: Optional[int] = None) -> TrainResult:
    """Mixed-precision friendly trainer with optional Genetic Attention telemetry/control."""
    use_cuda_amp = bool(use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(
        enabled=use_cuda_amp) if use_cuda_amp else None
    if use_cuda_amp:
        def _autocast():
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        _autocast = nullcontext
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()

    print(f"\n--- Training {model_name} ---")
    t0 = time.time()

    global_step = 0
    last_epoch_avg: Optional[float] = None
    reached_limit = False
    epochs_completed = 0

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        stats_agg = _HeadStatsAggregator() if log_head_stats else None
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(
                device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast():
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            if stats_agg is not None:
                stats_agg.update_from_model(model)

            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            global_step += 1
            total_loss += loss.item()
            n_batches += 1

            if i % 200 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | Batch {i} | Loss {loss.item():.4f}")

            if max_steps is not None and global_step >= max_steps:
                reached_limit = True
                break

        avg = total_loss / max(1, n_batches)
        last_epoch_avg = avg
        print(f"Epoch {epoch}/{epochs} | Avg Loss {avg:.4f}")
        if stats_agg is not None:
            for line in stats_agg.summary_lines():
                print(f"  [GenAttn] {line}")
            if sparse_controller is not None:
                avg_density = stats_agg.mean("max_active_density")
                sparse_controller.maybe_adjust(avg_density, epoch, stats_agg)
        elif sparse_controller is not None:
            sparse_controller.maybe_adjust(None, epoch, None)

        epochs_completed = epoch

        if reached_limit:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.2f}s ({global_step} steps).")
    if last_epoch_avg is not None:
        print(
            f"{model_name} summary: steps={global_step}, avg_loss={last_epoch_avg:.4f}")
    return TrainResult(
        steps=global_step,
        avg_loss=last_epoch_avg,
        elapsed=elapsed,
        epochs_completed=epochs_completed,
    )


@torch.no_grad()
def generate_text(model: nn.Module,
                  tokenizer: CharTokenizer,
                  device: torch.device,
                  prompt: str = "Hello",
                  num_tokens: int = 200):
    model.eval()
    start = torch.tensor(tokenizer.encode(
        prompt), dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(start, max_new_tokens=num_tokens)
    txt = tokenizer.decode(out[0].tolist())
    print("-" * 80)
    print(txt)
    print("-" * 80)

# --------------------------------- Main ---------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare Genetic Attention (DMoAH) vs Standard Attention.")
    parser.add_argument("text_file", nargs="?", default="input.txt",
                        help="Path to training text (default: input.txt; downloads Tiny Shakespeare if missing).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--compile", action="store_false", dest="no_compile",
                        help="Enable torch.compile if available (may be unstable with custom models).")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP even on CUDA.")
    parser.add_argument("--gen_tokens", type=int, default=200,
                        help="Tokens to generate after training each model.")
    parser.add_argument("--target_density", type=float, default=0.28,
                        help="Desired average max_active_density for Genetic Attention (DMoAH) (0-1).")
    parser.add_argument("--density_tol", type=float, default=0.03,
                        help="Tolerance band around target density before adjustments.")
    parser.add_argument("--density_patience", type=int, default=1,
                        help="Epochs the density can stay out-of-band before adjusting heads.")
    parser.add_argument("--min_active_heads", type=int, default=0,
                        help="Override minimum active heads (0 keeps config default).")
    parser.add_argument("--max_active_heads", type=int, default=0,
                        help="Override maximum active heads (0 keeps config default).")
    parser.add_argument("--dense_threshold", type=float, default=0.30,
                        help="Force-dense threshold passed to Genetic Attention (DMoAH) (0-1).")
    parser.add_argument("--no_sparse_ctrl", action="store_true",
                        help="Disable adaptive sparse controller adjustments.")
    parser.add_argument("--token_sparse", action="store_true",
                        help="Enable token-level sparsity routing.")
    parser.add_argument("--token_keep_ratio", type=float, default=0.85,
                        help="Fraction of tokens to keep when token sparsity is enabled (0-1).")
    parser.add_argument("--token_keep_min", type=int, default=8,
                        help="Minimum tokens to keep per sequence when token sparsity is enabled.")
    parser.add_argument("--token_keep_threshold", type=float, default=0.0,
                        help="Proto/importance threshold for keeping tokens.")
    parser.add_argument("--token_keep_guard", type=int, default=1,
                        help="Always keep the first N tokens in each sequence.")
    parser.add_argument("--genetic_steps", type=int, default=0,
                        help="Maximum training steps for the Genetic Attention run (0 uses all batches).")
    parser.add_argument("--dmoah_steps", type=int,
                        default=0, help=argparse.SUPPRESS)
    parser.add_argument("--standard_steps", type=int, default=0,
                        help="Maximum training steps for the standard baseline (0 uses all batches).")
    parser.add_argument("--summary_json", type=str, default="",
                        help="Optional path to append JSON summary of this comparison.")
    parser.add_argument("--run_label", type=str, default="",
                        help="Identifier label for the summary entry (defaults to context/batch info).")
    args = parser.parse_args()

    device = get_device()
    set_reproducible(1337)

    print(f"Using device: {device}")
    if device.type == "cpu" and torch.backends.mps.is_available():
        print("Note: MPS detected but DMoAH/Triton path is unsupported on MPS. Running on CPU.")

    # ---------- Data ----------
    data_path = Path(args.text_file)
    if not data_path.exists():
        print(f"'{data_path}' not found. Downloading Tiny Shakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        try:
            urllib.request.urlretrieve(url, data_path)
            print("Download complete.")
        except (urllib.error.URLError, OSError) as exc:
            print(f"Download failed ({exc}). Using built-in fallback corpus.")
            fallback = (
                "ROMEO:\n"
                "But, soft! what light through yonder window breaks?\n"
                "JULIET:\n"
                "O Romeo, Romeo! wherefore art thou Romeo?\n"
                "Nurse:\n"
                "I think it best you married with the county.\n"
            )
            data_path.write_text(fallback, encoding="utf-8")

    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    dataset = TextDataset(text, tokenizer, args.block_size)

    # DataLoader tuned for GPU transfers
    num_workers = min(4, os.cpu_count() or 1)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=(device.type == "cuda" and num_workers > 0))

    vocab = tokenizer.vocab_size
    print(
        f"Corpus size: {len(text):,} chars | Vocab: {vocab} | Block size: {args.block_size}")

    target_density = max(0.0, min(args.target_density, 1.0))
    density_tol = max(0.0, args.density_tol)
    dense_threshold = max(0.0, min(args.dense_threshold, 1.0))
    min_active = args.min_active_heads if args.min_active_heads > 0 else 2
    max_active = args.max_active_heads if args.max_active_heads > 0 else 4
    if max_active < min_active:
        max_active = min_active
    token_sparse = bool(args.token_sparse)
    token_keep_ratio = max(
        0.0, min(args.token_keep_ratio, 1.0)) if token_sparse else 1.0
    token_keep_min = max(0, args.token_keep_min if token_sparse else 0)
    token_keep_threshold = max(
        0.0, args.token_keep_threshold if token_sparse else 0.0)
    token_keep_guard = max(0, args.token_keep_guard if token_sparse else 1)

    # ---------- Configs ----------
    # DMoAH config mirrors your tinytoy defaults, tuned to small contexts.
    genetic_steps = args.genetic_steps if args.genetic_steps > 0 else args.dmoah_steps

    dmoah_cfg = ModelConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_ctx=args.block_size,        # used by DMoAH internals (masks/curves)
        vocab_size=vocab,
        p_dropout=0.1,
        bias=False,
        use_sdpa=True,
        # DMoAH specifics
        attn_h_total=8,
        attn_h_active=max_active,
        attn_h_active_min=min_active,
        attn_h_active_max=max_active,
        attn_active_seq_low=args.block_size // 2,
        attn_active_seq_high=args.block_size,
        attn_small_seq_dense=args.block_size // 2,
        attn_force_dense_threshold=dense_threshold,  # smooth the mid-seq wobble
        attn_gates=16,
        attn_proto_enable=True,
        attn_quantize_int8=True,          # enable int8 path you benchmarked
        attn_token_sparse=token_sparse,
        attn_token_keep_ratio=token_keep_ratio,
        attn_token_keep_min=token_keep_min,
        attn_token_keep_threshold=token_keep_threshold,
        attn_token_keep_guard=token_keep_guard,
    )
    dmoah_cfg.block_size = args.block_size

    # Baseline config (SimpleNamespace-like)
    class StdCfg:
        pass
    std_cfg = StdCfg()
    std_cfg.d_model = args.d_model
    std_cfg.n_layer = args.n_layer
    std_cfg.block_size = args.block_size
    std_cfg.n_ctx = args.block_size  # for positional embeddings
    std_cfg.vocab_size = vocab
    std_cfg.p_dropout = 0.1
    std_cfg.bias = False
    std_cfg.n_head = 8  # match attn_h_total for fairness

    # ---------- Models ----------
    model_dmoah = SimpleGPT(dmoah_cfg, model_type="dmoah")
    model_std = SimpleGPT(std_cfg, model_type="standard")

    # Optional torch.compile (great on CUDA/ROCm; safe to skip on CPU)
    if hasattr(torch, "compile") and not args.no_compile:
        try:
            model_dmoah = torch.compile(model_dmoah, mode="max-autotune")
            model_std = torch.compile(model_std, mode="max-autotune")
            print("Models compiled with torch.compile()")
        except Exception as e:
            print(f"torch.compile failed, continuing without it: {e}")

    sparse_ctrl = None
    if not args.no_sparse_ctrl:
        sparse_ctrl = SparseController(
            model_dmoah,
            target_density=target_density,
            tolerance=density_tol,
            patience=max(1, args.density_patience),
            min_heads=args.min_active_heads if args.min_active_heads > 0 else None,
            max_heads=args.max_active_heads if args.max_active_heads > 0 else None,
            dense_threshold=dense_threshold,
        )

    # ---------- Optimizers ----------
    opt_dmoah = torch.optim.AdamW(model_dmoah.parameters(), lr=args.lr)
    opt_std = torch.optim.AdamW(model_std.parameters(), lr=args.lr)

    # ---------- Train DMoAH ----------
    result_dmoah = train_model(
        model_dmoah,
        loader,
        opt_dmoah,
        device,
        epochs=args.epochs,
        model_name="Genetic Attention (DMoAH)",
        use_amp=(not args.no_amp),
        log_head_stats=True,
        sparse_controller=sparse_ctrl,
        max_steps=genetic_steps if genetic_steps > 0 else None,
    )

    if args.gen_tokens > 0:
        print("\n--- DMoAH Generation ---")
        generate_text(
            model_dmoah,
            tokenizer,
            device,
            prompt="JULIET:\nO Romeo, Romeo! wherefore art thou",
            num_tokens=args.gen_tokens,
        )

    # ---------- Train Standard ----------
    result_standard = train_model(
        model_std,
        loader,
        opt_std,
        device,
        epochs=args.epochs,
        model_name="Standard Attention",
        use_amp=(not args.no_amp),
        log_head_stats=False,
        max_steps=args.standard_steps if args.standard_steps > 0 else None,
    )

    if args.gen_tokens > 0:
        print("\n--- Standard Generation ---")
        generate_text(
            model_std,
            tokenizer,
            device,
            prompt="JULIET:\nO Romeo, Romeo! wherefore art thou",
            num_tokens=args.gen_tokens,
        )

    run_label = args.run_label.strip(
    ) or f"ctx{args.block_size}_bs{args.batch_size}_d{args.d_model}_L{args.n_layer}"

    def _result_summary(result: TrainResult) -> dict[str, float | int | None]:
        steps_per_sec = result.steps / result.elapsed if result.elapsed > 0 else None
        return {
            "steps": result.steps,
            "avg_loss": result.avg_loss,
            "elapsed_sec": result.elapsed,
            "steps_per_sec": steps_per_sec,
            "epochs_completed": result.epochs_completed,
        }

    summary_payload = {
        "label": run_label,
        "device": str(device),
        "config": {
            "block_size": args.block_size,
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "n_layer": args.n_layer,
            "epochs": args.epochs,
            "genetic_steps_cap": genetic_steps if genetic_steps > 0 else None,
            "standard_steps_cap": args.standard_steps if args.standard_steps > 0 else None,
            "token_sparse": bool(args.token_sparse),
        },
        "models": {
            "dmoah": _result_summary(result_dmoah),
            "standard": _result_summary(result_standard),
        },
        "comparisons": {},
    }

    d_elapsed = summary_payload["models"]["dmoah"]["elapsed_sec"]
    s_elapsed = summary_payload["models"]["standard"]["elapsed_sec"]
    d_loss = summary_payload["models"]["dmoah"]["avg_loss"]
    s_loss = summary_payload["models"]["standard"]["avg_loss"]
    comp = {}
    if isinstance(d_elapsed, (int, float)) and isinstance(s_elapsed, (int, float)) and d_elapsed > 0:
        comp["standard_over_genetic_speedup"] = s_elapsed / d_elapsed
        comp["genetic_over_standard_speedup"] = d_elapsed / \
            s_elapsed if s_elapsed > 0 else None
    if isinstance(d_loss, (int, float)) and isinstance(s_loss, (int, float)):
        comp["loss_delta_genetic_minus_standard"] = d_loss - s_loss
    summary_payload["comparisons"] = comp

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = []
        if summary_path.exists():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    existing = data
                elif isinstance(data, dict):
                    existing = [data]
            except json.JSONDecodeError:
                print(
                    f"Warning: could not parse existing JSON at {summary_path}; starting fresh.")
        existing = [entry for entry in existing if entry.get(
            "label") != run_label]
        existing.append(summary_payload)
        summary_path.write_text(json.dumps(
            existing, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Run summary written to {summary_path}")

    print("\nComparison complete. Inspect loss trends and sample generations for qualitative differences.")


if __name__ == "__main__":
    main()
