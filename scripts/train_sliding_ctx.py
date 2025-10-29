#!/usr/bin/env python
"""
Sliding-context training run for DMoAH.

The run progresses through short, mid, and long context windows so we can
evaluate how the model adapts as the receptive field expands.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
import torch.nn.functional as F

from protean_forge.examples.dmoah_train import (  # type: ignore[attr-defined]
    CharTokenizer,
    SimpleGPT,
    get_device,
    set_reproducible,
)
from protean_forge.kernels.sparse_attn import get_last_backend_info
from protean_forge.models.dmoah import ModelConfig


def _load_corpus(path: Path, limit: int | None = None) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if limit is not None and limit > 0:
        return text[:limit]
    return text


def _build_data_tensor(tokenizer: CharTokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        raise RuntimeError("Corpus too small after tokenisation.")
    return torch.tensor(ids, dtype=torch.long)


def _sample_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if block_size + 1 >= data.size(0):
        raise RuntimeError(
            f"Block size {block_size} exceeds available tokens ({data.size(0)})."
        )
    max_start = data.size(0) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    inputs = []
    targets = []
    for start in starts.tolist():
        slice_x = data[start : start + block_size]
        slice_y = data[start + 1 : start + block_size + 1]
        inputs.append(slice_x)
        targets.append(slice_y)
    x = torch.stack(inputs, dim=0)
    y = torch.stack(targets, dim=0)
    return x, y


def _phase_schedule(total_steps: int, short_pct: float, mid_pct: float) -> Tuple[int, int, int]:
    short_steps = int(total_steps * short_pct)
    mid_steps = int(total_steps * mid_pct)
    long_steps = total_steps - short_steps - mid_steps
    if short_steps <= 0 or mid_steps <= 0 or long_steps <= 0:
        raise RuntimeError("Invalid schedule percentages; each phase must receive >0 steps.")
    return short_steps, mid_steps, long_steps


def train_schedule(
    model: SimpleGPT,
    data: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    phases: List[dict],
    total_steps: int,
    log_interval: int = 200,
    max_grad_norm: float = 1.0,
) -> Tuple[List[dict], int]:
    amp_enabled = device.type == "cuda"
    if amp_enabled:
        scaler = None
        amp_module = getattr(torch, "amp", None)
        if amp_module is not None:
            GradScaler = getattr(amp_module, "GradScaler", None)
            if GradScaler is not None:
                for args in (("cuda",), tuple()):
                    try:
                        scaler = GradScaler(*args, enabled=True)
                        break
                    except TypeError:
                        try:
                            scaler = GradScaler(*args)
                            break
                        except TypeError:
                            scaler = None
        if scaler is None:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    data = data.to("cpu")
    global_step = 0
    phase_logs: List[dict] = []
    model.to(device)
    model.train()

    for phase_idx, phase in enumerate(phases):
        steps = int(phase["steps"])
        block = int(phase["block_size"])
        batch_size = int(phase["batch_size"])
        label = str(phase["name"])
        losses: List[float] = []
        trace: List[dict] = []
        start_time = time.time()

        for local_step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            x_cpu, y_cpu = _sample_batch(data, block, batch_size)
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)

            if amp_enabled:
                amp_module = getattr(torch, "amp", None)
                if amp_module is not None and hasattr(amp_module, "autocast"):
                    context_mgr = amp_module.autocast("cuda", dtype=torch.bfloat16)  # type: ignore[attr-defined]
                else:
                    context_mgr = torch.cuda.amp.autocast(dtype=torch.bfloat16)  # type: ignore[attr-defined]
            else:
                context_mgr = nullcontext()

            with context_mgr:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            loss_val = float(loss.detach().cpu())
            losses.append(loss_val)
            global_step += 1

            if (
                (local_step + 1) % log_interval == 0
                or local_step == 0
                or (local_step + 1) == steps
            ):
                recent = losses[-min(log_interval, len(losses)) :]
                recent_mean = sum(recent) / len(recent)
                print(
                    f"[Phase {phase_idx+1}/{len(phases)}:{label}] "
                    f"step {local_step+1}/{steps} | ctx={block} | "
                    f"batch={batch_size} | loss={loss_val:.4f} | recent_avg={recent_mean:.4f}"
                )
                trace.append(
                    {
                        "global_step": global_step,
                        "phase_step": local_step + 1,
                        "loss": loss_val,
                        "recent_avg": recent_mean,
                    }
                )

        duration = time.time() - start_time
        backend_info = get_last_backend_info()
        mean_loss = sum(losses) / max(1, len(losses))
        phase_logs.append(
            {
                "name": label,
                "block_size": block,
                "batch_size": batch_size,
                "steps": steps,
                "mean_loss": mean_loss,
                "final_loss": losses[-1] if losses else None,
                "duration_s": duration,
                "backend": backend_info,
                "loss_trace": trace,
            }
        )

    return phase_logs, global_step


def parse_args() -> argparse.Namespace:
    default_dataset = (
        Path("/var/mnt/SDD/Data/text-english-code-fiction-nonfiction/fiction_100mb.txt")
    )
    parser = argparse.ArgumentParser(description="Sliding-context DMoAH training run.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Path to training corpus (default: fiction_100mb sample).",
    )
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--short-ctx", type=int, default=512)
    parser.add_argument("--mid-ctx", type=int, default=4096)
    parser.add_argument("--long-ctx", type=int, default=16384)
    parser.add_argument("--short-pct", type=float, default=0.70)
    parser.add_argument("--mid-pct", type=float, default=0.20)
    parser.add_argument("--short-batch", type=int, default=32)
    parser.add_argument("--mid-batch", type=int, default=12)
    parser.add_argument("--long-batch", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/sliding_ctx"))
    parser.add_argument("--corpus-limit", type=int, default=0, help="Optionally truncate the corpus to this many characters for faster experiments.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    set_reproducible(1337)

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    text = _load_corpus(args.dataset, limit=args.corpus_limit or None)
    tokenizer = CharTokenizer(text)
    data_tensor = _build_data_tensor(tokenizer, text)

    short_steps, mid_steps, long_steps = _phase_schedule(
        args.total_steps,
        args.short_pct,
        args.mid_pct,
    )

    long_ctx = max(args.short_ctx, args.mid_ctx, args.long_ctx)
    vocab = tokenizer.vocab_size
    print(
        f"Corpus chars: {len(text):,} | vocab size: {vocab} | "
        f"Total steps: {args.total_steps} | Device: {device}"
    )

    h_total = max(1, args.n_head)
    h_active_min = max(1, min(h_total, 2))
    h_active_target = max(h_active_min, min(h_total, 4))
    h_active_max = max(h_active_target, min(h_total, 6))

    config = ModelConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_ctx=long_ctx,
        vocab_size=vocab,
        p_dropout=0.1,
        bias=False,
        use_sdpa=True,
        attn_h_total=h_total,
        attn_h_active=h_active_target,
        attn_h_active_min=h_active_min,
        attn_h_active_max=h_active_max,
        attn_active_seq_low=args.short_ctx,
        attn_active_seq_high=long_ctx,
        attn_small_seq_dense=args.short_ctx // 2,
        attn_force_dense_threshold=0.25,
        attn_gates=16,
        attn_dna_enable=True,
        attn_quantize_int8=True,
        attn_token_sparse=True,
        attn_token_keep_ratio=0.85,
        attn_token_keep_min=16,
        attn_token_keep_threshold=0.0,
        attn_token_keep_guard=8,
        attn_linear_switch_ctx=long_ctx,
    )
    config.block_size = long_ctx

    model = SimpleGPT(config, model_type="dmoah")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    phases = [
        {
            "name": "short",
            "steps": short_steps,
            "block_size": args.short_ctx,
            "batch_size": args.short_batch,
        },
        {
            "name": "mid",
            "steps": mid_steps,
            "block_size": args.mid_ctx,
            "batch_size": args.mid_batch,
        },
        {
            "name": "long",
            "steps": long_steps,
            "block_size": args.long_ctx,
            "batch_size": args.long_batch,
        },
    ]

    t_start = time.time()
    phase_logs, consumed_steps = train_schedule(
        model=model,
        data=data_tensor,
        device=device,
        optimizer=optimizer,
        phases=phases,
        total_steps=args.total_steps,
        log_interval=args.log_interval,
    )
    total_elapsed = time.time() - t_start

    summary = {
        "dataset": str(args.dataset),
        "total_chars": len(text),
        "vocab_size": vocab,
        "device": str(device),
        "model_config": config.to_dict(),
        "phases": phase_logs,
        "total_steps_requested": args.total_steps,
        "total_steps_completed": consumed_steps,
        "total_time_s": total_elapsed,
        "optimizer": {"type": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
        "log_interval": args.log_interval,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    summary_path = output_dir / f"sliding_ctx_run_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
