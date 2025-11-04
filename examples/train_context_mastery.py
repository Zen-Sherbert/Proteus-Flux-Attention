#!/usr/bin/env python
"""
Adaptive context mastery trainer.

Gradually increases the Flux alpha (and optional sequence length) when loss
improvements plateau, pushing the model toward the linear sparse regime.
The trainer now supports mixed-precision, validation-driven adaptive jumps,
and gradient accumulation so longer contexts remain stable as they scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":  # pragma: no cover - direct execution
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import examples.common as common  # type: ignore
    from examples.modeling import MiniProteusLM, ModelConfig  # type: ignore
else:  # pragma: no cover - package execution
    from . import common
    from .modeling import MiniProteusLM, ModelConfig


def run_training(
    *,
    steps: int,
    base_seq_len: int,
    max_seq_len: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    alpha_start: float,
    alpha_step: float,
    plateau_window: int,
    plateau_tol: float,
    plateau_patience: int,
    plateau_cooldown: int,
    grad_accum_steps: int = 1,
    val_fraction: float = 0.1,
    val_iters: int = 2,
    use_amp: bool = True,
    seq_growth_factor: float = 1.5,
    warmup_steps: int = 0,
    data_path: Optional[Path],
    prompt: str,
    sample_tokens: int,
) -> dict:
    tokenizer = common.get_tokenizer()
    text = common.load_corpus(data_path)
    tokens = tokenizer.encode(text)

    vocab_size = getattr(tokenizer, "n_vocab", 256)
    train_cfg = common.TrainingConfig(
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        device=str(device),
    )
    model_cfg = ModelConfig(
        vocab_size=train_cfg.vocab_size,
        embed_dim=train_cfg.embed_dim,
        num_heads=train_cfg.num_heads,
        num_layers=train_cfg.num_layers,
        dim_feedforward=train_cfg.dim_feedforward,
        dropout=train_cfg.dropout,
        max_seq_len=train_cfg.max_seq_len,
    )
    ckpt_dir = common.ensure_checkpoint_dir("context_mastery")
    amp_enabled = bool(use_amp and device.type == "cuda")
    grad_accum_steps = max(1, int(grad_accum_steps))
    seq_growth_factor = max(1.0, float(seq_growth_factor))
    val_fraction = float(max(0.0, min(0.5, val_fraction)))
    val_iters = max(1, int(val_iters)) if val_fraction > 0.0 else 0
    warmup_steps = max(0, int(warmup_steps))

    min_stride = base_seq_len + 1
    total_tokens = len(tokens)
    if (
        val_fraction > 0.0
        and total_tokens >= min_stride * 2
    ):
        split_point = int(total_tokens * (1.0 - val_fraction))
        split_point = max(min_stride, min(split_point, total_tokens - min_stride))
        train_tokens = tokens[:split_point]
        val_tokens = tokens[split_point:]
    else:
        train_tokens = tokens
        window = max(min_stride, min(total_tokens, max_seq_len))
        val_tokens = tokens[-window:] if total_tokens > 0 else tokens

    model = MiniProteusLM(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    total_sched_steps = max(1, steps)
    if warmup_steps > 0 and warmup_steps < total_sched_steps:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, total_sched_steps - warmup_steps),
            eta_min=lr * 0.1,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_sched_steps, eta_min=lr * 0.1
        )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    alpha = float(max(0.0, min(1.0, alpha_start)))
    model.set_flux_alpha(alpha)
    seq_len = base_seq_len

    def _build_train_batches(current_seq_len: int):
        return common.build_batches(train_tokens, current_seq_len, batch_size, device=device)

    train_batches = _build_train_batches(seq_len)

    history: list[dict[str, float | int | str | None]] = []
    recent_metrics: list[float] = []
    prev_metric_avg: Optional[float] = None
    plateau_hits = 0
    last_adjust_step = 0
    best_val_loss = float("inf")
    best_checkpoint_path: Optional[Path] = None
    train_loss_baseline: Optional[float] = None
    val_loss_baseline: Optional[float] = None

    def _run_validation(current_seq_len: int) -> Optional[tuple[float, float]]:
        if val_fraction <= 0.0 or val_iters <= 0:
            return None
        val_batches = common.build_batches(
            val_tokens, current_seq_len, batch_size, device=device
        )
        losses: list[float] = []
        model.eval()
        with torch.inference_mode():
            for _ in range(val_iters):
                vx, vy = next(val_batches)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    val_logits = model(vx)
                    val_loss = F.cross_entropy(val_logits.view(-1, vocab_size), vy.view(-1))
                losses.append(float(val_loss.item()))
        model.train()
        if not losses:
            return None
        avg_loss = sum(losses) / len(losses)
        return avg_loss, float(math.exp(avg_loss))

    model.train()
    for step in range(1, steps + 1):
        if seq_len > model_cfg.max_seq_len:
            raise ValueError("Sequence length exceeded model maximum during training.")
        opt.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _ in range(grad_accum_steps):
            x, y = next(train_batches)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            micro_losses.append(float(loss.item()))
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        loss_value = float(sum(micro_losses) / len(micro_losses))
        ppl = float(math.exp(loss_value))
        current_lr = opt.param_groups[0]["lr"]

        val_result = _run_validation(seq_len)
        if val_result is not None:
            val_loss, val_ppl = val_result
        else:
            val_loss = None
            val_ppl = None
        if train_loss_baseline is None:
            train_loss_baseline = max(loss_value, 1e-6)
        train_norm = loss_value / train_loss_baseline
        metric_components = [train_norm]
        if val_loss is not None:
            if val_loss_baseline is None:
                val_loss_baseline = max(val_loss, 1e-6)
            val_norm = val_loss / val_loss_baseline
            metric_components.append(val_norm)

        record = {
            "step": step,
            "alpha": alpha,
            "seq_len": seq_len,
            "loss": loss_value,
            "ppl": ppl,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "lr": current_lr,
        }
        history.append(record)

        if step % max(1, steps // 10) == 0 or step == 1:
            log_parts = [
                f"[context-mastery] step={step}/{steps}",
                f"seq_len={seq_len}",
                f"alpha={alpha:.2f}",
                f"loss={loss_value:.4f}",
                f"ppl={ppl:.2f}",
                f"lr={current_lr:.2e}",
            ]
            if val_loss is not None:
                log_parts.append(f"val_loss={val_loss:.4f}")
                log_parts.append(f"val_ppl={val_ppl:.2f}")
            print(" ".join(log_parts))

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "model_cfg": asdict(model_cfg),
                    "training_cfg": asdict(train_cfg),
                    "history": history,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                best_checkpoint_path,
            )

        metric_for_plateau = sum(metric_components) / len(metric_components)
        recent_metrics.append(metric_for_plateau)
        if len(recent_metrics) >= plateau_window:
            window_avg = sum(recent_metrics[-plateau_window:]) / plateau_window
            if prev_metric_avg is not None:
                improvement = (prev_metric_avg - window_avg) / max(prev_metric_avg, 1e-6)
                if improvement < plateau_tol:
                    plateau_hits += 1
                    if (
                        plateau_hits >= plateau_patience
                        and (step - last_adjust_step) >= plateau_cooldown
                    ):
                        if alpha < 1.0:
                            alpha = min(1.0, alpha + alpha_step)
                            model.set_flux_alpha(alpha)
                        if seq_len < max_seq_len:
                            proposed = int(seq_len * seq_growth_factor)
                            if proposed <= seq_len:
                                proposed = seq_len + max(1, base_seq_len // 4 or 1)
                            seq_len = min(max_seq_len, proposed)
                            train_batches = _build_train_batches(seq_len)
                        last_adjust_step = step
                        plateau_hits = 0
                        recent_metrics.clear()
                        prev_metric_avg = None
                        print(
                            f"[context-mastery] plateau detected. "
                            f"new alpha={alpha:.2f}, seq_len={seq_len}"
                        )
                        continue
                else:
                    plateau_hits = 0
            prev_metric_avg = window_avg

    model.eval()
    with torch.inference_mode():
        encoded_prompt = tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
        out_tokens = model.generate(input_ids, max_new_tokens=sample_tokens)
        completion = tokenizer.decode(out_tokens[0].tolist())

    ckpt_path = ckpt_dir / "model.pt"
    torch.save(
        {
            "model_cfg": asdict(model_cfg),
            "training_cfg": asdict(train_cfg),
            "history": history,
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    history_csv_path = ckpt_dir / "history.csv"
    fieldnames = ["step", "alpha", "seq_len", "loss", "ppl", "val_loss", "val_ppl", "lr"]
    with history_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    sample_path = ckpt_dir / "sample.txt"
    sample_path.write_text(completion, encoding="utf-8")
    print(f"[context-mastery] saved checkpoint to {ckpt_path}")
    if best_checkpoint_path is not None:
        print(f"[context-mastery] best checkpoint updated at {best_checkpoint_path}")
    print(f"[context-mastery] sample:\n{completion}")
    return {
        "checkpoint": str(ckpt_path),
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        "history": history,
        "sample": str(sample_path),
        "history_csv": str(history_csv_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive context mastery trainer.")
    parser.add_argument("--steps", type=int, default=1200, help="Total optimization steps.")
    parser.add_argument("--base-seq-len", type=int, default=256, help="Initial sequence length.")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--device", default=None, help="Device string (default auto).")
    parser.add_argument("--data", type=Path, default=None, help="Optional training text.")
    parser.add_argument("--alpha-start", type=float, default=0.0, help="Initial Flux alpha.")
    parser.add_argument("--alpha-step", type=float, default=0.1, help="Increment applied on plateau.")
    parser.add_argument("--plateau-window", type=int, default=320, help="Steps per plateau window.")
    parser.add_argument("--plateau-tol", type=float, default=0.01, help="Relative improvement threshold.")
    parser.add_argument("--plateau-patience", type=int, default=2, help="Number of consecutive plateau windows before adjustment.")
    parser.add_argument("--plateau-cooldown", type=int, default=150, help="Minimum steps between adjustments.")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Micro-batches to accumulate before each optimizer step.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of tokens reserved for validation (0 disables validation).")
    parser.add_argument("--val-iters", type=int, default=2, help="Validation batches sampled each step.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--seq-growth-factor", type=float, default=1.25, help="Multiplier applied to sequence length when plateaus occur.")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Number of linear warmup steps before cosine decay.")
    parser.add_argument("--prompt", default="Proteus evolves", help="Prompt for generation.")
    parser.add_argument("--sample-tokens", type=int, default=160, help="Tokens to generate.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    run_training(
        steps=args.steps,
        base_seq_len=args.base_seq_len,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        alpha_start=args.alpha_start,
        alpha_step=args.alpha_step,
        plateau_window=args.plateau_window,
        plateau_tol=args.plateau_tol,
        plateau_patience=args.plateau_patience,
        plateau_cooldown=args.plateau_cooldown,
        grad_accum_steps=args.grad_accum_steps,
        val_fraction=args.val_fraction,
        val_iters=args.val_iters,
        use_amp=not args.no_amp,
        seq_growth_factor=args.seq_growth_factor,
        warmup_steps=args.warmup_steps,
        data_path=args.data,
        prompt=args.prompt,
        sample_tokens=args.sample_tokens,
    )


if __name__ == "__main__":
    main()
