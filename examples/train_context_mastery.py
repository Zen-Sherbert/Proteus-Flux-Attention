#!/usr/bin/env python
"""
Adaptive context mastery trainer.

Gradually increases the Shortlist alpha (and optional sequence length) when loss
improvements plateau, pushing the model toward the linear sparse regime.
The trainer now supports mixed-precision, validation-driven adaptive jumps,
and gradient accumulation so longer contexts remain stable as they scale.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import contextlib
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
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


def _resolve_embedding_weight(model: torch.nn.Module) -> Optional[torch.Tensor]:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        return lm_head.weight
    token_emb = getattr(model, "token_emb", None)
    if token_emb is not None and hasattr(token_emb, "weight"):
        return token_emb.weight
    return None


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
    sample_tokens: int,
    top_a_mode: str = "blend",
    top_a_lambda: float = 0.5,
    top_a_tau: float = 0.7,
    top_a_threshold: Optional[float] = None,
    top_a_train_weight: float = 0.0,
    top_a_train_tau: float = 0.7,
) -> dict:
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    tokenizer = common.get_tokenizer()
    text = common.load_corpus(data_path, max_chars=4 * 1024 * 1024)

    def _inject_delimiters(raw_text: str) -> str:
        eos_token = getattr(tokenizer, "eos_token", None) or getattr(tokenizer, "eot_token", None)
        if not raw_text:
            return raw_text
        records = [segment.strip() for segment in raw_text.split("\n\n") if segment.strip()]
        if len(records) < 2:
            return raw_text
        separator = f"\n{eos_token}\n" if eos_token else "\n\n"
        return separator.join(records)

    def _resolve_eos_id() -> Optional[int]:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        eos_token = getattr(tokenizer, "eos_token", None) or getattr(tokenizer, "eot_token", None)
        converter = getattr(tokenizer, "convert_tokens_to_ids", None)
        if eos_token and callable(converter):
            try:
                converted = converter(eos_token)
                if converted is not None:
                    return int(converted)
            except Exception:
                pass
        return None

    def _resolve_vocab_size() -> int:
        for attr in ("vocab_size", "n_vocab"):
            value = getattr(tokenizer, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        vocab = getattr(tokenizer, "get_vocab", None)
        if callable(vocab):
            try:
                entries = vocab()
                if isinstance(entries, dict) and entries:
                    return len(entries)
            except Exception:
                pass
        encoder = getattr(tokenizer, "encoder", None)
        if isinstance(encoder, dict) and encoder:
            return len(encoder)
        return 256

    text = _inject_delimiters(text)
    tokens = list(tokenizer.encode(text))
    eos_token_id = _resolve_eos_id()
    if eos_token_id is not None and (not tokens or tokens[-1] != eos_token_id):
        tokens.append(eos_token_id)

    vocab_size = _resolve_vocab_size()
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
        attn_proto_enable=True,
        attn_memory_enable=True,
        attn_memory_slots=64,
        attn_memory_decay=0.95,
    )
    ckpt_dir = common.ensure_checkpoint_dir("context_mastery")
    is_cuda = device.type == "cuda"
    amp_enabled = bool(use_amp and is_cuda)
    can_bf16 = bool(
        amp_enabled
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    amp_dtype = torch.bfloat16 if can_bf16 else torch.float16
    autocast_kwargs: dict[str, object] = {
        "device_type": "cuda" if is_cuda else "cpu",
        "enabled": amp_enabled,
    }
    if amp_enabled:
        autocast_kwargs["dtype"] = amp_dtype

    def _autocast():
        return torch.amp.autocast(**autocast_kwargs) if amp_enabled else contextlib.nullcontext()
    grad_accum_steps = max(1, int(grad_accum_steps))
    seq_growth_factor = max(1.0, float(seq_growth_factor))
    val_fraction = float(max(0.0, min(0.5, val_fraction)))
    val_iters = max(1, int(val_iters)) if val_fraction > 0.0 else 0
    warmup_steps = max(0, int(warmup_steps))
    plateau_window = max(1, int(plateau_window))
    NUDGE_FACTOR = 1.15
    NUDGE_STEPS = 50
    EMA_BETA = 0.9
    max_lr_allowed = max(lr * max(3.0, NUDGE_FACTOR * 2.0), 1e-8)

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

    BUCKET_TARGETS = [("short", 256), ("medium", 1024), ("long", 2048)]
    fixed_val_seq_len = max(8, min(max_seq_len, base_seq_len)) if (val_fraction > 0.0 and val_iters > 0) else None
    bucket_configs: list[tuple[str, int]] = []
    if fixed_val_seq_len is not None:
        seen_bucket_lengths: set[int] = set()
        for label, target_len in BUCKET_TARGETS:
            actual_len = min(max_seq_len, target_len)
            if actual_len < 8 or actual_len in seen_bucket_lengths:
                continue
            seen_bucket_lengths.add(actual_len)
            bucket_configs.append((label, actual_len))
    bucket_labels = [label for label, _ in bucket_configs]

    model = MiniProteusLM(model_cfg).to(device)
    ema_helper = common.EMA(model, decay=0.999)
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
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    def _snapshot_rng_state() -> dict[str, object]:
        state: dict[str, object] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _checkpoint_payload() -> dict[str, object]:
        return {
            "model_cfg": asdict(model_cfg),
            "training_cfg": asdict(train_cfg),
            "history": history,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema_shadow": {name: tensor.clone() for name, tensor in ema_helper.shadow.items()},
            "rng_state": _snapshot_rng_state(),
        }

    alpha = float(max(0.0, min(1.0, alpha_start)))
    model.set_shortlist_alpha(alpha)
    seq_len = base_seq_len
    target_tokens_per_step = batch_size * base_seq_len * grad_accum_steps

    def adjust_accum_for_seq_len(current_seq_len: int) -> int:
        if current_seq_len <= 0:
            return 1
        tokens_per_batch = batch_size * current_seq_len
        new_accum = max(1, round(target_tokens_per_step / max(1, tokens_per_batch)))
        return int(new_accum)

    grad_accum_steps = adjust_accum_for_seq_len(seq_len)

    def _build_train_batches(current_seq_len: int):
        return common.build_batches(train_tokens, current_seq_len, batch_size, device=device)

    train_batches = _build_train_batches(seq_len)

    history: list[dict[str, float | int | str | None]] = []
    attn_counters = common.init_attn_counters()
    plateau_hits = 0
    last_adjust_step = 0
    best_fixed_checkpoint_loss = float("inf")
    best_checkpoint_path: Optional[Path] = None
    ema_fixed_val: Optional[float] = None
    best_fixed_ema: Optional[float] = None
    lr_scale = 1.0
    lr_nudge_steps_remaining = 0
    ema_updates = 0
    tokens_trained = 0

    def _run_validation(current_seq_len: int) -> Optional[dict[str, object]]:
        if val_fraction <= 0.0 or val_iters <= 0 or fixed_val_seq_len is None:
            return None

        def _gather_losses(length: int) -> list[float]:
            batch_iter = common.build_batches(val_tokens, length, batch_size, device=device)
            collected: list[float] = []
            skip = random.randint(0, 15)
            for _ in range(skip):
                next(batch_iter)
            for _ in range(val_iters):
                vx, vy = next(batch_iter)
                with _autocast():
                    val_logits = model(vx)
                    val_loss = F.cross_entropy(
                        val_logits.view(-1, vocab_size),
                        vy.view(-1),
                        ignore_index=-100,
                    )
                collected.append(float(val_loss.item()))
            return collected

        bucket_stats: dict[str, tuple[float, float]] = {}
        with ema_helper.swap(model):
            model.eval()
            with torch.no_grad():
                match_losses = _gather_losses(current_seq_len)
                fixed_losses = _gather_losses(fixed_val_seq_len)
                for label, bucket_len in bucket_configs:
                    bucket_losses = _gather_losses(bucket_len)
                    if not bucket_losses:
                        continue
                    bucket_loss = sum(bucket_losses) / len(bucket_losses)
                    bucket_stats[label] = (
                        bucket_loss,
                        float(math.exp(min(30.0, bucket_loss))),
                    )
        model.train()

        def _summarize(losses: list[float]) -> Optional[tuple[float, float]]:
            if not losses:
                return None
            avg = sum(losses) / len(losses)
            return avg, float(math.exp(min(30.0, avg)))

        match_summary = _summarize(match_losses)
        fixed_summary = _summarize(fixed_losses)
        return {
            "match": match_summary,
            "fixed": fixed_summary,
            "buckets": bucket_stats,
        }

    model.train()
    for step in range(1, steps + 1):
        if seq_len > model_cfg.max_seq_len:
            raise ValueError("Sequence length exceeded model maximum during training.")
        opt.zero_grad(set_to_none=True)
        tokens_per_step = batch_size * seq_len * grad_accum_steps
        tokens_trained += tokens_per_step
        micro_losses: list[float] = []
        align_loss_values: list[float] = []
        for _ in range(grad_accum_steps):
            x, y = next(train_batches)
            with _autocast():
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    y.view(-1),
                    ignore_index=-100,
                )
                if top_a_train_weight > 0.0:
                    head_consensus = None
                    if hasattr(model, "last_head_consensus") and callable(
                        getattr(model, "last_head_consensus")
                    ):
                        try:
                            head_consensus = model.last_head_consensus()
                        except Exception:
                            head_consensus = None
                    else:
                        head_consensus = getattr(model, "last_head_consensus", None)
                    embed_weight = _resolve_embedding_weight(model)
                    if (
                        head_consensus is not None
                        and embed_weight is not None
                        and head_consensus.shape == logits.shape
                    ):
                        norm_consensus = F.normalize(head_consensus.to(embed_weight.device), dim=-1)
                        norm_embed = F.normalize(embed_weight, dim=-1)
                        align_logits = torch.matmul(
                            norm_consensus, norm_embed.transpose(0, 1)
                        ) / max(top_a_train_tau, 1e-5)
                        align_loss_tensor = F.cross_entropy(
                            align_logits.view(-1, vocab_size),
                            y.view(-1),
                            ignore_index=-100,
                        )
                        loss = loss + top_a_train_weight * align_loss_tensor
                        align_loss_values.append(float(align_loss_tensor.item()))
            micro_losses.append(float(loss.item()))
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema_helper.update(model)
        scheduler.step()
        if lr_nudge_steps_remaining > 0:
            lr_nudge_steps_remaining -= 1
            if lr_nudge_steps_remaining == 0:
                lr_scale = 1.0
        try:
            base_lrs = scheduler.get_last_lr()
        except AttributeError:
            base_lrs = [group["lr"] / max(lr_scale, 1e-9) for group in opt.param_groups]
        scaled_lrs = [lr_value * lr_scale for lr_value in base_lrs]
        for group, lr_value in zip(opt.param_groups, scaled_lrs):
            group["lr"] = lr_value
        current_lr = scaled_lrs[0] if scaled_lrs else opt.param_groups[0]["lr"]
        if (not math.isfinite(current_lr)) or current_lr <= 0 or current_lr > max_lr_allowed:
            raise RuntimeError(
                f"[context-mastery] learning rate out of bounds: {current_lr:.3e} (limit {max_lr_allowed:.3e})"
            )
        common.accumulate_attention_counters(model, attn_counters)

        loss_value = float(sum(micro_losses) / len(micro_losses))
        align_loss_value = (
            float(sum(align_loss_values) / len(align_loss_values))
            if align_loss_values
            else None
        )
        if not math.isfinite(loss_value):
            print("[context-mastery] warning: non-finite loss detected; reducing LR and skipping metrics.")
            for group in opt.param_groups:
                group["lr"] = max(group["lr"] * 0.5, 1e-8)
            continue
        ppl = float(math.exp(min(30.0, loss_value)))

        val_metrics = _run_validation(seq_len)
        match_loss: Optional[float] = None
        match_ppl: Optional[float] = None
        fixed_loss: Optional[float] = None
        fixed_ppl: Optional[float] = None
        bucket_stats: dict[str, tuple[float, float]] = {}
        if isinstance(val_metrics, dict):
            match_entry = val_metrics.get("match")
            if isinstance(match_entry, tuple):
                match_loss, match_ppl = match_entry
            fixed_entry = val_metrics.get("fixed")
            if isinstance(fixed_entry, tuple):
                fixed_loss, fixed_ppl = fixed_entry
            bucket_entry = val_metrics.get("buckets")
            if isinstance(bucket_entry, dict):
                bucket_stats = bucket_entry

        record = {
            "step": step,
            "alpha": alpha,
            "seq_len": seq_len,
            "loss": loss_value,
            "ppl": ppl,
            "val_loss": match_loss,
            "val_ppl": match_ppl,
            "val_loss_fixed": fixed_loss,
            "val_ppl_fixed": fixed_ppl,
            "lr": current_lr,
            "align_loss": align_loss_value,
        }
        for label in bucket_labels:
            bucket_pair = bucket_stats.get(label)
            record[f"val_loss_{label}"] = bucket_pair[0] if bucket_pair else None
            record[f"val_ppl_{label}"] = bucket_pair[1] if bucket_pair else None
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
            if match_loss is not None and match_ppl is not None:
                log_parts.append(f"val_match={match_loss:.4f}/{match_ppl:.2f}")
            if fixed_loss is not None and fixed_ppl is not None:
                log_parts.append(f"val_fixed={fixed_loss:.4f}/{fixed_ppl:.2f}")
            for label in bucket_labels:
                bucket_pair = bucket_stats.get(label)
                if bucket_pair is not None:
                    log_parts.append(f"{label}_ppl={bucket_pair[1]:.2f}")
            print(" ".join(log_parts))

        if step % 50 == 0:
            print(
                f"[context-mastery] stats tokens/step≈{tokens_per_step} "
                f"(accum={grad_accum_steps}, seq={seq_len}) "
                f"tokens_trained≈{int(tokens_trained):,}"
            )
        if step % 100 == 0:
            print(
                f"[context-mastery] stats grad_norm={float(grad_norm):.3f} "
                f"lr={current_lr:.3e}"
            )

        if fixed_loss is not None and fixed_loss < best_fixed_checkpoint_loss:
            best_fixed_checkpoint_loss = fixed_loss
            best_checkpoint_path = ckpt_dir / "best.pt"
            payload = _checkpoint_payload()
            payload["best_val_loss_fixed"] = best_fixed_checkpoint_loss
            torch.save(payload, best_checkpoint_path)
        plateau_ready = False
        if fixed_loss is not None and math.isfinite(fixed_loss):
            if ema_fixed_val is None:
                ema_fixed_val = fixed_loss
            else:
                ema_fixed_val = EMA_BETA * ema_fixed_val + (1.0 - EMA_BETA) * fixed_loss
            ema_updates += 1
            plateau_ready = ema_updates >= plateau_window
            if best_fixed_ema is None or ema_fixed_val < best_fixed_ema:
                best_fixed_ema = ema_fixed_val
                plateau_hits = 0
            else:
                improvement = (best_fixed_ema - ema_fixed_val) / max(abs(best_fixed_ema), 1e-6)
                if improvement < plateau_tol:
                    plateau_hits += 1
                else:
                    plateau_hits = 0
        elif val_metrics is None:
            plateau_hits = 0

        if (
            plateau_ready
            and plateau_hits >= plateau_patience
            and (step - last_adjust_step) >= plateau_cooldown
        ):
            adjusted = False
            if alpha < 1.0:
                alpha = min(1.0, alpha + alpha_step)
                model.set_shortlist_alpha(alpha)
                adjusted = True
            if seq_len < max_seq_len:
                proposed = int(seq_len * seq_growth_factor)
                if proposed <= seq_len:
                    proposed = seq_len + max(1, base_seq_len // 4 or 1)
                seq_len = min(max_seq_len, proposed)
                grad_accum_steps = adjust_accum_for_seq_len(seq_len)
                train_batches = _build_train_batches(seq_len)
                adjusted = True
            if adjusted:
                last_adjust_step = step
                plateau_hits = 0
                ema_updates = 0
                best_fixed_ema = ema_fixed_val
                lr_scale = min(NUDGE_FACTOR, 1.5)
                lr_nudge_steps_remaining = NUDGE_STEPS
                tokens_per_step = batch_size * seq_len * grad_accum_steps
                print(
                    f"[context-mastery] plateau detected -> alpha={alpha:.2f}, "
                    f"seq_len={seq_len}, tokens/step≈{tokens_per_step}"
                )

    ckpt_path = ckpt_dir / "model.pt"
    final_payload = _checkpoint_payload()
    torch.save(final_payload, ckpt_path)
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    history_csv_path = ckpt_dir / "history.csv"
    fieldnames = [
        "step",
        "alpha",
        "seq_len",
        "loss",
        "ppl",
        "val_loss",
        "val_ppl",
        "val_loss_fixed",
        "val_ppl_fixed",
        "lr",
        "align_loss",
    ]
    for label in bucket_labels:
        fieldnames.append(f"val_loss_{label}")
        fieldnames.append(f"val_ppl_{label}")
    with history_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    final_record = history[-1] if history else {}
    summary = {
        "steps": steps,
        "train_loss": final_record.get("loss"),
        "train_ppl": final_record.get("ppl"),
        "val_loss_match": final_record.get("val_loss"),
        "val_ppl_match": final_record.get("val_ppl"),
        "val_loss_fixed": final_record.get("val_loss_fixed"),
        "val_ppl_fixed": final_record.get("val_ppl_fixed"),
        "tokens_trained": tokens_trained,
        "align_loss": final_record.get("align_loss"),
    }
    for label in bucket_labels:
        summary[f"val_loss_{label}"] = final_record.get(f"val_loss_{label}")
        summary[f"val_ppl_{label}"] = final_record.get(f"val_ppl_{label}")
    attn_summary = common.summarize_attention_counters(attn_counters, steps)
    print(f"[context-mastery] saved checkpoint to {ckpt_path}")
    if best_checkpoint_path is not None:
        print(f"[context-mastery] best checkpoint updated at {best_checkpoint_path}")
    common.print_metric_block("context-mastery", summary)
    common.print_metric_block("context-mastery.attention", attn_summary)
    common.interactive_inference_loop(
        model,
        tokenizer,
        device=device,
        max_seq_len=model_cfg.max_seq_len,
        sample_tokens=sample_tokens,
        label="context-mastery",
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        min_new_tokens=20,
        max_new_tokens=min(sample_tokens, 160),
        top_a_mode=top_a_mode,
        top_a_lambda=top_a_lambda,
        top_a_tau=top_a_tau,
        top_a_threshold=top_a_threshold,
    )
    return {
        "checkpoint": str(ckpt_path),
        "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        "history": history,
        "history_csv": str(history_csv_path),
        "metrics": summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive context mastery trainer.")
    parser.add_argument("--steps", type=int, default=2500, help="Total optimization steps.")
    parser.add_argument("--base-seq-len", type=int, default=256, help="Initial sequence length.")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--device", default=None, help="Device string (default auto).")
    parser.add_argument("--data", type=Path, default=None, help="Optional training text.")
    parser.add_argument("--alpha-start", type=float, default=0.0, help="Initial Shortlist alpha.")
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
    parser.add_argument("--sample-tokens", type=int, default=160, help="Tokens to generate.")
    parser.add_argument("--top-a-mode", choices=["off", "blend", "filter"], default="blend", help="Top-A sampling mode for interactive decoding.")
    parser.add_argument("--top-a-lambda", type=float, default=0.5, help="Logit blend factor for Top-A sampling.")
    parser.add_argument("--top-a-tau", type=float, default=0.7, help="Temperature for Top-A agreement logits.")
    parser.add_argument("--top-a-threshold", type=float, default=None, help="Agreement threshold for filter mode.")
    parser.add_argument("--top-a-train-weight", type=float, default=0.0, help="Weight applied to the consensus alignment loss.")
    parser.add_argument("--top-a-train-tau", type=float, default=0.7, help="Temperature for the consensus alignment loss.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    data_path = common.resolve_dataset_path(args.data, label="context-mastery")
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
        data_path=data_path,
        sample_tokens=args.sample_tokens,
        top_a_mode=args.top_a_mode,
        top_a_lambda=args.top_a_lambda,
        top_a_tau=args.top_a_tau,
        top_a_threshold=args.top_a_threshold,
        top_a_train_weight=args.top_a_train_weight,
        top_a_train_tau=args.top_a_train_tau,
    )


if __name__ == "__main__":
    main()
