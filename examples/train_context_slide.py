#!/usr/bin/env python
"""
Curriculum training example that slides context length in stages.

70% of steps train on short sequences (dense mode), 20% on medium contexts
with blended alpha, and the final 10% on long contexts using the linear path.
"""

from __future__ import annotations

import argparse
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


def _set_alpha(model: MiniProteusLM, value: float) -> None:
    model.set_shortlist_alpha(value)


def run_training(
    *,
    total_steps: int,
    short_seq: int,
    medium_seq: int,
    long_seq: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    data_path: Optional[Path] = None,
    sample_tokens: int,
) -> dict:
    tokenizer = common.get_tokenizer()
    text = common.load_corpus(data_path, max_chars=4 * 1024 * 1024)
    tokens = tokenizer.encode(text)
    train_tokens, val_tokens = common.split_train_val(tokens, short_seq, val_fraction=0.1)

    vocab_size = getattr(tokenizer, "n_vocab", 256)
    train_cfg = common.TrainingConfig(
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        vocab_size=vocab_size,
        max_seq_len=max(long_seq, 512),
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
    model = MiniProteusLM(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    stages = [
        ("short", 0.0, short_seq, 0.7),
        ("medium", 0.5, medium_seq, 0.2),
        ("long", 1.0, long_seq, 0.1),
    ]

    logs = []
    attn_counters = common.init_attn_counters()
    remaining = total_steps
    base_tokens = short_seq * batch_size

    for stage_index, (label, alpha, seq_len, fraction) in enumerate(stages):
        steps = int(round(total_steps * fraction))
        if stage_index == len(stages) - 1:
            steps = remaining
        steps = max(1, steps)
        remaining -= steps

        stage_batch = max(1, int(base_tokens // seq_len))
        if stage_batch < 1:
            stage_batch = 1
        batches = common.build_batches(train_tokens, seq_len, stage_batch, device=device)
        _set_alpha(model, alpha)

        stage_losses: list[float] = []
        model.train()
        for step in range(1, steps + 1):
            x, y = next(batches)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stage_losses.append(float(loss.item()))
            common.accumulate_attention_counters(model, attn_counters)
            if step % max(1, steps // 5) == 0 or step == 1:
                ppl = float(torch.exp(loss.detach()).item())
                print(
                f"[context-slide] stage={label} step={step}/{steps} "
                f"seq_len={seq_len} batch={stage_batch} alpha={alpha:.2f} "
                f"loss={loss.item():.4f} ppl={ppl:.2f}"
            )

        logs.append(
            {
                "stage": label,
                "alpha": alpha,
                "seq_len": seq_len,
                "batch_size": stage_batch,
                "steps": steps,
                "loss_mean": float(sum(stage_losses) / len(stage_losses)),
                "loss_final": stage_losses[-1],
            }
        )

    val_loss, val_ppl = common.evaluate_language_model(
        model,
        val_tokens,
        seq_len=long_seq,
        batch_size=batch_size,
        device=device,
    )
    final_stage = logs[-1] if logs else {}
    final_loss = final_stage.get("loss_final", float("nan"))
    final_ppl = float(torch.exp(torch.tensor(final_loss)).item()) if not math.isnan(final_loss) else float("nan")
    summary = {
        "steps": total_steps,
        "train_loss": final_loss,
        "train_ppl": final_ppl,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
    }
    attn_summary = common.summarize_attention_counters(attn_counters, total_steps)

    ckpt_dir = common.ensure_checkpoint_dir("context_slide")
    ckpt_path = ckpt_dir / "model.pt"
    torch.save(
        {
            "model_cfg": asdict(model_cfg),
            "training_cfg": asdict(train_cfg),
            "stage_logs": logs,
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    (ckpt_dir / "stages.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    print(f"[context-slide] saved checkpoint to {ckpt_path}")
    common.print_metric_block("context-slide", summary)
    common.print_metric_block("context-slide.attention", attn_summary)
    common.interactive_inference_loop(
        model,
        tokenizer,
        device=device,
        max_seq_len=model_cfg.max_seq_len,
        sample_tokens=sample_tokens,
        label="context-slide",
    )
    return {"checkpoint": str(ckpt_path), "stages": logs, "metrics": summary}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proteus Attention context-slide trainer.")
    parser.add_argument("--steps", type=int, default=2500, help="Total optimization steps.")
    parser.add_argument("--short-seq", type=int, default=256, help="Short curriculum length.")
    parser.add_argument("--medium-seq", type=int, default=1024, help="Medium curriculum length.")
    parser.add_argument("--long-seq", type=int, default=4096, help="Long curriculum length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--device", default=None, help="Device string (default auto).")
    parser.add_argument("--data", type=Path, default=None, help="Optional training text.")
    parser.add_argument("--sample-tokens", type=int, default=120, help="Tokens to sample.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    data_path = common.resolve_dataset_path(args.data, label="context-slide")
    run_training(
        total_steps=args.steps,
        short_seq=args.short_seq,
        medium_seq=args.medium_seq,
        long_seq=args.long_seq,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        data_path=data_path,
        sample_tokens=args.sample_tokens,
    )


if __name__ == "__main__":
    main()
