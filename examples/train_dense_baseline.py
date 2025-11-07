#!/usr/bin/env python
"""
Dense Transformer baseline for comparison with Proteus Attention.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":  # pragma: no cover
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import examples.common as common  # type: ignore
    from examples.modeling import ModelConfig, MiniDenseLM  # type: ignore
else:  # pragma: no cover
    from . import common
    from .modeling import ModelConfig, MiniDenseLM


def run_training(
    *,
    steps: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    sample_tokens: int,
    data_path: Optional[Path] = None,
) -> dict:
    tokenizer = common.get_tokenizer()
    text = common.load_corpus(data_path, max_chars=4 * 1024 * 1024)
    tokens = tokenizer.encode(text)
    train_tokens, val_tokens = common.split_train_val(tokens, seq_len, val_fraction=0.1)

    vocab_size = getattr(tokenizer, "n_vocab", 256)
    train_cfg = common.TrainingConfig(
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        vocab_size=vocab_size,
        max_seq_len=max(seq_len, 512),
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
    model = MiniDenseLM(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    batches = common.build_batches(train_tokens, seq_len, batch_size, device=device)
    losses: list[float] = []
    model.train()
    for step in range(1, steps + 1):
        x, y = next(batches)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, model_cfg.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if step % max(1, steps // 10) == 0 or step == 1:
            ppl = float(torch.exp(loss.detach()).item())
            print(f"[dense-baseline] step={step}/{steps} loss={loss.item():.4f} ppl={ppl:.2f}")

    model.eval()
    val_loss, val_ppl = common.evaluate_language_model(
        model,
        val_tokens,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
    )
    final_loss = losses[-1]
    train_ppl = float(torch.exp(torch.tensor(final_loss)).item())
    metrics = {
        "steps": steps,
        "train_loss": final_loss,
        "train_ppl": train_ppl,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
    }

    ckpt_dir = common.ensure_checkpoint_dir("dense_baseline")
    ckpt_path = ckpt_dir / "model.pt"
    torch.save(
        {
            "model_cfg": asdict(model_cfg),
            "training_cfg": asdict(train_cfg),
            "state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    meta = {
        "steps": steps,
        "final_loss": final_loss,
        "final_ppl": train_ppl,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "checkpoint": str(ckpt_path),
    }
    (ckpt_dir / "training.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[dense-baseline] saved checkpoint to {ckpt_path}")
    common.print_metric_block("dense-baseline", metrics)
    common.interactive_inference_loop(
        model,
        tokenizer,
        device=device,
        max_seq_len=model_cfg.max_seq_len,
        sample_tokens=sample_tokens,
        label="dense-baseline",
    )
    return meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense Transformer baseline trainer.")
    parser.add_argument("--steps", type=int, default=2500, help="Number of optimization steps.")
    parser.add_argument("--seq-len", type=int, default=256, help="Training sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--device", default=None, help="Device string (default auto).")
    parser.add_argument("--sample-tokens", type=int, default=80, help="Tokens to generate per query in inference mode.")
    parser.add_argument("--data", type=Path, default=None, help="Optional path to training text.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    data_path = common.resolve_dataset_path(args.data, label="dense-baseline")
    run_training(
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        sample_tokens=args.sample_tokens,
        data_path=data_path,
    )


if __name__ == "__main__":
    main()
