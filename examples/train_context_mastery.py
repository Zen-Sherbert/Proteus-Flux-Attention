#!/usr/bin/env python
"""
Adaptive context mastery trainer.

Gradually increases the Flux alpha (and optional sequence length) when loss
improvements plateau, pushing the model toward the linear sparse regime.
"""

from __future__ import annotations

import argparse
import json
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
    model = MiniProteusLM(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    alpha = float(max(0.0, min(1.0, alpha_start)))
    model.set_flux_alpha(alpha)
    seq_len = base_seq_len
    batches = common.build_batches(tokens, seq_len, batch_size, device=device)

    history = []
    recent_losses: list[float] = []
    prev_window_avg: Optional[float] = None
    plateau_hits = 0
    last_adjust_step = 0

    model.train()
    for step in range(1, steps + 1):
        if seq_len > model_cfg.max_seq_len:
            raise ValueError("Sequence length exceeded model maximum during training.")

        x, y = next(batches)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_value = float(loss.item())
        recent_losses.append(loss_value)
        history.append({"step": step, "alpha": alpha, "seq_len": seq_len, "loss": loss_value})

        if step % max(1, steps // 10) == 0 or step == 1:
            ppl = float(torch.exp(loss.detach()).item())
            print(
                f"[context-mastery] step={step}/{steps} seq_len={seq_len} "
                f"alpha={alpha:.2f} loss={loss_value:.4f} ppl={ppl:.2f}"
            )

        if len(recent_losses) >= plateau_window:
            window_avg = sum(recent_losses[-plateau_window:]) / plateau_window
            if prev_window_avg is not None:
                improvement = (prev_window_avg - window_avg) / max(prev_window_avg, 1e-6)
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
                            seq_len = min(max_seq_len, seq_len + base_seq_len // 2)
                            batches = common.build_batches(tokens, seq_len, batch_size, device=device)
                        last_adjust_step = step
                        plateau_hits = 0
                        print(
                            f"[context-mastery] plateau detected. "
                            f"new alpha={alpha:.2f}, seq_len={seq_len}"
                        )
                else:
                    plateau_hits = 0
            prev_window_avg = window_avg

    model.eval()
    with torch.inference_mode():
        encoded_prompt = tokenizer.encode(prompt)
        input_ids = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
        out_tokens = model.generate(input_ids, max_new_tokens=sample_tokens)
        completion = tokenizer.decode(out_tokens[0].tolist())

    ckpt_dir = common.ensure_checkpoint_dir("context_mastery")
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
    sample_path = ckpt_dir / "sample.txt"
    sample_path.write_text(completion, encoding="utf-8")
    print(f"[context-mastery] saved checkpoint to {ckpt_path}")
    print(f"[context-mastery] sample:\n{completion}")
    return {"checkpoint": str(ckpt_path), "history": history, "sample": str(sample_path)}


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
    parser.add_argument("--plateau-window", type=int, default=200, help="Steps per plateau window.")
    parser.add_argument("--plateau-tol", type=float, default=0.01, help="Relative improvement threshold.")
    parser.add_argument("--plateau-patience", type=int, default=3, help="Number of consecutive plateau windows before adjustment.")
    parser.add_argument("--plateau-cooldown", type=int, default=150, help="Minimum steps between adjustments.")
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
        data_path=args.data,
        prompt=args.prompt,
        sample_tokens=args.sample_tokens,
    )


if __name__ == "__main__":
    main()
