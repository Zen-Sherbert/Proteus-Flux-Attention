#!/usr/bin/env python3
"""Staged training harness for Proteus Adaptive Sparse Attention.

This script consumes the run artefacts produced by ``proteus_convertor.py`` and
executes the staged curriculum: prototype warm-up, sparse ramp, long-context
training, and optional consolidation.  It is intentionally modular so teams can
swap in architecture-specific adapters while reusing the training logic.

The trainer keeps terminal output succinct by redrawing a single status panel
per stage; detailed metrics are persisted to JSONL/CSV files under the run
directory.  After each stage the script pauses and presents a menu so users can
inspect progress, switch datasets, or abort.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, IterableDataset

from proteus_attention.integration import AdapterConfig, load_and_prepare_model


# --------------------------------------------------------------------------- IO


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run_dir(workspace: Path) -> Path:
    runs_root = workspace / "proteus_runs"
    if not runs_root.exists():
        raise FileNotFoundError(f"No proteus_runs directory under {workspace}")
    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {runs_root}")
    return max(candidates, key=lambda p: p.name)


# -------------------------------------------------------------------- Data Prep


class StreamingTextDataset(IterableDataset):
    def __init__(
        self,
        files: List[Path],
        tokenizer,
        seq_len: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.files = files
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    def _file_iterator(self) -> Iterator[str]:
        while True:
            for path in self.files:
                try:
                    yield path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    print(f"[dataset] skipping non-UTF8 file {path}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        pad_id = self.tokenizer.pad_token_id or 0
        generator = self._file_iterator()
        for text in generator:
            encoded = self.tokenizer(text, return_attention_mask=False, add_special_tokens=False)
            ids = encoded["input_ids"]
            if len(ids) < 2:
                continue
            start = 0
            while start + 1 < len(ids):
                end = min(start + self.seq_len, len(ids))
                chunk = ids[start:end]
                if len(chunk) < 2:
                    break
                if len(chunk) < self.seq_len:
                    chunk = chunk + [pad_id] * (self.seq_len - len(chunk))
                input_ids = torch.tensor(chunk, dtype=torch.long)
                labels = input_ids.clone()
                labels[:-1] = input_ids[1:]
                labels[-1] = -100
                yield {"input_ids": input_ids.unsqueeze(0), "labels": labels.unsqueeze(0)}
                start += self.stride


def resolve_dataset_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"No files found in dataset path {path}")
    return files


# ---------------------------------------------------------------- Metrics / UX


@dataclass
class StageContext:
    name: str
    description: str
    max_steps: int
    sequence_length: int
    alpha_start: float
    alpha_end: float


class PlateauDetector:
    def __init__(self, window: int, min_improvement: float) -> None:
        self.window = max(1, int(window))
        self.min_improvement = float(min_improvement)
        self.history: Deque[float] = deque(maxlen=self.window)
        self.best: float = float("inf")

    def update(self, value: float) -> bool:
        improved = False
        if value < self.best - self.min_improvement:
            self.best = value
            improved = True
        self.history.append(value)
        if improved:
            return False
        if len(self.history) < self.window:
            return False
        window_best = min(self.history)
        return (window_best >= self.best - self.min_improvement)


def format_status(stage: StageContext, step: int, loss: float, lr: float, alpha: float, elapsed: float) -> str:
    return (
        f"[{stage.name} | {stage.description}] step {step}/{stage.max_steps} "
        f"loss={loss:.4f} lr={lr:.2e} alpha={alpha:.3f} elapsed={elapsed/60:.1f}m"
    )


def prompt_menu(options: List[Tuple[str, str]]) -> str:
    print("\n== Stage complete ==")
    for key, label in options:
        print(f"  [{key}] {label}")
    while True:
        choice = input("Select option: ").strip().lower()
        if any(choice == key for key, _ in options):
            return choice
        print("Invalid choice, try again.")


# ----------------------------------------------------------- Training Helpers


def evaluate_model(model, dataloader: Iterable[Dict[str, torch.Tensor]], device: torch.device, limit: int) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= limit:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            tokens = (labels != -100).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    model.train()
    if total_tokens == 0:
        return float("inf"), float("inf")
    mean_loss = total_loss / total_tokens
    ppl = math.exp(mean_loss)
    return mean_loss, ppl


def save_checkpoint(model, optimizer, stage_dir: Path, step: int) -> None:
    ckpt_dir = stage_dir / f"checkpoint-step{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")


# -------------------------------------------------------------------- Main Run


def run_stage(
    stage: StageContext,
    model,
    adapter,
    tokenizer,
    dataset_path: Path,
    run_dir: Path,
    device: torch.device,
    *,
    batch_size: int,
    lr: float,
    proto_lr_scale: float,
    grad_clip: float,
    eval_interval: int,
    eval_batches: int,
    save_interval: int,
    mastery_cfg: Optional[dict],
) -> bool:
    stage_dir = run_dir / stage.name.replace(" ", "_")
    stage_dir.mkdir(parents=True, exist_ok=True)
    train_log = stage_dir / "train_log.jsonl"
    eval_log = stage_dir / "eval_log.jsonl"

    files = resolve_dataset_files(dataset_path)
    dataset = StreamingTextDataset(files, tokenizer, seq_len=stage.sequence_length, stride=max(1, stage.sequence_length // 2))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    adapter.freeze_for_stage(stage.name)
    param_groups = adapter.parameter_groups(base_lr=lr, proto_lr_scale=proto_lr_scale)
    optimizer = torch.optim.AdamW(param_groups)

    start_time = time.time()
    alpha = stage.alpha_start
    adapter.set_shortlist_alpha(alpha)

    mastery_detector: Optional[PlateauDetector] = None
    alpha_increment = None
    if mastery_cfg and stage.alpha_start != stage.alpha_end:
        mastery_detector = PlateauDetector(
            window=int(mastery_cfg.get("plateau_window", 400)),
            min_improvement=float(mastery_cfg.get("min_improvement", 0.005)),
        )
        alpha_increment = float(mastery_cfg.get("alpha_increment", 0.05))

    with train_log.open("w", encoding="utf-8") as train_handle, eval_log.open("w", encoding="utf-8") as eval_handle:
        total_tokens = 0
        for step, batch in enumerate(dataloader, start=1):
            if step > stage.max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            tokens = (labels != -100).sum().item()
            total_tokens += tokens

            progress = step / max(1, stage.max_steps)
            if stage.alpha_end != stage.alpha_start and not mastery_cfg:
                alpha = stage.alpha_start + progress * (stage.alpha_end - stage.alpha_start)
                adapter.set_shortlist_alpha(alpha)

            elapsed = time.time() - start_time
            status = format_status(stage, step, loss.item(), optimizer.param_groups[0]["lr"], alpha, elapsed)
            print(status, end="\r", flush=True)

            train_handle.write(json.dumps({
                "step": step,
                "loss": loss.item(),
                "tokens": tokens,
                "lr": optimizer.param_groups[0]["lr"],
                "alpha": alpha,
                "elapsed_sec": elapsed,
            }) + "\n")
            train_handle.flush()

            if eval_interval and step % eval_interval == 0:
                eval_loss, eval_ppl = evaluate_model(model, dataloader, device, eval_batches)
                eval_handle.write(json.dumps({
                    "step": step,
                    "val_loss": eval_loss,
                    "val_ppl": eval_ppl,
                    "tokens": total_tokens,
                }) + "\n")
                eval_handle.flush()
                if mastery_detector is not None and math.isfinite(eval_loss):
                    plateau = mastery_detector.update(eval_loss)
                    if plateau and alpha_increment is not None:
                        alpha = min(stage.alpha_end, alpha + alpha_increment)
                        adapter.set_shortlist_alpha(alpha)
                        print(f"\n[mastery] plateau detected -> alpha -> {alpha:.3f}")

            if save_interval and step % save_interval == 0:
                save_checkpoint(model, optimizer, stage_dir, step)

        print("\n")

    return True


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Proteus staged trainer")
    parser.add_argument("--workspace", required=True, help="Path to the converted -Proteus workspace")
    parser.add_argument("--run-id", help="Specific run directory (defaults to latest)")
    parser.add_argument("--dataset", help="Override dataset path")
    parser.add_argument("--device", default="auto", help="cpu|cuda|cuda:idx")
    parser.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Model dtype")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size per step")
    parser.add_argument("--lr", type=float, default=2e-4, help="Base learning rate")
    parser.add_argument("--proto-lr-scale", type=float, default=5.0, help="Scale factor for Proteus-specific params (adapter dependent)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping (0 disables)")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval in steps")
    parser.add_argument("--eval-batches", type=int, default=4, help="Number of batches per evaluation")
    parser.add_argument("--save-interval", type=int, default=2000, help="Checkpoint interval in steps")

    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        raise FileNotFoundError(workspace)

    run_dir = Path(args.run_id).expanduser().resolve() if args.run_id else latest_run_dir(workspace)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    baseline = load_json(run_dir / "baseline_metrics.json")
    plan = load_json(run_dir / "stage_plan.json")
    options_cfg = plan["training_options"]

    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else Path(plan.get("dataset") or baseline.get("dataset_path") or baseline.get("dataset") or "")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path '{dataset_path}' not found")

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if args.precision == "auto":
        if device.type == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    elif args.precision == "fp32":
        torch_dtype = torch.float32
    elif args.precision == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    adapter_cfg = AdapterConfig(
        heads=int(options_cfg["heads"]),
        gate_ratio=float(options_cfg["gate_ratio"]),
        strategy=options_cfg["strategy"],
        target_ctx=options_cfg.get("target_ctx"),
        context_mastery=options_cfg.get("context_mastery_settings"),
    )

    model, adapter, tokenizer = load_and_prepare_model(workspace, adapter_cfg, device=device, torch_dtype=torch_dtype)
    model.train()

    stages = [
        StageContext(
            name=stage_dict["name"],
            description=stage_dict["description"],
            max_steps=int(stage_dict["max_steps"]),
            sequence_length=int(stage_dict["sequence_length"]),
            alpha_start=float(stage_dict["shortlist_alpha_start"]),
            alpha_end=float(stage_dict["shortlist_alpha_end"]),
        )
        for stage_dict in plan["stages"]
    ]

    print(f"Workspace: {workspace}")
    print(f"Run directory: {run_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device} | dtype: {torch_dtype}")

    for idx, stage in enumerate(stages, start=1):
        print(f"\n=== Stage {idx}/{len(stages)} — {stage.name} ===")
        print(stage.description)
        run_stage(
            stage,
            model,
            adapter,
            tokenizer,
            dataset_path,
            run_dir,
            device,
            batch_size=args.batch_size,
            lr=args.lr,
            proto_lr_scale=args.proto_lr_scale,
            grad_clip=args.grad_clip,
            eval_interval=args.eval_interval,
            eval_batches=args.eval_batches,
            save_interval=args.save_interval,
            mastery_cfg=adapter_cfg.context_mastery if adapter_cfg.strategy == "context_mastery" else None,
        )

        choice = prompt_menu([
            ("c", "Continue to next stage"),
            ("d", "Change dataset path"),
            ("q", "Quit training"),
        ])
        if choice == "d":
            new_path = input("New dataset path: ").strip()
            candidate = Path(new_path).expanduser().resolve()
            if not candidate.exists():
                print(f"Path {candidate} not found; keeping previous dataset.")
            else:
                dataset_path = candidate
        if choice == "q":
            print("Stopping training per user request.")
            break

    print("Training run complete.")


if __name__ == "__main__":
    main()
