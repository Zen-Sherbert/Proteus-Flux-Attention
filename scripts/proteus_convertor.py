#!/usr/bin/env python3
"""Interactive converter + staging CLI for Proteus attention fine-tuning.

The program guides users through:

1. Selecting a Hugging Face checkpoint directory.
2. Providing an evaluation corpus (single file or directory tree of text).
3. Running a baseline loss / perplexity pass using the checkpoint's tokenizer
   and weights (no gradients updated).
4. Confirming whether to proceed with conversion, including presenting a
   cautionary message that Proteus is still experimental.
5. Collecting core hyperparameters (heads, gate ratio, context plan) and
   storing them in a structured run plan.
6. Cloning the checkpoint into a ``-Proteus`` workspace with injected config
   hints, README, and a run directory containing baseline metrics plus stage
   definitions.

Actual fine-tuning is not launched automatically—the generated run plan is
designed to feed the forthcoming training harness.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CONFIG_INJECTION: Dict[str, Any] = {
    "attention_impl": "proteus_adaptive_sparse",
    "proteus_attention": {
        "attn_proto_enable": True,
        "attn_proto_threshold": 0.25,
        "attn_proto_blend": 0.6,
        "attn_proto_temp": 0.2,
        "attn_proto_usage_boost": 0.1,
        "attn_shortlist_alpha_schedule": {
            "switch_ctx": 32768,
            "low_ctx": 4096,
        },
        "token_keep_ratio": 0.12,
        "per_layer_overrides": {},
    },
    "training_recipe": {
        "notes": "See README_PROTEUS.md for staged curriculum guidance.",
    },
}


@dataclass
class BaselineMetrics:
    loss: float
    ppl: float
    total_tokens: int
    chunks: int
    seq_len: int
    stride: int
    device: str
    dataset_path: str


@dataclass
class TrainingOptions:
    heads: int
    gate_ratio: float
    strategy: str  # "fixed" or "context_mastery"
    target_ctx: Optional[int]
    context_mastery_settings: Optional[Dict[str, Any]]


@dataclass
class StagePlan:
    name: str
    description: str
    max_steps: int
    sequence_length: int
    shortlist_alpha_start: float
    shortlist_alpha_end: float


def prompt(text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        response = input(f"{text}{suffix}: ")
    except EOFError:  # pragma: no cover - piped invocation
        response = ""
    if not response and default is not None:
        return default
    return response.strip()


def prompt_yes_no(text: str, default: bool = True) -> bool:
    options = "Y/n" if default else "y/N"
    while True:
        resp = prompt(f"{text} ({options})")
        if not resp:
            return default
        if resp.lower() in {"y", "yes"}:
            return True
        if resp.lower() in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def resolve_model_path(path_arg: Optional[str]) -> Path:
    if path_arg:
        candidate = Path(path_arg).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Model path '{candidate}' does not exist")
    while True:
        response = prompt("Enter Hugging Face model directory")
        if not response:
            print("A path is required; try again.")
            continue
        candidate = Path(response).expanduser().resolve()
        if candidate.exists():
            return candidate
        print(f"Path '{candidate}' not found; please retry.")


def resolve_dataset_path(path_arg: Optional[str]) -> Path:
    if path_arg:
        candidate = Path(path_arg).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Dataset path '{candidate}' does not exist")
    while True:
        response = prompt("Enter evaluation dataset (file or directory)")
        if not response:
            print("A dataset path is required; try again.")
            continue
        candidate = Path(response).expanduser().resolve()
        if candidate.exists():
            return candidate
        print(f"Path '{candidate}' not found; please retry.")


def list_text_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files: List[Path] = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in {".txt", ".md", ".jsonl", ".log"}:
            files.append(candidate)
    if not files:
        raise RuntimeError(f"No text-like files found under {path}")
    return files


def iter_text_samples(path: Path) -> Iterator[str]:
    files = list_text_files(path)
    for file in files:
        try:
            yield file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"[warn] skipping non-UTF8 file {file}")


def evaluate_baseline(
    model_dir: Path,
    dataset_path: Path,
    seq_len: int = 2048,
    stride: int = 1024,
) -> BaselineMetrics:
    print("\n=== Baseline Evaluation ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading tokenizer/model from {model_dir} (device={device})...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    chunks = 0

    texts = list(iter_text_samples(dataset_path))
    if not texts:
        raise RuntimeError("No usable text samples for baseline evaluation.")

    print(f"Evaluating on {len(texts)} text sample(s)...")
    for idx, text in enumerate(texts, start=1):
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        length = input_ids.size(1)
        if length < 2:
            continue
        max_start = max(1, length - 1)
        start = 0
        while start < max_start:
            end = min(start + seq_len, length)
            chunk = input_ids[:, start:end]
            if chunk.size(1) < 2:
                break
            labels = chunk.clone()
            labels[:, :-1] = chunk[:, 1:]
            labels[:, -1] = -100
            attn = None
            if attention_mask is not None:
                attn = attention_mask[:, start:end]
            with torch.no_grad():
                outputs = model(chunk, attention_mask=attn, labels=labels)
                loss = outputs.loss
            tokens = chunk.size(1) - 1
            total_loss += loss.item() * tokens
            total_tokens += tokens
            chunks += 1
            start += stride
            print(
                f"Sample {idx}/{len(texts)} | chunk {chunks} | loss={loss.item():.4f} | tokens={total_tokens}",
                end="\r",
                flush=True,
            )
    print("")
    if total_tokens == 0:
        raise RuntimeError("Dataset too small; no tokens evaluated.")
    mean_loss = total_loss / total_tokens
    ppl = math.exp(mean_loss)
    print(f"Baseline: loss={mean_loss:.4f}, ppl={ppl:.2f}, tokens={total_tokens}")
    return BaselineMetrics(
        loss=mean_loss,
        ppl=ppl,
        total_tokens=total_tokens,
        chunks=chunks,
        seq_len=seq_len,
        stride=stride,
        device=str(device),
        dataset_path=str(dataset_path),
    )


def derive_destination(src: Path, explicit: Optional[str]) -> Path:
    if explicit:
        dest = Path(explicit).expanduser().resolve()
    else:
        dest = src.parent / f"{src.name}-Proteus"
    if dest.exists():
        raise FileExistsError(f"Destination '{dest}' already exists")
    return dest


def copy_model_tree(src: Path, dest: Path) -> None:
    shutil.copytree(src, dest)


def augment_config(config_path: Path) -> None:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unable to parse {config_path}: {exc}") from exc
    for key, value in DEFAULT_CONFIG_INJECTION.items():
        payload.setdefault(key, value)
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_training_notes(dest: Path, src: Path) -> None:
    readme = dest / "README_PROTEUS.md"
    content = f"""# Proteus Attention Workspace

Source model: `{src}`

This directory mirrors the original Hugging Face checkpoint and adds metadata
for Proteus Adaptive Sparse Attention integration. Follow the staged curriculum
documented in `docs/proteus_integration_guide.md`.
"""
    readme.write_text(content, encoding="utf-8")


def collect_training_options(default_heads: Optional[int]) -> TrainingOptions:
    print("\n=== Configuration ===")
    while True:
        heads_str = prompt("Number of attention heads", default=str(default_heads or 16))
        try:
            heads = int(heads_str)
            if heads <= 0:
                raise ValueError
            break
        except ValueError:
            print("Heads must be a positive integer.")
    while True:
        ratio_str = prompt("Gate-to-head ratio (>=2.0)", default="2.0")
        try:
            ratio = float(ratio_str)
            if ratio < 2.0:
                raise ValueError
            break
        except ValueError:
            print("Ratio must be a number >= 2.0.")

    strategy_choice = prompt("Context strategy: 'fixed' or 'context_mastery'", default="fixed").lower()
    if strategy_choice not in {"fixed", "context_mastery"}:
        print("Unknown choice; defaulting to 'fixed'.")
        strategy_choice = "fixed"
    target_ctx: Optional[int] = None
    mastery_settings: Optional[Dict[str, Any]] = None
    if strategy_choice == "fixed":
        while True:
            ctx_str = prompt("Target context length", default="16384")
            try:
                target_ctx = int(ctx_str)
                if target_ctx <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Context length must be a positive integer.")
    else:
        mastery_settings = {
            "alpha_increment": 0.05,
            "plateau_window": 400,
            "min_improvement": 0.005,
        }
        print("Context mastery selected; alpha increments by 0.05 on plateau.")

    return TrainingOptions(
        heads=heads,
        gate_ratio=ratio,
        strategy=strategy_choice,
        target_ctx=target_ctx,
        context_mastery_settings=mastery_settings,
    )


def default_stage_plan(options: TrainingOptions) -> List[StagePlan]:
    return [
        StagePlan(
            name="Stage A",
            description="Prototype warm-up (dense mode, routers only)",
            max_steps=8000,
            sequence_length=4096,
            shortlist_alpha_start=0.0,
            shortlist_alpha_end=0.0,
        ),
        StagePlan(
            name="Stage B",
            description="Sparse head ramp (unfreeze attention/MLP)",
            max_steps=30000,
            sequence_length=8192,
            shortlist_alpha_start=0.0,
            shortlist_alpha_end=1.0,
        ),
        StagePlan(
            name="Stage C",
            description="Long-context curriculum with chunked shortlist",
            max_steps=120000,
            sequence_length=options.target_ctx or 65536,
            shortlist_alpha_start=1.0,
            shortlist_alpha_end=1.0,
        ),
        StagePlan(
            name="Stage D",
            description="Optional consolidation (prototypes/controllers)",
            max_steps=15000,
            sequence_length=options.target_ctx or 65536,
            shortlist_alpha_start=1.0,
            shortlist_alpha_end=1.0,
        ),
    ]


def mk_run_directory(dest: Path) -> Path:
    runs_dir = dest / "proteus_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_dir / f"run-{timestamp}"
    run_dir.mkdir()
    return run_dir


def write_metrics(run_dir: Path, baseline: BaselineMetrics) -> None:
    metrics_path = run_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(asdict(baseline), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_plan(run_dir: Path, baseline: BaselineMetrics, options: TrainingOptions, stages: List[StagePlan]) -> None:
    plan_path = run_dir / "stage_plan.json"
    payload = {
        "training_options": asdict(options),
        "stages": [asdict(stage) for stage in stages],
        "dataset": baseline.dataset_path,
    }
    plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Also create a CSV for quick browsing.
    csv_path = run_dir / "stage_plan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "description", "max_steps", "sequence_length", "alpha_start", "alpha_end"])
        for stage in stages:
            writer.writerow([
                stage.name,
                stage.description,
                stage.max_steps,
                stage.sequence_length,
                stage.shortlist_alpha_start,
                stage.shortlist_alpha_end,
            ])


def write_run_summary(run_dir: Path, baseline: BaselineMetrics, options: TrainingOptions) -> None:
    summary = run_dir / "SUMMARY.md"
    opts = asdict(options)
    summary.write_text(
        """# Proteus Conversion Run Summary

## Baseline

- Loss: {loss:.4f}
- Perplexity: {ppl:.2f}
- Tokens evaluated: {tokens}
- Device: {device}
- Dataset: {dataset}

## Training Options

- Heads: {heads}
- Gate ratio: {gate_ratio}
- Strategy: {strategy}
- Target context: {target}
- Context mastery: {mastery}

""".format(
            loss=baseline.loss,
            ppl=baseline.ppl,
            tokens=baseline.total_tokens,
            device=baseline.device,
            dataset=baseline.dataset_path,
            heads=options.heads,
            gate_ratio=options.gate_ratio,
            strategy=options.strategy,
            target=options.target_ctx or "n/a",
            mastery=json.dumps(options.context_mastery_settings, indent=2) if options.context_mastery_settings else "n/a",
        ),
        encoding="utf-8",
    )


def describe_pauses() -> None:
    print(
        "\nDuring training the CLI will pause after each stage, present a menu (view metrics,"
        " adjust corpus/hyperparameters, or stop), and only refresh a single status"
        " panel in the terminal to avoid log spam. All metrics are captured under"
        " the run directory for later analysis."
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive Proteus conversion CLI")
    parser.add_argument("--model-path", help="Path to the base HF checkpoint directory.")
    parser.add_argument("--dataset-path", help="Path to evaluation corpus (file or directory).")
    parser.add_argument("--output-path", help="Optional explicit destination for the Proteus workspace.")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length for baseline evaluation.")
    parser.add_argument("--stride", type=int, default=1024, help="Stride for baseline evaluation.")

    args = parser.parse_args(argv)

    model_path = resolve_model_path(args.model_path)
    dataset_path = resolve_dataset_path(args.dataset_path)

    baseline = evaluate_baseline(model_path, dataset_path, seq_len=args.seq_len, stride=args.stride)

    print("\nProteus attention is still experimental; improvements are not guaranteed.")
    proceed = prompt_yes_no("Proceed with Proteus conversion?", default=True)
    if not proceed:
        print("Aborting; workspace not created.")
        return

    destination = derive_destination(model_path, args.output_path)
    print(f"Copying checkpoint to {destination} ...")
    copy_model_tree(model_path, destination)

    config_path = destination / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Converted directory missing config.json at {config_path}")
    augment_config(config_path)
    write_training_notes(destination, model_path)

    options = collect_training_options(default_heads=None)
    stages = default_stage_plan(options)

    run_dir = mk_run_directory(destination)
    write_metrics(run_dir, baseline)
    write_plan(run_dir, baseline, options, stages)
    write_run_summary(run_dir, baseline, options)
    describe_pauses()

    print("\nConversion complete.")
    print(f"Proteus workspace: {destination}")
    print(f"Run artifacts: {run_dir}")
    print("Review the stage plan and launch the training harness to begin the staged curriculum.")


if __name__ == "__main__":
    main()
