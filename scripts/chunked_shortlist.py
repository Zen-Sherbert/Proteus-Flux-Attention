#!/usr/bin/env python
"""
Professional-grade Chunked Shortlist CLI utility.

The tool exposes the reusable :mod:`proteus_attention.tools.chunked_shortlist`
pipeline via a convenient command-line interface.  It can operate on synthetic
sequences or on user-provided embeddings saved as ``.pt``/``.npy`` tensors.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from proteus_attention.tools.chunked_shortlist import (
    ChunkedShortlistConfig,
    ChunkedShortlistRunner,
)

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _load_sequence(path: Path, *, seq_len: int, d_model: int) -> torch.Tensor:
    """Load an input tensor from ``.pt`` or ``.npy`` files."""

    if path.suffix == ".pt":
        tensor = torch.load(path)
    elif path.suffix == ".npy":
        if np is None:
            raise RuntimeError("NumPy is required to load .npy inputs (pip install numpy).")
        array = np.load(path)
        tensor = torch.from_numpy(array)
    else:
        raise ValueError(f"Unsupported input format '{path.suffix}'. Expected .pt or .npy.")

    if tensor.dim() != 3:
        raise ValueError(f"Input tensor must have shape (B, T, D); received {tensor.shape}.")
    if tensor.size(0) != 1:
        raise ValueError("Chunked Shortlist currently expects a batch dimension of size 1.")
    if tensor.size(1) < seq_len:
        raise ValueError(
            f"Input sequence length {tensor.size(1)} is smaller than requested seq_len={seq_len}."
        )
    if tensor.size(2) != d_model:
        raise ValueError(
            f"Input hidden size {tensor.size(2)} does not match configured d_model={d_model}."
        )
    return tensor[:, :seq_len].contiguous()


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)


def _save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunked Shortlist pipeline utility.")
    parser.add_argument("--seq-len", type=int, default=1_000_000, help="Total sequence length to process.")
    parser.add_argument("--d-model", type=int, default=512, help="Model width for embeddings.")
    parser.add_argument("--chunk-len", type=int, default=131_072, help="Tokens per streaming chunk.")
    parser.add_argument("--buffer-tokens", type=int, default=32_768, help="Maximum tokens kept after chunking.")
    parser.add_argument("--per-chunk-budget", type=int, default=4_096, help="Tokens promoted from each chunk.")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads for both passes.")
    parser.add_argument("--chunk-sparse-ratio", type=float, default=0.05, help="Keep ratio used in the chunk streaming stage.")
    parser.add_argument("--final-sparse-ratio", type=float, default=0.5, help="Keep ratio for the final pass.")
    parser.add_argument("--seed", type=int, default=7, help="Seed used when generating synthetic inputs.")
    parser.add_argument("--shortlist-alpha", type=float, default=1.0, help="Shortlist alpha slider (0=dense, 1=linear shortlist).")
    parser.add_argument("--nucleus-top-p", type=float, default=0.9, help="Top-p (nucleus) filter applied during chunk promotion.")
    parser.add_argument("--device", default=None, help="Target device (defaults to CUDA if available).")
    parser.add_argument("--progress", action="store_true", help="Display chunk progress (requires tqdm).")
    parser.add_argument("--report-latency", action="store_true", help="Capture CUDA events for runtime metrics.")
    parser.add_argument("--no-final-pass", action="store_true", help="Skip the final attention pass.")
    parser.add_argument("--storage", choices=["auto", "cpu", "disk"], default="auto", help="Where to stage the streamed sequence (cpu/disk/auto).")
    parser.add_argument("--temp-dir", type=Path, help="Scratch directory used when spilling chunks to disk.")
    parser.add_argument("--ram-limit-mb", type=int, help="Soft cap for host RAM usage; spill to disk when exceeded.")
    parser.add_argument("--input", type=Path, help="Optional input tensor (.pt or .npy) with shape (1, seq_len, d_model).")
    parser.add_argument("--save-indices", type=Path, help="Optional path to save the retained indices (.pt).")
    parser.add_argument("--save-reduced", type=Path, help="Optional path to save the reduced sequence (.pt).")
    parser.add_argument("--save-final", type=Path, help="Optional path to save the final pass output (.pt).")
    parser.add_argument("--save-report", type=Path, help="Optional JSON report destination.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    sequence: Optional[torch.Tensor] = None
    if args.input:
        sequence = _load_sequence(Path(args.input), seq_len=args.seq_len, d_model=args.d_model)
        logging.info("Loaded input tensor from %s", args.input)

    ram_limit_bytes = int(args.ram_limit_mb * 1024 * 1024) if args.ram_limit_mb else None

    config = ChunkedShortlistConfig(
        seq_len=args.seq_len,
        d_model=args.d_model,
        chunk_len=args.chunk_len,
        buffer_tokens=args.buffer_tokens,
        per_chunk_budget=args.per_chunk_budget,
        device=device,
        heads=args.heads,
        chunk_sparse_ratio=args.chunk_sparse_ratio,
        final_sparse_ratio=args.final_sparse_ratio,
        seed=args.seed,
        report_latency=args.report_latency,
        progress=args.progress,
        run_final_pass=not args.no_final_pass,
        shortlist_alpha=args.shortlist_alpha,
        nucleus_top_p=args.nucleus_top_p,
        storage=args.storage,
        temp_dir=args.temp_dir,
        ram_limit_bytes=ram_limit_bytes,
    )

    runner = ChunkedShortlistRunner(config)
    result = runner.run(sequence=sequence)
    metrics = result.metrics

    logging.info(
        "Chunked Shortlist complete: retained %s/%s tokens (%.2f%%) on %s",
        metrics.retained_tokens,
        metrics.original_tokens,
        metrics.retention_ratio * 100.0,
        metrics.device,
    )
    if metrics.chunk_time_ms is not None:
        logging.info("Chunk streaming latency: %.1f ms", metrics.chunk_time_ms)
    if metrics.chunk_tokens_per_s is not None:
        logging.info("Chunk throughput: %.2f tok/s", metrics.chunk_tokens_per_s)
    if metrics.final_time_ms is not None:
        logging.info("Final pass latency: %.1f ms", metrics.final_time_ms)
    if metrics.final_tokens_per_s is not None:
        logging.info("Final pass throughput: %.2f tok/s", metrics.final_tokens_per_s)
    if metrics.peak_memory_mb is not None:
        logging.info("Final pass peak memory: %.1f MB", metrics.peak_memory_mb)
    if metrics.total_tokens_per_s is not None:
        logging.info("Overall throughput: %.2f tok/s", metrics.total_tokens_per_s)
    logging.info("Storage mode: %s", metrics.storage_mode)
    if metrics.storage_reason:
        logging.info("Storage reasoning: %s", metrics.storage_reason)
    if metrics.host_required_mb is not None:
        logging.info("Host requirement: %.1f MB", metrics.host_required_mb)
    if metrics.host_allocated_mb is not None:
        logging.info("Host allocated: %.1f MB", metrics.host_allocated_mb)
    if metrics.host_limit_mb is not None:
        logging.info("Host limit: %.1f MB", metrics.host_limit_mb)

    if args.save_indices:
        _save_tensor(Path(args.save_indices), result.keep_indices)
        logging.info("Saved keep indices to %s", args.save_indices)
    if args.save_reduced:
        _save_tensor(Path(args.save_reduced), result.reduced_sequence)
        logging.info("Saved reduced sequence to %s", args.save_reduced)
    if args.save_final and result.final_output is not None:
        _save_tensor(Path(args.save_final), result.final_output)
        logging.info("Saved final output tensor to %s", args.save_final)
    if args.save_report:
        report = {
            "metrics": asdict(metrics),
            "keep_indices": result.keep_indices.detach().cpu().tolist(),
            "keep_scores": result.keep_scores.detach().cpu().tolist(),
            "chunks": [asdict(chunk) for chunk in result.chunks],
            "backend_info": result.backend_info,
            "final_stats": result.final_stats,
        }
        _save_report(Path(args.save_report), report)
        logging.info("Saved JSON report to %s", args.save_report)


if __name__ == "__main__":
    main()
