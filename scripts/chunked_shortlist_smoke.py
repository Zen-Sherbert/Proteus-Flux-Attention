#!/usr/bin/env python
"""
Quick smoke-test CLI for the Chunked Shortlist pipeline.

The goal is to provide a minimal command that runs the reusable
``ChunkedShortlistRunner`` with synthetic data so users can sanity check their
environment before launching heavier benchmarks.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import os
import torch

from proteus_attention.tools.chunked_shortlist import (
    ChunkedShortlistConfig,
    ChunkedShortlistRunner,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a lightweight Chunked Shortlist smoke test."
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1_000_000,
        help="Total sequence length used for the synthetic input.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Hidden size (embedding width) of the synthetic input.",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=65_536,
        help="Tokens per streaming chunk (capped internally for ROCm safety).",
    )
    parser.add_argument(
        "--buffer-tokens",
        type=int,
        default=32_768,
        help="Maximum number of tokens retained after streaming.",
    )
    parser.add_argument(
        "--per-chunk-budget",
        type=int,
        default=4_096,
        help="Tokens promoted from each chunk into the global shortlist.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Attention heads used in both the streaming and final passes.",
    )
    parser.add_argument(
        "--chunk-sparse-ratio",
        type=float,
        default=0.05,
        help="Keep ratio applied during the streaming stage.",
    )
    parser.add_argument(
        "--final-sparse-ratio",
        type=float,
        default=0.5,
        help="Keep ratio applied during the final pass.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Target device (e.g. 'cuda', 'cpu', 'cuda:0'); defaults to CUDA when available.",
    )
    parser.add_argument(
        "--storage",
        choices=["auto", "cpu", "disk"],
        default="cpu",
        help="Stage chunk data in RAM ('cpu'), spill to disk ('disk'), or let the runner choose ('auto').",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Optional scratch directory when using --storage=disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="PRNG seed for the synthetic input.",
    )
    parser.add_argument(
        "--report-latency",
        action="store_true",
        help="Capture CUDA event timings for streaming and final passes.",
    )
    parser.add_argument(
        "--no-final-pass",
        action="store_true",
        help="Skip the final attention pass to measure streaming alone.",
    )
    return parser.parse_args()


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("PROTEUS_TUNE_DISABLE", "1")
    device = _resolve_device(args.device)

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
        progress=True,
        run_final_pass=not args.no_final_pass,
        storage=args.storage,
        temp_dir=args.temp_dir,
    )

    runner = ChunkedShortlistRunner(config)
    result = runner.run()
    metrics = result.metrics

    print("\n=== Chunked Shortlist Smoke Test ===")
    print(f"device              : {metrics.device}")
    print(f"sequence length     : {metrics.original_tokens}")
    print(f"retained tokens     : {metrics.retained_tokens} ({metrics.retention_ratio:.2%})")
    print(f"chunks processed    : {metrics.chunk_count}")
    if metrics.chunk_time_ms is not None:
        print(f"stream latency (ms) : {metrics.chunk_time_ms:.2f}")
    if metrics.chunk_tokens_per_s is not None:
        print(f"stream throughput   : {metrics.chunk_tokens_per_s:,.2f} tok/s")
    if metrics.final_time_ms is not None:
        print(f"final latency (ms)  : {metrics.final_time_ms:.2f}")
    if metrics.final_tokens_per_s is not None:
        print(f"final throughput    : {metrics.final_tokens_per_s:,.2f} tok/s")
    if metrics.peak_memory_mb is not None:
        print(f"final peak memory   : {metrics.peak_memory_mb:.1f} MB")
    if metrics.total_tokens_per_s is not None:
        print(f"overall throughput  : {metrics.total_tokens_per_s:,.2f} tok/s")
    print("==============================\n")


if __name__ == "__main__":
    main()
