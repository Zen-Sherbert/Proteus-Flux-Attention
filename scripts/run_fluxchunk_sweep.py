#!/usr/bin/env python
"""
FluxChunk sweep: run large-context tests (1M–10M tokens) and report latency,
throughput, retention, and VRAM for each size.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

try:
    from torch.nn.attention import sdpa_kernel as nn_sdpa_kernel  # PyTorch 2.5+
except Exception:  # pragma: no cover - optional
    nn_sdpa_kernel = None  # type: ignore[assignment]

import torch

from proteus_attention.tools.chunked_flux import (
    ChunkedFluxConfig,
    ChunkedFluxRunner,
    MAX_FLUX_CHUNK_TOKENS,
)


def _sdpa_override(device: torch.device):
    """
    Return a context manager that forces PyTorch SDPA into a safe mode or a no-op.

    We rely on the custom Flux kernels, but on some builds PyTorch may still try
    to pick Flash/efficient SDPA kernels by default.  When the backend exposes a
    context manager (torch.backends.cuda.sdp_kernel or nn_sdpa_kernel), we
    disable the Flash/MemEfficient paths.  If the signature is incompatible we
    gracefully fall back to a nullcontext so that the sweep never crashes.
    """

    if device.type != "cuda":
        return nullcontext()

    backend_sdpa = None
    if nn_sdpa_kernel is not None:
        backend_sdpa = nn_sdpa_kernel
    else:
        backend_sdpa = getattr(torch.backends.cuda, "sdpa_kernel", None)

    if backend_sdpa is None:
        return nullcontext()

    try:
        ctx = backend_sdpa(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )
    except TypeError:
        try:
            ctx = backend_sdpa(enable_flash=False, enable_mem_efficient=False)
        except TypeError:
            return nullcontext()

    if hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__"):
        return ctx
    return nullcontext()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FluxChunk long-context sweep.")
    parser.add_argument("--device", default=None, help="Device string (default autodetect).")
    parser.add_argument("--d-model", type=int, default=128, help="Embedding width.")
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=MAX_FLUX_CHUNK_TOKENS,
        help=f"Streaming chunk length (capped at {MAX_FLUX_CHUNK_TOKENS}).",
    )
    parser.add_argument(
        "--per-chunk-budget", type=int, default=4_096, help="Tokens promoted per chunk."
    )
    parser.add_argument(
        "--flux-alpha",
        type=float,
        default=1.0,
        help="Flux alpha slider (0=dense, 1=fully sparse linear shortlist).",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01] + [0.02] * 7,
        help="Buffer ratios for each step (1M..10M).",
    )
    parser.add_argument(
        "--report-latency",
        action="store_true",
        help="Capture CUDA events for chunk/final timings.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to append CSV lines (seq_len,buffer,chunk_ms,chunk_tps,final_ms,final_tps,total_ms,total_tps,peak_mb,retained).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_handle = None
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_file.open("a", encoding="utf-8")
        if log_handle.tell() == 0:
            log_handle.write(
                "seq_len,buffer,chunk_ms,chunk_tps,final_ms,final_tps,total_ms,total_tps,peak_mb,retained\n"
            )

    chunk_warned = False
    for idx, ratio in enumerate(args.ratios, start=1):
        seq_len = idx * 1_000_000
        buffer_tokens = max(1, int(seq_len * ratio))
        chunk_len = min(args.chunk_len, MAX_FLUX_CHUNK_TOKENS)
        if chunk_len < args.chunk_len and not chunk_warned:
            print(
                f"[FluxChunk] Requested chunk_len={args.chunk_len:,} exceeds "
                f"cap {MAX_FLUX_CHUNK_TOKENS:,}; using {chunk_len:,} instead."
            )
            chunk_warned = True
        cfg = ChunkedFluxConfig(
            seq_len=seq_len,
            d_model=args.d_model,
            chunk_len=chunk_len,
            buffer_tokens=buffer_tokens,
            per_chunk_budget=args.per_chunk_budget,
            device=device,
            heads=8,
            chunk_sparse_ratio=0.05,
            final_sparse_ratio=0.5,
            seed=123,
            report_latency=args.report_latency and device.type == "cuda",
            progress=False,
            run_final_pass=True,
            flux_alpha=args.flux_alpha,
        )
        runner = ChunkedFluxRunner(cfg)
        print(
            f"[FluxChunk] seq_len={seq_len:,} buffer={buffer_tokens:,} "
            f"alpha={args.flux_alpha:.3f} device={device}"
        )
        ctx = _sdpa_override(device)
        with ctx:
            result = runner.run()
        metrics = result.metrics
        print(
            f"  chunk: {metrics.chunk_time_ms:.1f} ms | {metrics.chunk_tokens_per_s:,.0f} tok/s"
            if metrics.chunk_time_ms is not None and metrics.chunk_tokens_per_s is not None
            else "  chunk: n/a"
        )
        if metrics.final_time_ms is not None:
            print(
                f"  final: {metrics.final_time_ms:.1f} ms | {metrics.final_tokens_per_s:,.0f} tok/s"
            )
        if metrics.peak_memory_mb is not None:
            print(f"  peak VRAM: {metrics.peak_memory_mb:.1f} MB")
        print(
            f"  total: {metrics.total_time_ms:.1f} ms | {metrics.total_tokens_per_s:,.0f} tok/s"
            if metrics.total_time_ms is not None and metrics.total_tokens_per_s is not None
            else "  total: n/a"
        )
        print(
            f"  retained={metrics.retained_tokens:,} ({metrics.retention_ratio:.4f}) "
            f"| fallback={metrics.used_fallback}"
        )

        if log_handle is not None:
            log_handle.write(
                f"{seq_len},{buffer_tokens},"
                f"{metrics.chunk_time_ms or 0.0},{metrics.chunk_tokens_per_s or 0.0},"
                f"{metrics.final_time_ms or 0.0},{metrics.final_tokens_per_s or 0.0},"
                f"{metrics.total_time_ms or 0.0},{metrics.total_tokens_per_s or 0.0},"
                f"{metrics.peak_memory_mb or 0.0},{metrics.retained_tokens}\n"
            )
            log_handle.flush()

    if log_handle is not None:
        log_handle.close()


if __name__ == "__main__":
    main()
