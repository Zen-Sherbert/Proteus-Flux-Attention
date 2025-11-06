#!/usr/bin/env python
"""
One-shot helper to autotune the Triton block sizes for the Proteus Attention kernel.

Running this script will execute a single forward pass on random data to trigger the
automatic tuner and persist the discovered configuration in the cache directory
(`~/.cache/proteus_attention/shortlist_block_config.json` by default). Subsequent runs on
the same GPU will reuse the cached settings automatically.
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import torch

from proteus_attention.modules import CausalGeneticMultiheadAttention
from proteus_attention.kernels.sparse_attn import get_block_config_cache


def _parse_tuple(arg: str, default: Sequence[int]) -> Sequence[int]:
    if not arg:
        return default
    try:
        values = [int(item.strip()) for item in arg.split(",") if item.strip()]
    except ValueError as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"Invalid comma-separated integers: {arg}") from exc
    return values or default


def main() -> None:
    parser = argparse.ArgumentParser(description="Autotune Triton block sizes for Proteus Attention.")
    parser.add_argument("--embed-dim", type=int, default=1024, help="Embedding dimension to tune with.")
    parser.add_argument("--num-heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length used during tuning.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size for the synthetic input.")
    parser.add_argument(
        "--head-dims",
        type=lambda s: _parse_tuple(s, []),
        default="",
        help="Optional comma-separated list of head dimensions to tune explicitly.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="CUDA device index. Defaults to the current device.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for autotuning.")

    if args.device_index is not None:
        torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", torch.cuda.current_device())

    # Force a fresh autotune run.
    os.environ["PROTEUS_TUNE_FORCE"] = "1"
    os.environ["PROTEUS_TUNE_BRUTE_FORCE"] = "1"
    from proteus_attention.kernels import sparse_attn as _sparse_module

    _sparse_module._BRUTE_FORCE_ENABLED = True
    _sparse_module._BRUTE_FORCE_WARNED = False

    head_dims = list(args.head_dims) or [args.embed_dim // args.num_heads]

    for head_dim in head_dims:
        embed_dim = head_dim * args.num_heads
        model = CausalGeneticMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.num_heads,
            batch_first=True,
            max_seq_len=args.seq_len,
        ).to(device)
        x = torch.randn(args.batch, args.seq_len, embed_dim, device=device)
        with torch.inference_mode():
            model(x, need_weights=False)
        torch.cuda.synchronize(device)

    cache = get_block_config_cache()
    cache_path = _sparse_module._CACHE_FILE
    print("Autotune complete.")
    print(f"Results stored in {cache_path}")
    print("Cached configurations (per device/head_dim):")
    for key, cfg in cache.items():
        print(f"  {key}: BLOCK_M={cfg[0]}, BLOCK_N={cfg[1]}, BLOCK_D={cfg[2]}")
    print("You can rerun this script after major hardware or driver changes; otherwise the cached values are reused automatically.")


if __name__ == "__main__":
    main()
