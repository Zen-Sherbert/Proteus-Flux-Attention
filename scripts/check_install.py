#!/usr/bin/env python3
"""
Quick sanity check for Protean Forge installations.

The script validates that PyTorch and Triton (when available) can import the
Proteus Flux-enabled attention modules, then runs a tiny forward pass on CPU and, if a
GPU/ROCm device is visible, repeats the exercise with the Proteus Flux slider forced to
its linear regime.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

import torch

from protean_forge.models.dmoah import CausalDynamicAttention, ModelConfig


@dataclass
class CheckResult:
    label: str
    device: torch.device
    success: bool
    message: str


def _run_smoke(device: torch.device, *, flux_alpha: float) -> CheckResult:
    torch.manual_seed(0)
    batch, seq_len, d_model, heads = 2, 128, 64, 4
    cfg = ModelConfig(
        d_model=d_model,
        n_head=heads,
        attn_mode="auto",
        n_ctx=seq_len,
        attn_linear_switch_ctx=64,
        attn_linear_L=32,
        attn_linear_L_min=16,
        attn_linear_L_max=32,
    )
    model = CausalDynamicAttention(cfg).to(device=device)
    if hasattr(model, "set_flux_alpha"):
        model.set_flux_alpha(flux_alpha)
    x = torch.randn(batch, seq_len, d_model, device=device, dtype=torch.get_default_dtype())
    try:
        with torch.inference_mode():
            y = model(x)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            label=f"Proteus Flux smoke (alpha={flux_alpha})",
            device=device,
            success=False,
            message=f"{type(exc).__name__}: {exc}",
        )
    return CheckResult(
        label=f"Proteus Flux smoke (alpha={flux_alpha})",
        device=device,
        success=True,
        message=f"output={tuple(y.shape)}",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Protean Forge installation.")
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip the optional GPU/ROCm smoke test even if a device is available.",
    )
    args = parser.parse_args(argv)

    results: list[CheckResult] = []
    # CPU baseline (no flux override so the module auto-selects dense/subquad paths)
    results.append(_run_smoke(torch.device("cpu"), flux_alpha=0.0))

    if not args.no_gpu and torch.cuda.is_available():
        results.append(_run_smoke(torch.device("cuda"), flux_alpha=1.0))
    elif not args.no_gpu and torch.backends.mps.is_available():  # pragma: no cover - macOS convenience
        results.append(_run_smoke(torch.device("mps"), flux_alpha=1.0))

    failed = [result for result in results if not result.success]
    for result in results:
        status = "OK" if result.success else "FAIL"
        print(f"[{status}] {result.label} on {result.device}: {result.message}")

    if failed:
        print("\nOne or more checks failed. See messages above.", file=sys.stderr)
        return 1
    print("\nAll Protean Forge smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
