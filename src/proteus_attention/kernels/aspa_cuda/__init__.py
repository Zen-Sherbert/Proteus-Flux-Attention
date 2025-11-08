"""
Runtime-built CUDA backend for the ASPA sparse attention kernel.

The extension is compiled on demand via ``torch.utils.cpp_extension.load`` so
that environments like Google Colab can build it without a dedicated wheel. We
only attempt to build when the user opts in by exporting
``ASPA_USE_CUDA_KERNEL=1``.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_BUILD_DIR = _HERE / "_build"
_BUILD_DIR.mkdir(parents=True, exist_ok=True)

_autobuild_flag = os.getenv("ASPA_CUDA_AUTOBUILD", "1")
if _autobuild_flag.strip().lower() in {"0", "false", "no"}:
    raise RuntimeError(
        "Automatic ASPA CUDA backend build disabled by ASPA_CUDA_AUTOBUILD."
    )

_SOURCES = [str(_HERE / "aspa_cuda.cpp")]
_MODULE_NAME = "aspa_cuda_ext"

_verbose_flag = os.getenv("ASPA_CUDA_VERBOSE", "0")
_VERBOSE = _verbose_flag.strip().lower() not in {
    "0",
    "false",
    "no",
}

_EXTRA_CFLAGS = ["-O3"]
_EXTRA_CUDA_CFLAGS = ["-O3", "--use_fast_math"]
if torch.version.cuda is None:
    # No CUDA toolkit available; fall back to pure C++ build.
    _EXTRA_CUDA_CFLAGS = []

_EXTENSION = load(
    name=_MODULE_NAME,
    sources=_SOURCES,
    extra_cflags=_EXTRA_CFLAGS,
    extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
    with_cuda=torch.cuda.is_available(),
    build_directory=str(_BUILD_DIR),
    verbose=_VERBOSE,
)

aspa_sparse_attention = _EXTENSION.aspa_sparse_attention

__all__ = ["aspa_sparse_attention"]
