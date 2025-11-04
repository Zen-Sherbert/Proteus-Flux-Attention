"""
Runtime-built CUDA backend for the DMoAH sparse attention kernel.

The extension is compiled on demand via ``torch.utils.cpp_extension.load`` so
that environments like Google Colab can build it without a dedicated wheel.  We
only attempt to build when the user opts in by exporting
``DMOAH_USE_CUDA_KERNEL=1`` (the parent module controls this import).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_BUILD_DIR = _HERE / "_build"
_BUILD_DIR.mkdir(parents=True, exist_ok=True)

if os.getenv("DMOAH_CUDA_AUTOBUILD", "1").strip().lower() in {"0", "false", "no"}:
    raise RuntimeError(
        "Automatic DMoAH CUDA backend build disabled by DMOAH_CUDA_AUTOBUILD."
    )

_SOURCES = [str(_HERE / "dmoah_cuda.cpp")]
_MODULE_NAME = "dmoah_cuda_ext"

_VERBOSE = os.getenv("DMOAH_CUDA_VERBOSE", "0").strip().lower() not in {
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

dmoah_sparse_attention = _EXTENSION.dmoah_sparse_attention

__all__ = ["dmoah_sparse_attention"]
