"""Kernel utilities for the Proteus Attention package."""

from .sparse_attn import (
    build_shortlist_candidates,
    build_packed_shortlist_candidates,
    dmoah_sparse_attention,
    get_last_backend,
    get_last_backend_info,
)

__all__ = [
    "build_shortlist_candidates",
    "build_packed_shortlist_candidates",
    "dmoah_sparse_attention",
    "get_last_backend",
    "get_last_backend_info",
]
