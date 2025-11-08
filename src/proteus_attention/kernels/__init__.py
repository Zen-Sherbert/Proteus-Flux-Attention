"""Kernel utilities for the Proteus Attention package."""

from .sparse_attn import (
    aspa_sparse_attention,
    build_shortlist_candidates,
    build_packed_shortlist_candidates,
    get_last_backend,
    get_last_backend_info,
)

__all__ = [
    "aspa_sparse_attention",
    "build_shortlist_candidates",
    "build_packed_shortlist_candidates",
    "get_last_backend",
    "get_last_backend_info",
]
