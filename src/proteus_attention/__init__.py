"""
Proteus Attention package: Genetic Attention reference implementation powered by
the Dynamic Mixture-of-Attention-Heads (DMoAH) architecture.

This module exposes the primary building blocks so downstream projects can
install ``proteus-attention`` and import Genetic Attention kernels or models
directly while still accessing the DMoAH internals when needed.
"""

from .models.dmoah import (
    AdaptiveSparseAttentionBlock,
    AdaptiveSparseAttention,
    AdaptiveSparseProtoAttention,
    MLP,
    ModelConfig,
)
from .modules import CausalGeneticMultiheadAttention
from .kernels.sparse_attn import (
    dmoah_sparse_attention,
    get_last_backend,
    get_last_backend_info,
)
from .tools.chunked_shortlist import (
    ChunkedShortlistConfig,
    ChunkedShortlistMetrics,
    ChunkedShortlistResult,
    ChunkedShortlistRunner,
)

__all__ = [
    "AdaptiveSparseAttentionBlock",
    "AdaptiveSparseAttention",
    "AdaptiveSparseProtoAttention",
    "CausalGeneticMultiheadAttention",
    "MLP",
    "ModelConfig",
    "dmoah_sparse_attention",
    "get_last_backend",
    "get_last_backend_info",
    "ChunkedShortlistConfig",
    "ChunkedShortlistMetrics",
    "ChunkedShortlistResult",
    "ChunkedShortlistRunner",
    "__version__",
]

__version__ = "0.1.0"
