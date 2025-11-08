"""
Proteus Attention package: Adaptive Sparse Proto Attention (ASPA) reference
implementation powered by the Dynamic Mixture-of-Attention-Heads architecture.

This module exposes the primary building blocks so downstream projects can
install ``proteus-attention`` and import ASPA kernels or models directly while
still accessing the underlying sparsity controls when needed.
"""

from .models.aspa import (
    AdaptiveSparseAttentionBlock,
    AdaptiveSparseAttention,
    AdaptiveSparseProtoAttention,
    MLP,
    ModelConfig,
)
from .modules import CausalASPAMultiheadAttention
from .kernels.sparse_attn import (
    aspa_sparse_attention,
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
    "aspa_sparse_attention",
    "CausalASPAMultiheadAttention",
    "MLP",
    "ModelConfig",
    "get_last_backend",
    "get_last_backend_info",
    "ChunkedShortlistConfig",
    "ChunkedShortlistMetrics",
    "ChunkedShortlistResult",
    "ChunkedShortlistRunner",
    "__version__",
]

__version__ = "0.1.0"
