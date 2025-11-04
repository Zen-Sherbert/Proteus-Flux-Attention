"""
Proteus Attention package: Genetic Attention reference implementation powered by
the Dynamic Mixture-of-Attention-Heads (DMoAH) architecture.

This module exposes the primary building blocks so downstream projects can
install ``proteus-attention`` and import Genetic Attention kernels or models
directly while still accessing the DMoAH internals when needed.
"""

from .models.dmoah import (
    AttentionBlock,
    CausalDynamicAttention,
    CausalGeneticAttention,
    MLP,
    ModelConfig,
)
from .modules import CausalGeneticMultiheadAttention
from .kernels.sparse_attn import (
    dmoah_sparse_attention,
    get_last_backend,
    get_last_backend_info,
)
from .tools.chunked_flux import (
    ChunkedFluxConfig,
    ChunkedFluxMetrics,
    ChunkedFluxResult,
    ChunkedFluxRunner,
)

__all__ = [
    "AttentionBlock",
    "CausalDynamicAttention",
    "CausalGeneticAttention",
    "CausalGeneticMultiheadAttention",
    "MLP",
    "ModelConfig",
    "dmoah_sparse_attention",
    "get_last_backend",
    "get_last_backend_info",
    "ChunkedFluxConfig",
    "ChunkedFluxMetrics",
    "ChunkedFluxResult",
    "ChunkedFluxRunner",
    "__version__",
]

__version__ = "0.1.0"
