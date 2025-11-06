"""
Model components for the Proteus Attention package.

Re-export key transformer components so that consumers can import from
`proteus_attention.models` without diving into implementation files.
"""

from .dmoah import (
    AdaptiveSparseAttentionBlock,
    AdaptiveSparseAttention,
    AdaptiveSparseProtoAttention,
    MLP,
    ModelConfig,
    SwitchRouter,
)

__all__ = [
    "AdaptiveSparseAttentionBlock",
    "AdaptiveSparseAttention",
    "AdaptiveSparseProtoAttention",
    "MLP",
    "ModelConfig",
    "SwitchRouter",
]
