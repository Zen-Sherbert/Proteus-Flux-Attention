"""
Model components for the Proteus Attention package.

Re-export key transformer components so that consumers can import from
`proteus_attention.models` without diving into implementation files.
"""

from .dmoah import (
    AttentionBlock,
    CausalDynamicAttention,
    CausalGeneticAttention,
    MLP,
    ModelConfig,
    SwitchRouter,
)

__all__ = [
    "AttentionBlock",
    "CausalDynamicAttention",
    "CausalGeneticAttention",
    "MLP",
    "ModelConfig",
    "SwitchRouter",
]
