"""
Model components for the Protean Forge package.

Re-export key transformer components so that consumers can import from
`protean_forge.models` without diving into implementation files.
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
