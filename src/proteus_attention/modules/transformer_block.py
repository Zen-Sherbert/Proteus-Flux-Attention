"""
High-level Transformer block built around :class:`CausalASPAMultiheadAttention`.

The module mirrors the constructor and forward behaviour of PyTorch's
``nn.TransformerEncoderLayer`` (in ``batch_first=True`` mode) so downstream
projects can swap in Proteus Attention without re-writing their block wiring.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aspa_multihead import CausalASPAMultiheadAttention


def _get_activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation '{name}'.")


class CausalASPATransformerBlock(nn.Module):
    """
    Transformer encoder block that mirrors ``nn.TransformerEncoderLayer`` while
    routing attention through :class:`CausalASPAMultiheadAttention`.

    Parameters are kept as close as possible to the PyTorch baseline so existing
    models can switch with minimal edits.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        attention_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        attn_args = dict(attention_kwargs or {})
        attn_args.setdefault("batch_first", True)
        attn_args.setdefault("dropout", dropout)
        self.self_attn = CausalASPAMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            **attn_args,
        )

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm_first = bool(norm_first)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        *,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with the same signature as ``nn.TransformerEncoderLayer``.

        Additional keyword arguments are forwarded to
        :class:`CausalASPAMultiheadAttention`.
        """

        residual = src
        attn_weights: Optional[torch.Tensor] = None

        if self.norm_first:
            src = self.norm1(src)

        attn_out, attn_weights = self.self_attn(
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=return_attn,
            **kwargs,
        )
        src = residual + self.dropout1(attn_out)
        if not self.norm_first:
            src = self.norm1(src)

        residual = src
        if self.norm_first:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.norm_first:
            src = self.norm2(src)

        if return_attn:
            return src, attn_weights
        return src
__all__ = ["CausalASPATransformerBlock"]
