"""Genetic Attention drop-in wrapper that mirrors ``nn.MultiheadAttention`` while
delegating to the underlying DMoAH kernels."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..kernels.sparse_attn import get_last_backend_info
from ..models.dmoah import CausalGeneticAttention, ModelConfig
from .sparse_ctl import SparseHeadController, SparseCtlSnapshot


class CausalGeneticMultiheadAttention(nn.Module):
    """
    Thin adapter that exposes Genetic Attention via :class:`CausalGeneticAttention`
    using the familiar ``nn.MultiheadAttention`` API (restricted to causal
    self-attention for now).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_seq_len: int | None = None,
        return_stats_default: bool = False,
        sparse_ctl_config: Optional[Dict[str, Any]] = None,
        **config_overrides: Any,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.batch_first = bool(batch_first)
        self._return_stats_default = bool(return_stats_default)
        self._last_runtime_stats: Optional[Dict[str, Any]] = None
        self._sparse_ctl: Optional[SparseHeadController] = None

        cfg = {
            "d_model": self.embed_dim,
            "attn_h_total": self.num_heads,
            "n_head": self.num_heads,
            "attn_h_active": config_overrides.pop("attn_h_active", min(self.num_heads, 8)),
            "attn_h_active_min": config_overrides.pop("attn_h_active_min", 2),
            "attn_h_active_max": config_overrides.pop("attn_h_active_max", self.num_heads),
            "p_dropout": float(dropout),
            "bias": bool(bias),
            "attn_variant": "dmoah",
            "attn_mode": config_overrides.pop("attn_mode", "auto"),
            "attn_dna_enable": bool(config_overrides.pop("attn_dna_enable", True)),
            "n_ctx": int(max_seq_len or config_overrides.pop("n_ctx", 0)),
            "attn_linear_L": config_overrides.pop("attn_linear_L", 512),
            "attn_linear_window": config_overrides.pop("attn_linear_window", 512),
            "attn_linear_anchor_stride": config_overrides.pop("attn_linear_anchor_stride", 256),
            "attn_linear_head_k": config_overrides.pop("attn_linear_head_k", 2),
            "attn_small_seq_dense": config_overrides.pop("attn_small_seq_dense", 0),
            "attn_force_dense_threshold": config_overrides.pop("attn_force_dense_threshold", 0.6),
            "attn_quantize_int8": config_overrides.pop("attn_quantize_int8", False),
            "attn_quantize_int8_mode": config_overrides.pop("attn_quantize_int8_mode", "max"),
            "attn_quantize_int8_percentile": config_overrides.pop("attn_quantize_int8_percentile", 1.0),
            "attn_quantize_int8_ema_decay": config_overrides.pop("attn_quantize_int8_ema_decay", 0.9),
        }
        cfg.update(config_overrides)

        config = ModelConfig(**cfg)
        self.attention = CausalGeneticAttention(config)

        if device is not None or dtype is not None:
            self.attention.to(device=device, dtype=dtype)

        if sparse_ctl_config:
            self._sparse_ctl = SparseHeadController(self.attention, **sparse_ctl_config)

    @staticmethod
    def _ensure_fastpath(x: torch.Tensor) -> torch.Tensor:
        """Return a contiguous view for optimal kernel performance."""
        return x if x.is_contiguous() else x.contiguous()

    @staticmethod
    def _validate_attn_mask(
        attn_mask: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
    ) -> None:
        """
        Ensure the provided ``attn_mask`` is compatible with the current
        implementation.  We accept ``None`` or the standard causal mask; custom
        masks will raise ``NotImplementedError`` so callers know the semantics
        cannot be preserved yet.
        """
        if attn_mask is None:
            return

        mask = attn_mask
        if mask.dim() == 3 and mask.size(0) == 1:
            mask = mask.squeeze(0)
        if mask.dim() != 2 or mask.size(0) != seq_len or mask.size(1) != seq_len:
            raise NotImplementedError(
                "CausalGeneticMultiheadAttention currently supports attn_mask of shape (target_len, source_len)."
            )

        if mask.dtype == torch.bool:
            mask_bool = mask.to(torch.bool, copy=False)
        else:
            mask_bool = torch.isinf(mask).to(torch.bool)

        expected = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        if torch.equal(mask_bool.to(device), expected):
            return

        raise NotImplementedError(
            "Only standard causal masks are supported right now. Custom attn_mask patterns are not yet implemented."
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: Optional[bool] = None,
        *,
        return_stats: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if key is not None and key is not query:
            raise NotImplementedError("Only self-attention is supported (key == query).")
        if value is not None and value is not query:
            raise NotImplementedError("Only self-attention is supported (value == query).")
        if is_causal is not None and not is_causal:
            raise NotImplementedError("Non-causal attention is not supported by CausalGeneticMultiheadAttention.")

        x = query if self.batch_first else query.transpose(0, 1)
        x = self._ensure_fastpath(x)
        batch, seq_len, _ = x.shape
        device = x.device

        self._validate_attn_mask(attn_mask, seq_len, device)

        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2 or key_padding_mask.size(1) != seq_len:
                raise ValueError("key_padding_mask must have shape (batch, seq_len).")
            padding = key_padding_mask.to(device=device).unsqueeze(-1)
            x = x.masked_fill(padding, 0.0)

        attn_output = self.attention(x)

        if key_padding_mask is not None:
            attn_output = attn_output.masked_fill(padding, 0.0)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            if average_attn_weights:
                attn_weights = attn_output.new_zeros(batch, seq_len, seq_len)
            else:
                attn_weights = attn_output.new_zeros(batch * self.num_heads, seq_len, seq_len)

        want_stats = self._return_stats_default if return_stats is None else bool(return_stats)
        head_stats = getattr(self.attention, "last_head_stats", None)
        ctl_snapshot: Optional[SparseCtlSnapshot] = None
        if self._sparse_ctl is not None:
            ctl_snapshot = self._sparse_ctl.observe(head_stats)
        if want_stats:
            stats: Dict[str, Any] = {
                "head_stats": head_stats,
                "sparse_state": getattr(self.attention, "_last_sparse_state", None),
                "shortlist_backend": get_last_backend_info(),
            }
            if ctl_snapshot is not None:
                stats["sparse_ctl"] = {
                    "density": ctl_snapshot.density,
                    "action": ctl_snapshot.action,
                    "min_heads": ctl_snapshot.min_heads,
                    "max_heads": ctl_snapshot.max_heads,
                    "step": ctl_snapshot.step,
                }
            elif self._sparse_ctl is not None:
                stats["sparse_ctl"] = None
            self._last_runtime_stats = stats
        else:
            self._last_runtime_stats = None

        return attn_output, attn_weights

    @property
    def last_runtime_stats(self) -> Optional[Dict[str, Any]]:
        """Return the most recent runtime statistics, if collected."""
        return self._last_runtime_stats

    @property
    def sparse_controller(self) -> Optional[SparseHeadController]:
        """Expose the optional sparse head controller."""
        return self._sparse_ctl
