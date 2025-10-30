"""
Implementation of Dynamic Mixture-of-Attention-Heads (DMoAH).

This module provides new `CausalDynamicAttention` and `AttentionBlock` classes
that can be used as a drop-in replacement for the standard transformer blocks
to enable dynamic, sparse attention head activation.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.sparse_attn import (
    dmoah_sparse_attention,
    get_last_backend,
    get_last_backend_info,
    build_flux_candidates,
    _record_backend,
    _pack_active_rows,
    TRITON_SEQ_LEN_LIMIT,
)


class ModelConfig:
    """
    Minimal, flexible configuration container used by the DMoAH modules.

    The class intentionally behaves like a simple attribute bag so that callers
    can freely set new attributes without needing an exhaustive schema.
    """

    _DEFAULTS: Dict[str, Any] = {
        "vocab_size": 0,
        "n_ctx": 0,
        "n_layer": 0,
        "n_head": 1,
        "d_model": 0,
        "p_dropout": 0.0,
        "bias": False,
        "attn_variant": "standard",
        "moe_n_experts": 0,
        # Linear/auto mode defaults
        "attn_mode": "auto",
        "attn_linear_L": 512,
        "attn_linear_window": 512,
        "attn_linear_anchor_stride": 256,
        "attn_linear_head_k": 2,
        "attn_linear_switch_ctx": 20000,
        "attn_linear_latency_budget_ms": None,
        "attn_linear_policy": "local+anchors",
    }

    def __init__(self, **kwargs: Any) -> None:
        data = dict(self._DEFAULTS)
        data.update(kwargs)
        self.__dict__.update(data)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class LayerNorm(nn.Module):
    """Standard LayerNorm with optional bias."""

    def __init__(self, ndim: int, bias: bool, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, eps=self.eps)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation."""

    def __init__(self, ndim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x_scaled = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_scaled


def _build_norm(config: ModelConfig) -> nn.Module:
    """Return the configured normalisation layer (LayerNorm or RMSNorm)."""

    norm_type = str(getattr(config, "norm_type", "layernorm")
                    or "layernorm").lower()
    eps = float(getattr(config, "norm_eps", 1e-5))
    if norm_type == "rmsnorm":
        return RMSNorm(config.d_model, eps=eps)
    return LayerNorm(config.d_model, bias=bool(getattr(config, "bias", False)), eps=eps)


class MLP(nn.Module):
    """Simple feed-forward block used by the AttentionBlock."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_mult = int(getattr(config, "mlp_hidden_mult", 4))
        hidden_dim = hidden_mult * config.d_model
        self.fc1 = nn.Linear(config.d_model, hidden_dim,
                             bias=bool(config.bias))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, config.d_model,
                             bias=bool(config.bias))
        self.dropout = nn.Dropout(config.p_dropout)
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)


class SwitchRouter(nn.Module):
    """
    Lightweight router that produces per-token gate probabilities over heads.

    The interface mirrors the subset used by CausalDynamicAttention: the module
    returns a tuple of (indices, scores, probabilities), where the first two
    entries are provided for API compatibility but are not consumed.
    """

    def __init__(self, d_model: int, n_gates: int, noise_std: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, n_gates, bias=False)
        self.noise_std = float(noise_std)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.linear(x)
        if self.training and self.noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probs = F.softmax(logits, dim=-1)
        scores, indices = torch.max(probs, dim=-1)
        return indices, scores, probs

# A new helper function for this module


def _get_causal_mask(target_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return an upper-triangular (causal) mask suitable for SDPA."""
    mask = torch.full((target_len, target_len), float(
        "-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask


class CausalDynamicAttention(nn.Module):
    """
    Dynamic Mixture-of-Attention-Heads (DMoAH).

    This module replaces the standard Multi-Head Attention. It uses a router to
    select a small subset of "active" heads for each token, enabling massive
    scalability in the total number of heads.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # --- Standard Attention Setup ---
        self.d_model = config.d_model
        self.bias = bool(getattr(config, "bias", False))

        # --- DMoAH configuration ---
        total_heads_hint = int(
            getattr(config, "attn_h_total", 0) or getattr(config, "n_head", 1) or 1)
        if total_heads_hint <= 0:
            total_heads_hint = max(1, int(getattr(config, "n_head", 1)))
        self.h_total = total_heads_hint
        if self.d_model % self.h_total != 0:
            raise ValueError(
                "d_model must be divisible by the total number of attention heads (attn_h_total)")

        active_heads_hint = int(getattr(config, "attn_h_active", 0))
        if active_heads_hint <= 0 or active_heads_hint > self.h_total:
            active_heads_hint = min(4, self.h_total)
            if active_heads_hint <= 0:
                active_heads_hint = 1
        base_active = active_heads_hint

        min_active_cfg = int(getattr(config, "attn_h_active_min", 0) or 0)
        max_active_cfg = int(getattr(config, "attn_h_active_max", 0) or 0)
        if max_active_cfg <= 0:
            max_active_cfg = min(16, self.h_total)
        max_active_cfg = max(1, min(max_active_cfg, self.h_total))
        if min_active_cfg <= 0:
            min_active_cfg = max(1, min(2, max_active_cfg))
        min_active_cfg = max(1, min(min_active_cfg, max_active_cfg))
        self.h_active_min = min_active_cfg
        self.h_active_max = max_active_cfg
        self.h_active = max(self.h_active_min, min(
            base_active, self.h_active_max))

        curve_hint = getattr(config, "attn_active_curve", None)
        try:
            curve_val = float(curve_hint) if curve_hint is not None else 0.25
        except (TypeError, ValueError):
            curve_val = 0.25
        if curve_val <= 0.0:
            curve_val = 1.0
        self.h_active_curve = curve_val
        self.force_dense_when_full = bool(
            getattr(config, "attn_force_dense_when_full", False))

        # The dimension of each individual head.
        self.d_head = self.d_model // self.h_total

        self.dropout = nn.Dropout(config.p_dropout)
        self.use_sdpa = bool(getattr(config, "use_sdpa", True) and hasattr(
            F, "scaled_dot_product_attention"))
        self.router_reg_lambda = float(
            getattr(config, "attn_router_lambda", 0.0) or 0.0)
        self.router_reg_beta = float(
            getattr(config, "attn_router_beta", 1.0) or 1.0)
        self.router_reg_mode = str(
            getattr(config, "attn_router_reg_mode", "entropy"))
        raw_mode = str(getattr(config, "attn_mode", "auto") or "auto").lower()
        if raw_mode not in {"auto", "subquad", "linear"}:
            raw_mode = "auto"
        self.mode_setting = raw_mode
        self.linear_L = max(
            1, int(getattr(config, "attn_linear_L", 512) or 512))
        schedule_raw = str(
            getattr(config, "attn_linear_L_schedule", "fixed") or "fixed").lower()
        if schedule_raw not in {"fixed", "sqrt", "log", "mem_cap"}:
            schedule_raw = "fixed"
        self.linear_L_schedule = schedule_raw
        base_L_cfg = int(getattr(config, "attn_linear_L_base",
                         self.linear_L) or self.linear_L)
        self.linear_L_base = max(1, base_L_cfg)
        self.linear_L_min = max(1, int(getattr(config, "attn_linear_L_min", min(
            64, self.linear_L_base)) or min(64, self.linear_L_base)))
        max_cfg = getattr(config, "attn_linear_L_max", self.linear_L)
        self.linear_L_max = max(self.linear_L_min, int(
            max_cfg if max_cfg is not None else self.linear_L))
        scale_cfg = float(getattr(config, "attn_linear_L_scale", 1.0) or 1.0)
        self.linear_L_scale_init = max(0.1, float(scale_cfg))
        self.linear_L_scale_max = float(getattr(config, "attn_linear_L_scale_max", max(
            1.0, self.linear_L_scale_init)) or max(1.0, self.linear_L_scale_init))
        self._linear_scale_min = max(
            0.05, self.linear_L_min / max(1.0, self.linear_L_base))
        self._linear_L_scale_runtime = self.linear_L_scale_init
        shrink_cfg = float(
            getattr(config, "attn_linear_latency_shrink", 0.8) or 0.8)
        self.linear_latency_shrink = min(max(shrink_cfg, 0.1), 0.99)
        growth_cfg = float(
            getattr(config, "attn_linear_latency_growth", 1.05) or 1.05)
        self.linear_latency_growth = max(1.0, growth_cfg)
        mem_budget = getattr(config, "attn_linear_mem_budget_mb", None)
        self.linear_mem_budget_mb = float(
            mem_budget) if mem_budget is not None else None
        piece_local = float(
            getattr(config, "attn_linear_piece_local_frac", 0.6) or 0.6)
        piece_anchor = float(
            getattr(config, "attn_linear_piece_anchor_frac", 0.3) or 0.3)
        piece_local = max(0.0, min(1.0, piece_local))
        piece_anchor = max(0.0, min(1.0, piece_anchor))
        piece_dna = float(getattr(config, "attn_linear_piece_dna_frac", max(
            0.0, 1.0 - piece_local - piece_anchor)))
        piece_dna = max(0.0, min(1.0, piece_dna))
        total_piece = piece_local + piece_anchor + piece_dna
        if total_piece <= 0.0:
            piece_local, piece_anchor, piece_dna = 1.0, 0.0, 0.0
            total_piece = 1.0
        piece_local /= total_piece
        piece_anchor /= total_piece
        piece_dna = max(0.0, 1.0 - piece_local - piece_anchor)
        self.linear_piece_local_frac = piece_local
        self.linear_piece_anchor_frac = piece_anchor
        self.linear_piece_dna_frac = piece_dna
        window_hint = int(getattr(config, "attn_linear_window",
                          self.linear_L_base) or self.linear_L_base)
        if window_hint <= 0:
            window_hint = self.linear_L_base
        self.linear_window_base = max(1, window_hint)
        self.linear_window = self.linear_window_base
        anchor_stride = int(getattr(config, "attn_linear_anchor_stride", getattr(
            config, "attn_linear_anchors_every", 256)) or 0)
        self.linear_anchor_stride = max(0, anchor_stride)
        head_k = int(getattr(config, "attn_linear_head_k", 2) or 2)
        self.linear_head_k = max(1, min(head_k, self.h_total))
        switch_ctx = int(
            getattr(config, "attn_linear_switch_ctx", 20000) or 20000)
        self.linear_switch_ctx = max(0, switch_ctx)
        latency_budget = getattr(config, "attn_linear_latency_budget_ms", None)
        self.linear_latency_budget_ms = float(
            latency_budget) if latency_budget is not None else None
        linear_policy_raw = str(getattr(
            config, "attn_linear_policy", "local+anchors") or "local+anchors").lower()
        tokens = {part.strip() for token in linear_policy_raw.replace(
            "|", ",").replace("+", ",").split(",") for part in (token,) if token.strip()}
        if not tokens:
            tokens = {"local", "anchors"}
        self.linear_policy_tokens = tokens
        self.linear_use_anchors = any(name in tokens for name in {
                                      "anchors", "anchor", "global"})
        self.linear_use_dna = any(
            name in tokens for name in {"dna", "semantic"})
        self.linear_use_local = any(name in tokens for name in {"local", "window"}) or not (
            self.linear_use_anchors or self.linear_use_dna)
        if not self.linear_use_local:
            self.linear_piece_local_frac = 0.0
        if not self.linear_use_anchors:
            self.linear_piece_anchor_frac = 0.0
        if not self.linear_use_dna:
            self.linear_piece_dna_frac = 0.0
        piece_sum = self.linear_piece_local_frac + \
            self.linear_piece_anchor_frac + self.linear_piece_dna_frac
        if piece_sum <= 0.0:
            self.linear_piece_local_frac = 1.0
            self.linear_piece_anchor_frac = 0.0
            self.linear_piece_dna_frac = 0.0
        else:
            self.linear_piece_local_frac /= piece_sum
            self.linear_piece_anchor_frac /= piece_sum
            self.linear_piece_dna_frac = max(
                0.0, 1.0 - self.linear_piece_local_frac - self.linear_piece_anchor_frac)
        self._linear_runtime: Optional[dict] = None
        self._linear_runtime_seq_len: Optional[int] = None
        self._current_mode: str = "subquad"
        self.quantize_sparse_int8 = bool(
            getattr(config, "attn_quantize_int8", False))
        quant_mode = str(
            getattr(config, "attn_quantize_int8_mode", "max") or "max").lower()
        if quant_mode not in {"max", "percentile", "ema_percentile"}:
            quant_mode = "max"
        self.quantize_sparse_int8_mode = quant_mode
        percentile = float(
            getattr(config, "attn_quantize_int8_percentile", 1.0) or 1.0)
        percentile = min(max(percentile, 0.0), 1.0)
        self.quantize_sparse_int8_percentile = percentile
        ema_decay = float(
            getattr(config, "attn_quantize_int8_ema_decay", 0.9) or 0.9)
        ema_decay = min(max(ema_decay, 0.0), 0.9999)
        self.quantize_sparse_int8_ema_decay = ema_decay
        self.register_buffer("_quant_ema_q", torch.zeros(
            self.h_total, dtype=torch.float32), persistent=False)
        self.register_buffer("_quant_ema_k", torch.zeros(
            self.h_total, dtype=torch.float32), persistent=False)
        self.register_buffer("_quant_ema_v", torch.zeros(
            self.h_total, dtype=torch.float32), persistent=False)
        self._quant_ema_flags = {
            "_quant_ema_q": False,
            "_quant_ema_k": False,
            "_quant_ema_v": False,
        }
        seq_low_cfg = int(getattr(config, "attn_active_seq_low", 0) or 0)
        seq_high_cfg = int(getattr(config, "attn_active_seq_high", 0) or 0)
        if seq_low_cfg <= 0:
            seq_low_cfg = max(64, self.d_head)
        if seq_high_cfg <= 0:
            seq_high_cfg = int(getattr(config, "n_ctx", 0)
                               or (seq_low_cfg * 4))
        if seq_high_cfg < seq_low_cfg:
            seq_high_cfg = seq_low_cfg
        self.seq_low = seq_low_cfg
        self.seq_high = seq_high_cfg
        dense_cutoff_hint = getattr(config, "attn_small_seq_dense", None)
        try:
            dense_cutoff_val = int(
                dense_cutoff_hint) if dense_cutoff_hint is not None else max(0, self.seq_low // 2)
        except (TypeError, ValueError):
            dense_cutoff_val = max(0, self.seq_low // 2)
        if dense_cutoff_val < 0:
            dense_cutoff_val = 0
        self.small_seq_dense = dense_cutoff_val
        self._density_ema: Optional[float] = None
        self._last_density: Optional[float] = None
        self._last_target_k: Optional[int] = None
        dense_threshold_hint = getattr(
            config, "attn_force_dense_threshold", None)
        if dense_threshold_hint is not None:
            dense_threshold = float(dense_threshold_hint)
            if dense_threshold <= 0.0:
                dense_threshold = None
        else:
            dense_threshold = None
        self._dense_threshold: Optional[float] = dense_threshold

        gates_hint = int(getattr(config, "attn_gates", 0))
        if gates_hint <= 0:
            gates_hint = max(self.h_total, self.h_active_max,
                             self.h_active) * 2
        self.attn_gates = max(gates_hint, self.h_total, self.h_active_max)
        router_noise = float(getattr(config, "attn_router_noise_std", 0.01))
        self.head_router = SwitchRouter(
            self.d_model, self.attn_gates, noise_std=router_noise)

        gate_to_head = self._default_mapping(self.attn_gates, self.h_total)
        self.register_buffer("gate_to_head", gate_to_head, persistent=False)
        one_hot = F.one_hot(
            gate_to_head, num_classes=self.h_total).to(torch.float32)
        self.register_buffer("gate_to_head_onehot", one_hot, persistent=False)

        self._flux_alpha_override: Optional[float] = None
        # --- Standard layers ---
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=self.bias)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=self.bias)

        # --- Metrics & caches ---
        self.last_head_stats: Optional[dict] = None
        self._mask_cache: Dict[Tuple[str, int, torch.dtype], torch.Tensor] = {}
        self._last_sparse_state: Optional[dict] = None
        self._cached_active_mask: Optional[torch.Tensor] = None
        self._cached_packing: Optional[Tuple[torch.Tensor,
                                             torch.Tensor, torch.Tensor, int]] = None
        self.last_aux_loss: Optional[torch.Tensor] = None
        self._latency_ema: Optional[float] = None
        self.dna_enabled = bool(getattr(config, "attn_dna_enable", False))
        self.dna_decay = float(getattr(config, "attn_dna_decay", 0.97))
        self.dna_threshold = float(getattr(config, "attn_dna_threshold", 0.25))
        self.dna_blend = float(getattr(config, "attn_dna_blend", 0.6))
        self.dna_temp = float(max(getattr(config, "attn_dna_temp", 0.2), 1e-3))
        self.dna_usage_boost = float(
            max(getattr(config, "attn_dna_usage_boost", 0.0), 0.0))
        self.dna_init_scale = float(
            getattr(config, "attn_dna_init_scale", 0.02))
        if self.dna_enabled:
            proto = torch.randn(self.attn_gates, self.d_model,
                                dtype=torch.float32) * self.dna_init_scale
            usage = torch.zeros(self.attn_gates, dtype=torch.float32)
            confidence = torch.zeros(self.attn_gates, dtype=torch.float32)
            self.register_buffer("dna_proto", proto)
            self.register_buffer("dna_usage_ema", usage)
            self.register_buffer("dna_confidence", confidence)
        else:
            self.dna_proto = None  # type: ignore[attr-defined, assignment]
            self.dna_usage_ema = None  # type: ignore[attr-defined, assignment]
            # type: ignore[attr-defined, assignment]
            self.dna_confidence = None
        self._last_dna_stats: Optional[dict] = None
        self.token_sparse = bool(getattr(config, "attn_token_sparse", False))
        keep_ratio = float(
            getattr(config, "attn_token_keep_ratio", 1.0) or 1.0)
        keep_ratio = min(max(keep_ratio, 0.0), 1.0)
        self.token_keep_ratio_base = keep_ratio
        self._token_keep_ratio_runtime: Optional[float] = None
        token_keep_sched = str(
            getattr(config, "attn_linear_token_keep_schedule", "sqrt") or "sqrt").lower()
        if token_keep_sched not in {"fixed", "sqrt", "log"}:
            token_keep_sched = "sqrt"
        self.linear_token_keep_schedule = token_keep_sched
        self.linear_token_keep_ratio_min = float(
            getattr(config, "attn_linear_token_keep_min_ratio", 0.2) or 0.2)
        self.linear_token_keep_ratio_min = min(
            max(self.linear_token_keep_ratio_min, 0.0), self.token_keep_ratio_base)
        self.linear_token_keep_ratio_ref = int(
            getattr(config, "attn_linear_token_keep_ref", 2048) or 2048)
        min_keep_cfg = int(getattr(config, "attn_token_keep_min", 0) or 0)
        if min_keep_cfg < 0:
            min_keep_cfg = 0
        guard_cfg = int(getattr(config, "attn_token_keep_guard", 1) or 1)
        if guard_cfg < 0:
            guard_cfg = 0
        self.token_keep_guard = guard_cfg
        self.token_keep_min = max(
            min_keep_cfg, guard_cfg if self.token_sparse else min_keep_cfg)
        threshold_cfg = float(
            getattr(config, "attn_token_keep_threshold", 0.0) or 0.0)
        self.token_keep_threshold = float(max(0.0, threshold_cfg))
        self._last_token_importance: Optional[torch.Tensor] = None
        self._last_token_mask: Optional[torch.Tensor] = None
        self._last_token_blend: Optional[torch.Tensor] = None
        self._last_token_similarity: Optional[torch.Tensor] = None
        self._last_flux_backend_info: Optional[Dict[str, object]] = None
        self._last_batch_size: Optional[int] = None

    @staticmethod
    def _default_mapping(gates: int, heads: int) -> torch.Tensor:
        """Creates a default mapping from gates to heads."""
        if heads <= 0:
            heads = 1
        idx = torch.arange(gates, dtype=torch.long)
        return torch.clamp((idx * heads) // max(gates, 1), 0, heads - 1)

    def _route_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Blend DNA similarity routing with the learned router."""
        _, _, router_probs = self.head_router(x)
        gate_probs = router_probs.to(torch.float32)
        dna_stats: Optional[dict] = None
        dna_similarity: Optional[torch.Tensor] = None
        dna_blend_scalar: Optional[torch.Tensor] = None
        if self.dna_enabled and self.dna_proto is not None:
            proto = self.dna_proto
            token_norm = F.normalize(x.detach().to(
                torch.float32), dim=-1, eps=1e-6)
            proto_norm = F.normalize(proto, dim=-1, eps=1e-6)
            sims = torch.matmul(token_norm, proto_norm.transpose(0, 1))
            dna_similarity, _ = sims.max(dim=-1)
            dna_probs = F.softmax(sims / self.dna_temp, dim=-1)
            blend_scalar = torch.clamp(
                (dna_similarity - self.dna_threshold) /
                max(1e-6, 1.0 - self.dna_threshold),
                0.0,
                1.0,
            )
            if self.dna_confidence is not None:
                confidence = (self.dna_confidence.view(
                    1, 1, -1) + 1e-6).to(torch.float32)
                dna_probs = dna_probs * confidence
                dna_probs = dna_probs / \
                    dna_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            dna_probs = dna_probs.to(gate_probs.dtype)
            blend = self.dna_blend * \
                blend_scalar.unsqueeze(-1).to(gate_probs.dtype)
            gate_probs = gate_probs * (1.0 - blend) + dna_probs * blend
            usage_bias = None
            if self.dna_usage_boost > 0.0 and self.dna_usage_ema is not None:
                usage = self.dna_usage_ema + 1e-6
                usage_bias = torch.softmax(-usage, dim=0).to(gate_probs.dtype)
                gate_probs = gate_probs + usage_bias.view(1, 1, -1) * (
                    (1.0 - blend_scalar).unsqueeze(-1).to(gate_probs.dtype) *
                    self.dna_usage_boost
                )
            gate_probs = gate_probs.clamp_min(0.0)
            gate_probs = gate_probs / \
                gate_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            dna_stats = {
                "max_sim_mean": float(dna_similarity.mean().item()),
                "blend_mean": float(blend_scalar.mean().item()),
                "active_tokens": int((blend_scalar > 0).sum().item()),
            }
            if usage_bias is not None:
                entropy = - \
                    (usage_bias * usage_bias.clamp_min(1e-9).log()).sum()
                dna_stats["usage_bias_entropy"] = float(entropy.item())
            dna_blend_scalar = blend_scalar.detach()
        self._last_dna_stats = dna_stats
        if dna_similarity is not None:
            self._last_token_similarity = dna_similarity.detach().to(torch.float32)
        else:
            self._last_token_similarity = None
        if dna_blend_scalar is not None:
            self._last_token_blend = dna_blend_scalar.to(torch.float32)
        else:
            self._last_token_blend = None
        return gate_probs

    def _update_dna(self, token_feats: torch.Tensor, gate_probs: torch.Tensor) -> None:
        if not (self.dna_enabled and self.dna_proto is not None and self.dna_usage_ema is not None and self.dna_confidence is not None):
            return
        with torch.no_grad():
            feats = token_feats.detach().to(torch.float32)
            probs = gate_probs.detach().to(torch.float32)
            flat_feats = feats.view(-1, self.d_model)
            flat_probs = probs.view(-1, self.attn_gates)
            counts = flat_probs.sum(dim=0)
            mask = counts > 1e-6
            if not mask.any():
                return
            weighted = flat_probs.transpose(0, 1) @ flat_feats
            mean = torch.zeros_like(self.dna_proto)
            mean[mask] = weighted[mask] / counts[mask].unsqueeze(-1)
            decay = self.dna_decay
            proto = self.dna_proto
            proto[mask] = proto[mask] * decay + mean[mask] * (1.0 - decay)
            usage = self.dna_usage_ema
            usage.mul_(decay)
            usage[mask] += counts[mask] * (1.0 - decay)
            confidence = self.dna_confidence
            confidence.mul_(decay)
            confidence[mask] += (1.0 - decay)
            if self._last_dna_stats is None:
                self._last_dna_stats = {}
            self._last_dna_stats["updated_gates"] = int(mask.sum().item())

    def _causal_mask(self, target_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return (and cache) a causal mask of size (target_len, target_len)."""
        key = (device.type, int(device.index)
               if device.index is not None else -1, dtype)
        cached = self._mask_cache.get(key)
        if cached is None or cached.size(0) < target_len or cached.device != device or cached.dtype != dtype:
            mask = _get_causal_mask(target_len, device, dtype)
            self._mask_cache[key] = mask
            return mask
        return cached[:target_len, :target_len]

    def _choose_active_k(self, seq_len: int) -> int:
        if seq_len <= self.seq_low:
            target = self.h_active_max
        elif seq_len >= self.seq_high:
            target = self.h_active_min
        else:
            span = max(1, self.seq_high - self.seq_low)
            ratio = float(seq_len - self.seq_low) / float(span)
            curve = self.h_active_curve
            if curve != 1.0:
                ratio = max(0.0, min(1.0, ratio)) ** curve
            interp = self.h_active_max - ratio * \
                (self.h_active_max - self.h_active_min)
            target = int(math.floor(interp + 1e-6))
        target = max(self.h_active_min, min(target, self.h_active_max))
        self._last_target_k = target
        return target

    def _update_density(self, value: float) -> None:
        value = float(max(0.0, value))
        self._last_density = value
        if self._density_ema is None:
            self._density_ema = value
        else:
            self._density_ema = 0.8 * self._density_ema + 0.2 * value

    def _select_active_tokens(self, importance: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.token_sparse:
            return None
        if importance is None:
            return None
        if importance.dim() != 2:
            return None
        B, T = importance.shape
        if T == 0:
            return None
        ratio = float(
            self._token_keep_ratio_runtime if self._token_keep_ratio_runtime is not None else self.token_keep_ratio_base)
        if ratio >= 0.999 and self.token_keep_threshold <= 0.0:
            mask = torch.ones(B, T, dtype=torch.bool, device=importance.device)
            guard = min(self.token_keep_guard, T)
            if guard > 0:
                mask[:, :guard] = True
            self._last_token_mask = mask
            return mask

        guard = min(max(0, self.token_keep_guard), T)
        min_keep = min(max(0, self.token_keep_min), T)
        min_keep = max(min_keep, guard)
        keep_counts = torch.full(
            (B,), ratio * T, device=importance.device, dtype=torch.float32)
        keep_counts = keep_counts.round().to(torch.int64)
        if min_keep > 0:
            keep_counts = torch.clamp(keep_counts, min=min_keep)
        keep_counts = torch.clamp(keep_counts, min=1, max=T)

        threshold = float(self.token_keep_threshold)
        importance_fp32 = importance.to(torch.float32)
        mask = torch.zeros(B, T, dtype=torch.bool, device=importance.device)
        for b in range(B):
            base = torch.zeros(T, dtype=torch.bool, device=importance.device)
            if threshold > 0.0:
                base = importance_fp32[b] >= threshold
            need = int(keep_counts[b].item())
            need = max(need, int(base.sum().item()))
            need = max(need, guard)
            need = min(max(1, need), T)
            if need >= T:
                base = torch.ones(T, dtype=torch.bool,
                                  device=importance.device)
            else:
                topk_idx = torch.topk(
                    importance_fp32[b], k=need, largest=True).indices
                base[topk_idx] = True
            if guard > 0:
                base[:guard] = True
            mask[b] = base
        self._last_token_mask = mask
        return mask

    def _density_threshold(self) -> Optional[float]:
        return self._dense_threshold

    def _set_linear_scale(self, value: float) -> None:
        min_scale = self._linear_scale_min
        max_scale = max(min_scale, self.linear_L_scale_max)
        self._linear_L_scale_runtime = float(
            min(max(value, min_scale), max_scale))
        self._linear_runtime_seq_len = None

    def adjust_linear_L_scale(self, factor: float) -> None:
        """Adjust the linear shortlist scale multiplicatively (used by controllers)."""
        if factor == 0.0:
            return
        new_scale = self._linear_L_scale_runtime * factor
        self._set_linear_scale(new_scale)

    def _linear_token_keep_ratio(self, seq_len: int) -> float:
        if not self.token_sparse:
            return self.token_keep_ratio_base
        schedule = self.linear_token_keep_schedule
        ref = max(1, self.linear_token_keep_ratio_ref)
        base = self.token_keep_ratio_base
        seq = max(1, int(seq_len))
        if schedule == "sqrt":
            ratio = base * math.sqrt(ref / float(seq))
        elif schedule == "log":
            ratio = base * (math.log2(ref + 1.0) / math.log2(seq + 1.0))
        else:
            ratio = base
        min_ratio = min(base, self.linear_token_keep_ratio_min)
        ratio = max(min_ratio, min(base, ratio))
        return float(ratio)

    def _get_flux_alpha(self, seq_len: int) -> float:
        if self._flux_alpha_override is not None:
            return float(max(0.0, min(1.0, self._flux_alpha_override)))
        seq_len = int(max(1, seq_len))
        low = max(1, int(self.seq_low))
        high = max(low + 1, int(self.linear_switch_ctx)
                   ) if self.linear_switch_ctx > 0 else max(low + 1, seq_len)
        if seq_len <= low:
            return 0.0
        if seq_len >= high:
            return 1.0
        return float((seq_len - low) / (high - low))

    def set_flux_alpha(self, value: float) -> None:
        """Externally override the flux slider (0=dense, 1=linear)."""
        self._flux_alpha_override = float(max(0.0, min(1.0, value)))
        self._linear_runtime = None
        self._linear_runtime_seq_len = None

    def _compute_effective_linear_params(self, seq_len: int, batch_size: Optional[int], alpha: float) -> dict:
        seq_len = max(1, int(seq_len))
        scale = self._linear_L_scale_runtime
        schedule = self.linear_L_schedule
        coeff = float(max(1.0, self.linear_L_base))
        if schedule == "sqrt":
            base_val = coeff * math.sqrt(float(seq_len))
        elif schedule == "log":
            base_val = coeff * math.log2(float(seq_len) + 1.0)
        elif schedule == "mem_cap" and self.linear_mem_budget_mb:
            heads_est = max(
                1, int(getattr(self, "_last_target_k", self.linear_head_k)))
            batch_est = max(1, int(batch_size)
                            if batch_size is not None else 1)
            denom = heads_est * batch_est * seq_len * max(1, self.d_head) * 8
            if denom > 0:
                cap_from_mem = int(
                    self.linear_mem_budget_mb * 1024 * 1024 // denom)
            else:
                cap_from_mem = self.linear_L_max
            base_val = max(1, min(cap_from_mem, self.linear_L_max))
        else:
            base_val = coeff
        eff_raw = int(math.ceil(scale * max(1.0, base_val)))
        eff = max(self.linear_L_min, min(self.linear_L_max, eff_raw))
        window = min(self.linear_window_base, eff)

        dense_cap = int(
            min(seq_len, max(self.linear_L_min, self.linear_window_base + seq_len // 32)))
        dense_cap = max(self.linear_L_min, min(self.linear_L_max, dense_cap))
        eff = int(alpha * eff + (1.0 - alpha) * dense_cap)
        eff = max(self.linear_L_min, min(self.linear_L_max, eff))

        window_dense = max(
            1, min(seq_len, max(self.linear_window_base, seq_len // 8)))
        window = int(alpha * window + (1.0 - alpha) * window_dense)
        window = max(1, min(eff, window))

        dense_weights = (
            0.85 if self.linear_use_local else 0.0,
            0.10 if self.linear_use_anchors else 0.0,
            0.05 if self.linear_use_dna else 0.0,
        )
        target_weights = (
            self.linear_piece_local_frac if self.linear_use_local else 0.0,
            self.linear_piece_anchor_frac if self.linear_use_anchors else 0.0,
            self.linear_piece_dna_frac if self.linear_use_dna else 0.0,
        )
        weight_local = (1.0 - alpha) * \
            dense_weights[0] + alpha * target_weights[0]
        weight_anchor = (1.0 - alpha) * \
            dense_weights[1] + alpha * target_weights[1]
        weight_dna = (1.0 - alpha) * \
            dense_weights[2] + alpha * target_weights[2]
        total_weight = weight_local + weight_anchor + weight_dna
        if total_weight <= 0.0:
            weight_local = 1.0
            weight_anchor = 0.0
            weight_dna = 0.0
            total_weight = 1.0
        weight_local /= total_weight
        weight_anchor /= total_weight
        weight_dna = max(0.0, 1.0 - weight_local - weight_anchor)

        local_cap = int(round(eff * self.linear_piece_local_frac)
                        ) if self.linear_use_local else 0
        anchor_cap = int(round(eff * self.linear_piece_anchor_frac)
                         ) if self.linear_use_anchors else 0
        dna_cap = int(round(eff * self.linear_piece_dna_frac)
                      ) if self.linear_use_dna else 0
        local_cap = int(round(eff * weight_local)
                        ) if self.linear_use_local else 0
        anchor_cap = int(round(eff * weight_anchor)
                         ) if self.linear_use_anchors else 0
        dna_cap = int(round(eff * weight_dna)) if self.linear_use_dna else 0
        caps = [local_cap, anchor_cap, dna_cap]
        total_caps = sum(caps)
        if total_caps > eff and total_caps > 0:
            scale_down = eff / total_caps
            caps = [int(max(0, math.floor(c * scale_down))) for c in caps]
            total_caps = sum(caps)
        remaining = eff - total_caps
        order = [0, 1, 2]
        for idx in order:
            if remaining <= 0:
                break
            if (idx == 0 and not self.linear_use_local) or (idx == 1 and not self.linear_use_anchors) or (idx == 2 and not self.linear_use_dna):
                continue
            caps[idx] += 1
            remaining -= 1
        if remaining > 0:
            caps[0 if self.linear_use_local else 1 if self.linear_use_anchors else 2] += remaining
        if self.linear_use_local:
            caps[0] = min(caps[0], window)
            if caps[0] <= 0 and eff > 0:
                caps[0] = min(window, max(1, eff // 4))
        if self.linear_use_anchors and caps[1] < 0:
            caps[1] = 0
        if self.linear_use_dna and caps[2] < 0:
            caps[2] = 0
        local_cap, anchor_cap, dna_cap = caps
        return {
            "effective_L": eff,
            "local_cap": max(0, int(local_cap)),
            "anchor_cap": max(0, int(anchor_cap)),
            "dna_cap": max(0, int(dna_cap)),
            "window": max(1, int(window)),
            "flux_alpha": float(alpha),
        }

    def _prepare_linear_runtime(self, seq_len: int, batch_size: Optional[int]) -> dict:
        if self._linear_runtime is not None and self._linear_runtime_seq_len == seq_len:
            return self._linear_runtime
        alpha = self._get_flux_alpha(seq_len)
        params = self._compute_effective_linear_params(
            seq_len, batch_size, alpha)
        eff = params["effective_L"]
        self.linear_L = eff
        self.linear_window = params["window"]
        runtime = {
            "effective_L": eff,
            "local_cap": params["local_cap"],
            "anchor_cap": params["anchor_cap"],
            "dna_cap": params["dna_cap"],
            "window": params["window"],
            "anchor_stride": self.linear_anchor_stride,
            "flux_alpha": params["flux_alpha"],
        }
        self._linear_runtime = runtime
        self._linear_runtime_seq_len = seq_len
        return runtime

    def _select_mode(self, seq_len: int) -> str:
        """
        Decide whether to run the sub-quadratic sparse path or the constant-L linear path.

        * ``subquad`` keeps the existing token/head sparsity.
        * ``linear`` runs the shortlist-based constant-L attention.
        * ``auto`` switches based on configured sequence/latency thresholds.
        """

        if self.mode_setting == "linear":
            return "linear"
        if self.mode_setting == "subquad":
            return "subquad"
        if self.linear_latency_budget_ms is not None and self._latency_ema is not None:
            budget = self.linear_latency_budget_ms
            latency = self._latency_ema
            if latency > budget * 1.05:
                if self._linear_L_scale_runtime > self._linear_scale_min + 1e-3:
                    self.adjust_linear_L_scale(self.linear_latency_shrink)
                    self._linear_runtime_seq_len = None
                return "linear"
            if latency < budget * 0.80 and seq_len < self.linear_switch_ctx:
                if self._linear_L_scale_runtime < self.linear_L_scale_max:
                    self.adjust_linear_L_scale(self.linear_latency_growth)
                return "subquad"
        if self.linear_switch_ctx > 0:
            if seq_len >= self.linear_switch_ctx:
                return "linear"
            if self._current_mode == "linear" and seq_len >= max(1, int(self.linear_switch_ctx * 0.75)):
                return "linear"
        return "subquad"

    def _run_flux_attention(
        self,
        q_heads: torch.Tensor,
        k_heads: torch.Tensor,
        v_heads: torch.Tensor,
        active_mask: torch.Tensor,
        packed: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
        seq_len: int,
        dropout_p: float,
        head_indices: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Flux attention per head using fused shortlist candidates."""

        head_idx, token_idx, row_offsets, _ = packed
        active_heads = q_heads.size(0)
        total_rows = int(token_idx.numel())
        if total_rows == 0:
            return q_heads.new_zeros(q_heads.shape)

        runtime = self._prepare_linear_runtime(
            seq_len, getattr(self, "_last_batch_size", None))
        eff_L = runtime["effective_L"]

        alpha = runtime.get("flux_alpha", self._get_flux_alpha(seq_len))
        flux_candidates = token_idx.unsqueeze(
            1).expand(-1, eff_L).to(torch.long).clone()
        flux_lengths = torch.ones(
            total_rows, device=token_idx.device, dtype=torch.long)

        for head in range(active_heads):
            start = int(row_offsets[head].item())
            end = int(row_offsets[head + 1].item())
            if start == end:
                continue
            rows = token_idx[start:end]
            global_head = int(head_indices[head].item())
            batch_index = global_head // self.h_total
            dna_scores = None
            if self.linear_use_dna and self._last_token_similarity is not None:
                dna_scores = self._last_token_similarity[batch_index]
            candidates, lengths = build_flux_candidates(
                rows,
                seq_len,
                max_candidates=eff_L,
                linear_window=runtime["window"],
                anchor_stride=runtime["anchor_stride"],
                use_local=self.linear_use_local,
                use_anchors=self.linear_use_anchors,
                dna_scores=dna_scores,
                local_cap=runtime["local_cap"],
                anchor_cap=runtime["anchor_cap"],
                dna_cap=runtime["dna_cap"],
            )
            flux_candidates[start:end] = candidates.to(torch.long)
            flux_lengths[start:end] = lengths.to(torch.long)

        self._last_flux_backend_info = None
        attn_sparse = dmoah_sparse_attention(
            q_heads,
            k_heads,
            v_heads,
            active_mask=active_mask.unsqueeze(-1).contiguous(),
            causal_mask=causal_mask,
            dropout_p=dropout_p,
            training=self.training,
            prepacked=packed,
            flux_candidates=flux_candidates,
            flux_lengths=flux_lengths,
        )
        self._last_flux_backend_info = get_last_backend_info()
        if self._last_sparse_state is not None:
            self._last_sparse_state["flux_alpha"] = float(alpha)
        if self.last_head_stats is not None:
            self.last_head_stats["flux_alpha"] = float(alpha)
        return attn_sparse

    def _quantize_per_head_int8(self, tensor: torch.Tensor, ema_buffer: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a (H, T, D) tensor to int8 with per-head scales.

        Returns the quantized tensor and a float32 scale per head so that
        dequantized values can be recovered via ``quant * scale``.
        """
        if tensor.dim() != 3:
            raise ValueError(
                "Expected a rank-3 tensor for per-head quantization")
        tensor_fp32 = tensor.to(torch.float32)
        flat = tensor_fp32.view(tensor_fp32.size(0), -1)
        abs_flat = flat.abs()
        mode = self.quantize_sparse_int8_mode

        if mode == "max":
            base_vals = abs_flat.amax(dim=1, keepdim=True)
        else:
            percentile = self.quantize_sparse_int8_percentile
            if percentile <= 0.0:
                base_vals = abs_flat.amax(dim=1, keepdim=True)
            else:
                # torch.quantile expects floating tensors
                base_vals = torch.quantile(
                    abs_flat, percentile, dim=1, keepdim=True)
                max_vals = abs_flat.amax(dim=1, keepdim=True)
                base_vals = torch.where(base_vals > 0.0, base_vals, max_vals)
            if mode == "ema_percentile":
                decay = self.quantize_sparse_int8_ema_decay
                ema_tensor = getattr(self, ema_buffer)
                flags = getattr(self, "_quant_ema_flags", None)
                initialized = False
                if flags is not None:
                    initialized = bool(flags.get(ema_buffer, False))
                if ema_tensor is None or ema_tensor.numel() != flat.size(0) or not initialized:
                    ema_tensor = abs_flat.amax(dim=1)
                else:
                    ema_tensor = ema_tensor.to(
                        tensor.device, dtype=torch.float32)
                updated = decay * ema_tensor + \
                    (1.0 - decay) * base_vals.squeeze(1)
                setattr(self, ema_buffer, updated.detach())
                if flags is not None:
                    flags[ema_buffer] = True
                base_vals = updated.unsqueeze(1)

        scales = torch.clamp(base_vals / 127.0, min=1e-6)
        quant = torch.clamp((flat / scales).round(), -
                            127.0, 127.0).to(torch.int8)
        quant = quant.view_as(tensor)
        scales = scales.squeeze(1).to(tensor.device, dtype=torch.float32)
        return quant.contiguous(), scales.contiguous()

    def _forward_dense_fastpath(self, x: torch.Tensor) -> torch.Tensor:
        """Dense SDPA path for very short sequences."""
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)

        q = q.view(B, T, self.h_total, self.d_head).permute(
            0, 2, 1, 3).reshape(B * self.h_total, T, self.d_head)
        k = k.view(B, T, self.h_total, self.d_head).permute(
            0, 2, 1, 3).reshape(B * self.h_total, T, self.d_head)
        v = v.view(B, T, self.h_total, self.d_head).permute(
            0, 2, 1, 3).reshape(B * self.h_total, T, self.d_head)

        dropout_p = self.dropout.p if self.training else 0.0
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        attn = attn.view(B, self.h_total, T, self.d_head).permute(0, 2, 1, 3)
        y = attn.reshape(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)

        self._last_target_k = self.h_total
        self._last_sparse_state = {
            "mode": "dense_fastpath",
            "backend": "sdpa_fastpath",
            "max_rows": T,
            "density": 1.0,
            "target_k": self.h_total,
            "force_dense_threshold": None if self._dense_threshold is None else float(self._dense_threshold),
            "cutoff": int(self.small_seq_dense),
            "quantized": False,
            "token_keep_fraction": 1.0,
        }
        self._update_density(1.0)
        self.last_head_stats = {
            "top_k": int(self.h_total),
            "mean_gate_prob": float(1.0),
            "router_entropy": float(0.0),
            "mean_active_per_token": float(self.h_total),
            "unique_heads": int(self.h_total),
            "active_fraction": float(1.0),
            "max_active_rows": int(T),
            "max_active_density": float(1.0),
            "target_k": int(self.h_total),
            "seq_len": int(T),
            "active_min": int(self.h_active_min),
            "active_max": int(self.h_active_max),
            "force_dense_threshold": None if self._dense_threshold is None else float(self._dense_threshold),
            "active_curve": float(self.h_active_curve),
            "token_keep_fraction": float(1.0),
        }
        self.last_router_reg = None
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device
        self.last_aux_loss = None
        self._last_token_importance = None
        self._last_token_mask = None
        self._last_token_blend = None
        if not self.dna_enabled:
            self._last_token_similarity = None

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()

        def _finalize_latency(mode_label: str) -> float:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            if self._latency_ema is None:
                self._latency_ema = latency_ms
            else:
                self._latency_ema = 0.8 * self._latency_ema + 0.2 * latency_ms
            if self.last_head_stats is not None:
                self.last_head_stats["latency_ms"] = float(latency_ms)
                self.last_head_stats["mode_setting"] = self.mode_setting
                self.last_head_stats["mode_selected"] = mode_label
            if self._last_sparse_state is not None:
                self._last_sparse_state["latency_ms"] = float(latency_ms)
                self._last_sparse_state["mode_selected"] = mode_label
            return latency_ms

        self._last_batch_size = B

        if self.small_seq_dense and T <= self.small_seq_dense:
            y = self._forward_dense_fastpath(x)
            _finalize_latency("dense_fastpath")
            return y

        mode_now = self._select_mode(T)
        self._current_mode = mode_now
        if mode_now == "linear":
            self._token_keep_ratio_runtime = self._linear_token_keep_ratio(T)
        else:
            self._token_keep_ratio_runtime = None

        # --- 1. Head Routing (The NEW Step) ---
        # The head router predicts which *gates* are most relevant for each token.
        # Note: We are not using the expert_idx from the router directly. We use its probabilities.
        gate_probs = self._route_tokens(x)
        gate_max = gate_probs.detach().amax(dim=-1)
        importance = gate_max
        if self._last_token_similarity is not None:
            sim = self._last_token_similarity.to(gate_max.dtype)
            if self._last_token_blend is not None:
                blend = self._last_token_blend.to(gate_max.dtype)
                importance = gate_max * (1.0 - blend) + sim * blend
            else:
                importance = torch.maximum(gate_max, sim)
        self._last_token_importance = importance.detach().to(torch.float32)
        token_mask = self._select_active_tokens(self._last_token_importance)
        if token_mask is not None and token_mask.shape != (B, T):
            token_mask = None
        token_keep_fraction = float(
            token_mask.float().mean().item()) if token_mask is not None else 1.0
        self._last_token_mask = token_mask

        # Aggregate gate probabilities to head probabilities via the mapping
        # This is the same efficient matrix multiplication trick used in DynamicMoE
        map_mat = self.gate_to_head_onehot.to(
            dtype=gate_probs.dtype, device=gate_probs.device)
        head_probs = torch.matmul(gate_probs, map_mat)  # (B, T, h_total)
        router_reg = None
        reg_terms: list[torch.Tensor] = []
        lb_requested = False
        load_balance_term: Optional[torch.Tensor] = None
        if self.router_reg_lambda > 0.0:
            raw_mode = str(self.router_reg_mode or "").lower()
            raw_mode = raw_mode.replace("|", ",").replace("+", ",")
            mode_parts = {part.strip()
                          for part in raw_mode.split(",") if part.strip()}
            if not mode_parts:
                mode_parts = {"entropy"}

            def _mode_has(name: str) -> bool:
                aliases = {
                    "entropy": {"entropy", "ent"},
                    "l1": {"l1", "l_1"},
                    "load_balance": {"load_balance", "loadbalance", "lb", "balance"},
                }
                pool = aliases.get(name, {name})
                return any(token in pool for token in mode_parts)

            if _mode_has("entropy"):
                logp = gate_probs.clamp_min(1e-9).log()
                entropy_term = (logp * gate_probs).sum(dim=-1).mean()
                reg_terms.append(entropy_term)
            if _mode_has("l1"):
                per_token_mass = gate_probs.sum(dim=-1)
                l1_term = (per_token_mass ** self.router_reg_beta).mean()
                reg_terms.append(l1_term)
            lb_requested = _mode_has("load_balance")

        # Select the Top-K active heads for each token (adaptive top-k)
        topk = self._choose_active_k(T)
        if mode_now == "linear":
            topk = max(1, min(topk, self.linear_head_k))
            self._last_target_k = topk
        if topk <= 0:
            raise RuntimeError(
                "DMoAH requires at least one active head per token")
        head_scores, top_head_indices = torch.topk(
            head_probs, topk, dim=-1)  # (B, T, h_active)

        # Build a mask that marks active heads per token
        head_mask_bool = torch.zeros(
            B, T, self.h_total, device=device, dtype=torch.bool)
        head_mask_bool.scatter_(2, top_head_indices, True)
        if token_mask is not None:
            head_mask_bool &= token_mask.unsqueeze(-1)

        head_mask_per_head = head_mask_bool.permute(0, 2, 1)  # (B, h_total, T)
        active_mask_flat = head_mask_per_head.reshape(B * self.h_total, T)
        active_head_mask = active_mask_flat.any(dim=1)
        if lb_requested:
            head_mass = head_probs.float().mean(dim=(0, 1))
            assign_fraction = head_mask_bool.float().mean(dim=(0, 1))
            load_balance_term = self.h_total * \
                torch.sum(head_mass * assign_fraction)
            reg_terms.append(load_balance_term)
            self.last_aux_loss = load_balance_term
        if reg_terms:
            combined_reg = reg_terms[0]
            for term in reg_terms[1:]:
                combined_reg = combined_reg + term
            router_reg = self.router_reg_lambda * combined_reg
        total_active_heads = int(active_head_mask.sum().item())
        active_counts = head_mask_per_head.sum(dim=2).float()
        max_rows_est = int(active_counts.max().item()
                           ) if active_counts.numel() > 0 else 0
        density_est = float(max_rows_est) / float(max(1, T))
        self._update_density(density_est)
        density_threshold = self._density_threshold()
        all_rows_active = bool(active_mask_flat.all(
        ).item()) if active_mask_flat.numel() else False
        force_dense = False
        if density_threshold is not None and density_est >= density_threshold:
            force_dense = True
        elif all_rows_active and self.force_dense_when_full:
            force_dense = True
        if mode_now == "linear":
            force_dense = False

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)

        q = q.view(B, T, self.h_total, self.d_head)
        k = k.view(B, T, self.h_total, self.d_head)
        v = v.view(B, T, self.h_total, self.d_head)

        q_flat = q.permute(0, 2, 1, 3).reshape(
            B * self.h_total, T, self.d_head)
        k_flat = k.permute(0, 2, 1, 3).reshape(
            B * self.h_total, T, self.d_head)
        v_flat = v.permute(0, 2, 1, 3).reshape(
            B * self.h_total, T, self.d_head)

        limit = TRITON_SEQ_LEN_LIMIT
        long_context = limit > 0 and T >= limit
        causal_mask: Optional[torch.Tensor] = None
        dropout_p = self.dropout.p if self.training else 0.0

        exec_mode_label = "sparse"

        if force_dense:
            self._cached_active_mask = None
            self._cached_packing = None
            if causal_mask is None:
                causal_mask = self._causal_mask(T, device, q_flat.dtype)
            attn_flat = dmoah_sparse_attention(
                q_flat,
                k_flat,
                v_flat,
                active_mask=None,
                causal_mask=causal_mask,
                dropout_p=dropout_p,
                training=self.training,
            )

            attn_out = attn_flat.view(
                B, self.h_total, T, self.d_head).permute(0, 2, 1, 3)
            mode_str = "dense" if total_active_heads == B * self.h_total else "dense_auto"
            exec_mode_label = mode_str
            self._last_sparse_state = {
                "mode": mode_str,
                "backend": get_last_backend(),
                "max_rows": max_rows_est,
                "density": density_est,
                "target_k": int(topk),
                "force_dense_threshold": None if density_threshold is None else float(density_threshold),
                "quantized": False,
                "token_keep_fraction": float(token_keep_fraction),
                "mode_setting": self.mode_setting,
                "mode_selected": mode_now,
            }
        else:
            active_head_indices = active_head_mask.nonzero(
                as_tuple=False).squeeze(1)
            if active_head_indices.numel() == 0:
                attn_out = x.new_zeros(B, T, self.h_total, self.d_head)
                self._cached_active_mask = None
                self._cached_packing = None
                self._last_sparse_state = {
                    "mode": "dense",
                    "backend": "not_invoked",
                    "max_rows": max_rows_est,
                    "density": density_est,
                    "target_k": int(topk),
                    "force_dense_threshold": None if density_threshold is None else float(density_threshold),
                    "quantized": False,
                    "token_keep_fraction": float(token_keep_fraction),
                    "mode_setting": self.mode_setting,
                    "mode_selected": mode_now,
                }
            else:
                active_mask_comp = active_mask_flat.index_select(
                    0, active_head_indices)
                active_mask_comp = active_mask_comp.to(
                    dtype=torch.bool, device=device).contiguous()

                cache_hit = (
                    (not self.training)
                    and self._cached_active_mask is not None
                    and self._cached_packing is not None
                    and self._cached_active_mask.device == active_mask_comp.device
                    and self._cached_active_mask.shape == active_mask_comp.shape
                    and torch.equal(self._cached_active_mask, active_mask_comp)
                )
                if cache_hit:
                    packed = self._cached_packing
                else:
                    packed = _pack_active_rows(active_mask_comp)
                    if not self.training:
                        self._cached_active_mask = active_mask_comp.clone()
                        if packed is None:
                            self._cached_packing = None
                        else:
                            self._cached_packing = tuple(
                                t.clone() if isinstance(t, torch.Tensor) else t for t in packed
                            )
                    else:
                        self._cached_active_mask = None
                        self._cached_packing = None

                q_comp = q_flat.index_select(0, active_head_indices)
                k_comp = k_flat.index_select(0, active_head_indices)
                v_comp = v_flat.index_select(0, active_head_indices)

                if packed is None:
                    attn_sparse = q_comp.new_zeros(q_comp.shape)
                    attn_out_flat = q_flat.new_zeros(
                        B * self.h_total, T, self.d_head)
                    attn_out_flat.index_copy_(
                        0, active_head_indices, attn_sparse)
                    attn_out = attn_out_flat.view(
                        B, self.h_total, T, self.d_head).permute(0, 2, 1, 3)
                    self._last_sparse_state = {
                        "mode": "sparse",
                        "backend": get_last_backend(),
                        "active_head_indices": active_head_indices,
                        "max_rows": 0,
                        "density": density_est,
                        "target_k": int(topk),
                        "force_dense_threshold": None if density_threshold is None else float(density_threshold),
                        "cache_hit": bool(cache_hit),
                        "quantized": False,
                        "token_keep_fraction": float(token_keep_fraction),
                        "mode_setting": self.mode_setting,
                        "mode_selected": mode_now,
                    }
                    exec_mode_label = "sparse"
                else:
                    head_idx, token_idx, row_offsets, max_rows = packed
                    backend_name: str
                    mode_label = "sparse"
                    use_quant = False
                    if mode_now == "linear":
                        attn_sparse = self._run_flux_attention(
                            q_comp,
                            k_comp,
                            v_comp,
                            active_mask_comp,
                            packed,
                            T,
                            dropout_p,
                            active_head_indices,
                            causal_mask,
                        )
                        backend_info = self._last_flux_backend_info or get_last_backend_info()
                        backend_name = str(backend_info.get("name", "flux"))
                        mode_label = "flux"
                        exec_mode_label = "flux"
                    else:
                        use_quant = bool(
                            self.quantize_sparse_int8 and not self.training and not long_context
                        )
                        if not long_context and causal_mask is None:
                            causal_mask = self._causal_mask(
                                T, device, q_flat.dtype)
                        if use_quant:
                            q_quant, q_scales = self._quantize_per_head_int8(
                                q_comp, "_quant_ema_q")
                            k_quant, k_scales = self._quantize_per_head_int8(
                                k_comp, "_quant_ema_k")
                            v_quant, v_scales = self._quantize_per_head_int8(
                                v_comp, "_quant_ema_v")
                            attn_sparse = dmoah_sparse_attention(
                                q_quant,
                                k_quant,
                                v_quant,
                                active_mask=active_mask_comp.unsqueeze(
                                    -1).contiguous(),
                                causal_mask=causal_mask,
                                dropout_p=dropout_p,
                                training=self.training,
                                prepacked=packed,
                                q_scale=q_scales,
                                k_scale=k_scales,
                                v_scale=v_scales,
                                out_dtype=q_comp.dtype,
                            )
                        else:
                            attn_sparse = dmoah_sparse_attention(
                                q_comp,
                                k_comp,
                                v_comp,
                                active_mask=active_mask_comp.unsqueeze(
                                    -1).contiguous(),
                                causal_mask=causal_mask,
                                dropout_p=dropout_p,
                                training=self.training,
                                prepacked=packed,
                            )
                        backend_name = get_last_backend()

                    attn_out_flat = q_flat.new_zeros(
                        B * self.h_total, T, self.d_head)
                    attn_out_flat.index_copy_(
                        0, active_head_indices, attn_sparse)
                    attn_out = attn_out_flat.view(
                        B, self.h_total, T, self.d_head).permute(0, 2, 1, 3)
                    sparse_state = {
                        "mode": mode_label,
                        "backend": backend_name,
                        "active_head_indices": active_head_indices,
                        "head_idx": head_idx,
                        "token_idx": token_idx,
                        "row_offsets": row_offsets,
                        "max_rows": int(max_rows),
                        "density": density_est,
                        "target_k": int(topk),
                        "force_dense_threshold": None if density_threshold is None else float(density_threshold),
                        "cache_hit": bool(cache_hit),
                        "quantized": bool(use_quant),
                        "token_keep_fraction": float(token_keep_fraction),
                        "mode_setting": self.mode_setting,
                        "mode_selected": mode_now,
                    }
                    if mode_now == "linear":
                        sparse_state["linear_L"] = int(self.linear_L)
                        sparse_state["linear_window"] = int(self.linear_window)
                        sparse_state["linear_anchor_stride"] = int(
                            self.linear_anchor_stride)
                        backend_info = self._last_flux_backend_info or get_last_backend_info()
                        backend_name = backend_info.get("name") if isinstance(
                            backend_info, dict) else None
                        if backend_name is not None:
                            sparse_state["linear_shortlist_backend"] = backend_name
                            details = backend_info.get("details") if isinstance(
                                backend_info, dict) else None
                            if isinstance(details, dict):
                                for key, value in details.items():
                                    sparse_state[f"linear_shortlist_{key}"] = value
                    self._last_sparse_state = sparse_state
                    if mode_now != "linear":
                        exec_mode_label = mode_label

        # Merge head dimension back to (B, T, C)
        if token_mask is not None:
            attn_out = attn_out * token_mask.unsqueeze(-1).unsqueeze(-1)
        y = attn_out.reshape(B, T, C)

        # --- 4. Final Projection ---
        # The output `y` is now a sparse combination of head outputs. We project it back.
        y = self.proj(y)
        y = self.dropout(y)

        # --- 5. Telemetry ---
        with torch.no_grad():
            mask_bt_h = head_mask_bool
            head_scores_fp32 = torch.zeros_like(mask_bt_h, dtype=torch.float32)
            head_scores_fp32.scatter_(2, top_head_indices, head_scores.float())
            gate_probs_fp32 = gate_probs.float()
            entropy = -(gate_probs_fp32.clamp_min(1e-9).log()
                        * gate_probs_fp32).sum(dim=-1).mean()
            active_fraction = mask_bt_h.float().mean()
            active_per_token = mask_bt_h.sum(dim=2).float().mean()
            unique_heads = mask_bt_h.any(dim=0).any(dim=0).sum()
            denom = mask_bt_h.float().sum().clamp_min(1.0)
            mean_gate_prob = head_scores_fp32.sum() / denom
        self.last_head_stats = {
            "top_k": int(topk),
            "mean_gate_prob": float(mean_gate_prob.item()),
            "router_entropy": float(entropy.item()),
            "mean_active_per_token": float(active_per_token.item()),
            "unique_heads": int(unique_heads.item()),
            "active_fraction": float(active_fraction.item()),
            "max_active_rows": int(max_rows_est),
            "max_active_density": float(density_est),
            "target_k": int(self._last_target_k or topk),
            "seq_len": int(T),
            "active_min": int(self.h_active_min),
            "active_max": int(self.h_active_max),
            "force_dense_threshold": None if density_threshold is None else float(density_threshold),
            "active_curve": float(self.h_active_curve),
            "token_keep_fraction": float(token_keep_fraction),
            "mode": exec_mode_label,
        }
        if mode_now == "linear":
            self.last_head_stats["linear_L"] = int(self.linear_L)
            self.last_head_stats["linear_window"] = int(self.linear_window)
            self.last_head_stats["linear_anchor_stride"] = int(
                self.linear_anchor_stride)
            backend_info = self._last_flux_backend_info or get_last_backend_info()
            backend_name = backend_info.get("name") if isinstance(
                backend_info, dict) else None
            if backend_name is not None:
                self.last_head_stats["linear_shortlist_backend"] = backend_name
                details = backend_info.get("details") if isinstance(
                    backend_info, dict) else None
                if isinstance(details, dict):
                    for key, value in details.items():
                        self.last_head_stats[f"linear_shortlist_{key}"] = value
        if self._last_dna_stats:
            self.last_head_stats["dna"] = dict(self._last_dna_stats)
        self.last_router_reg = router_reg
        self._update_dna(x, gate_probs)

        _finalize_latency(exec_mode_label)
        if self.linear_mem_budget_mb and device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
            except RuntimeError:
                allocated = None
            if allocated is not None:
                high_cap = self.linear_mem_budget_mb * 1.1
                low_cap = self.linear_mem_budget_mb * 0.6
                if allocated > high_cap:
                    self.adjust_linear_L_scale(self.linear_latency_shrink)
                elif allocated < low_cap:
                    self.adjust_linear_L_scale(self.linear_latency_growth)
        return y


class CausalGeneticAttention(CausalDynamicAttention):
    """CausalDynamicAttention with DNA-based priors enabled by default."""

    def __init__(self, config: ModelConfig):
        if not getattr(config, "attn_dna_enable", False):
            try:
                setattr(config, "attn_dna_enable", True)
            except Exception:
                pass
        super().__init__(config)


class AttentionBlock(nn.Module):
    """
    The new Transformer block that uses CausalDynamicAttention.
    This is a drop-in replacement for the original `Block`.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = _build_norm(config)

        # --- USE THE NEW ATTENTION MODULE ---
        if bool(getattr(config, "attn_dna_enable", False)):
            self.attn = CausalGeneticAttention(config)
        else:
            self.attn = CausalDynamicAttention(config)

        self.ln2 = _build_norm(config)

        # The minimal standalone build always uses a dense MLP head.
        self.mlp = MLP(config)
        self._is_moe = False

        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_moe_stats: Optional[dict] = None
        self.last_router_reg: Optional[torch.Tensor] = None
        self.last_head_stats: Optional[dict] = None
        self.last_sparse_state: Optional[dict] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out

        mlp_input = self.ln2(x)
        mlp_out = self.mlp(mlp_input)
        x = x + mlp_out
        self.last_aux_loss = getattr(self.mlp, 'last_aux_loss', None)

        attn_reg = getattr(self.attn, 'last_router_reg', None)
        self.last_router_reg = attn_reg

        self.last_moe_stats = None
        self.last_head_stats = getattr(self.attn, 'last_head_stats', None)
        self.last_sparse_state = getattr(self.attn, '_last_sparse_state', None)
        return x

# NOTE: DMoAH blocks remain experimental; monitor training stability when enabling.
