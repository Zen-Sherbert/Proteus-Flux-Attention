"""
Sparse attention control helpers shared across training loops and drop-in modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch.nn as nn


@dataclass
class SparseCtlSnapshot:
    """Single decision snapshot emitted after a controller adjustment."""

    density: float
    action: str
    min_heads: int
    max_heads: int
    step: int


class SparseHeadController:
    """
    Lightweight controller that nudges ``AdaptiveSparseProtoAttention`` head budgets toward
    a desired active density.  The implementation mirrors the example training loop
    (`examples/aspa_train.py`) but keeps the API minimal so modules can use it directly.

    Defaults mirror the CLI defaults used in ``aspa_train.py`` (target density 0.28 with a
    ±0.03 band evaluated every 4 observations).
    """

    def __init__(
        self,
        attention_layer: nn.Module,
        *,
        target_density: float = 0.28,
        tolerance: float = 0.03,
        cooldown: int = 4,
        min_heads: Optional[int] = None,
        max_heads: Optional[int] = None,
        dense_threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        self._layer = attention_layer
        self.target_density = float(max(0.0, min(target_density, 1.0)))
        self.tolerance = float(max(0.0, tolerance))
        self.cooldown = max(1, int(cooldown))
        self.verbose = verbose
        self._global_min = min_heads if (min_heads and min_heads > 0) else None
        self._global_max = max_heads if (max_heads and max_heads > 0) else None
        self._dense_threshold = dense_threshold

        self._step = 0
        self._since_adjust = 0
        self._history: list[float] = []
        self._last_snapshot: Optional[SparseCtlSnapshot] = None

        self._apply_bounds(self._global_min, self._global_max)
        if self._dense_threshold is not None:
            self._layer._dense_threshold = float(
                max(0.0, min(self._dense_threshold, 1.0)))

    @property
    def last_snapshot(self) -> Optional[SparseCtlSnapshot]:
        return self._last_snapshot

    def reset(self) -> None:
        self._step = 0
        self._since_adjust = 0
        self._history.clear()
        self._last_snapshot = None

    def observe(self, head_stats: Optional[Dict[str, Any]]) -> Optional[SparseCtlSnapshot]:
        """
        Consume the ``last_head_stats`` dictionary emitted by ``AdaptiveSparseProtoAttention``.
        Returns a snapshot when an adjustment occurs, otherwise ``None``.
        """
        self._step += 1
        if not isinstance(head_stats, dict):
            self._history.clear()
            return None

        density = head_stats.get("max_active_density")
        if density is None:
            self._history.clear()
            return None

        density_val = float(density)
        self._history.append(density_val)
        self._since_adjust += 1

        if self._since_adjust < self.cooldown:
            return None

        avg_density = sum(self._history) / max(1, len(self._history))
        upper = self.target_density + self.tolerance
        lower = max(0.0, self.target_density - self.tolerance)

        if avg_density > upper:
            action = "decrease"
            self._adjust_heads(-1)
        elif avg_density < lower:
            action = "increase"
            self._adjust_heads(+1)
        else:
            action = "steady"

        snapshot = SparseCtlSnapshot(
            density=avg_density,
            action=action,
            min_heads=int(getattr(self._layer, "h_active_min", 0)),
            max_heads=int(getattr(self._layer, "h_active_max", 0)),
            step=self._step,
        )
        self._last_snapshot = snapshot
        self._history.clear()
        self._since_adjust = 0
        if self.verbose:
            print(
                f"[SparseCtl] step={snapshot.step} density={snapshot.density:.3f} "
                f"action={snapshot.action} min={snapshot.min_heads} max={snapshot.max_heads}"
            )
        return snapshot

    # ------------------------------------------------------------------ Helpers

    def _apply_bounds(self, min_heads: Optional[int], max_heads: Optional[int]) -> None:
        layer = self._layer
        total = getattr(layer, "h_total", None)
        if total is None:
            return

        if min_heads is not None:
            layer.h_active_min = max(1, min(int(min_heads), int(total)))
        if max_heads is not None:
            max_val = max(layer.h_active_min, min(int(max_heads), int(total)))
            layer.h_active_max = max_val
        layer.h_active = max(layer.h_active_min, min(
            layer.h_active, layer.h_active_max))

    def _adjust_heads(self, delta: int) -> None:
        layer = self._layer
        current_max = int(getattr(layer, "h_active_max", 1))
        total = int(getattr(layer, "h_total", 1))
        proposed_max = max(1, min(current_max + delta, total))
        if self._global_max is not None:
            proposed_max = min(proposed_max, self._global_max)
        if self._global_min is not None:
            proposed_min = max(1, min(self._global_min, proposed_max))
        else:
            proposed_min = min(layer.h_active_min, proposed_max)

        changed = self._apply_bounds(proposed_min, proposed_max)
        if changed and hasattr(layer, "adjust_linear_L_scale"):
            if delta < 0:
                layer.adjust_linear_L_scale(
                    getattr(layer, "linear_latency_shrink", 0.8))
            elif delta > 0:
                layer.adjust_linear_L_scale(
                    getattr(layer, "linear_latency_growth", 1.05))
