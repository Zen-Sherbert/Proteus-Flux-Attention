"""Hugging Face integration scaffolding for Proteus Attention.

This module intentionally keeps the "patching" step minimal so that downstream
projects can supply architecture-specific conversions.  By default the adapter
loads the checkpoint unchanged and exposes a thin wrapper that provides the
hooks used by the staged training harness (freeze policies, parameter grouping,
alpha updates).

To integrate with a particular family (e.g. Gemma, LLaMA, etc.) replace the
``ModelAdapter`` methods with logic that swaps the native attention blocks for
``proteus_attention.modules.CausalGeneticMultiheadAttention`` and configures
prototype/controller parameters appropriately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AdapterConfig:
    heads: int
    gate_ratio: float
    strategy: str
    target_ctx: Optional[int]
    context_mastery: Optional[dict]


class ModelAdapter:
    """Adapter exposing hooks expected by the staged training harness.

    The default implementation simply delegates to the underlying Hugging Face
    model without modifying its attention layers.  Override the methods below to
    perform real conversions; they are deliberately small and self-contained.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    # ------------------------------------------------------------------ Patching
    def apply_proteus_attention(self, config: AdapterConfig) -> None:
        """Hook for swapping native attention with Proteus modules.

        Override in downstream integrations.  The base implementation logs a
        gentle warning so users remember to wire in their architecture-specific
        patch before launching training.
        """

        print(
            "[adapter] WARNING: Proteus attention has not been injected. "
            "Replace ModelAdapter.apply_proteus_attention with an architecture-"
            "specific implementation to swap attention blocks."
        )

    # ------------------------------------------------------------ Stage Control
    def freeze_for_stage(self, stage_name: str) -> None:
        """Adjust parameter ``requires_grad`` flags for the stage.

        Default behaviour keeps everything trainable. Override this to freeze
        components (e.g. embeddings) during the warm-up stage.
        """

        # No-op by default.

    def parameter_groups(self, base_lr: float, proto_lr_scale: float) -> List[dict]:
        """Return parameter groups for the optimiser.

        The default returns a single group with the base learning rate. Override
        this to assign higher learning rates to newly introduced Proteus
        parameters (routers/prototypes).
        """

        return [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": base_lr}]

    def set_shortlist_alpha(self, value: float) -> None:
        """Best-effort broadcast of shortlist alpha to attention modules."""

        for module in self.model.modules():
            if hasattr(module, "set_shortlist_alpha"):
                try:
                    module.set_shortlist_alpha(value)
                except Exception:
                    continue


def load_and_prepare_model(
    workspace_path: str | bytes | "os.PathLike[str]",
    adapter_cfg: AdapterConfig,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[torch.nn.Module, ModelAdapter, AutoTokenizer]:
    """Load a HF model/tokenizer and wrap it in a ``ModelAdapter``."""

    tokenizer = AutoTokenizer.from_pretrained(workspace_path)
    model = AutoModelForCausalLM.from_pretrained(workspace_path, torch_dtype=torch_dtype)
    model.to(device)
    adapter = ModelAdapter(model)
    adapter.apply_proteus_attention(adapter_cfg)
    return model, adapter, tokenizer
