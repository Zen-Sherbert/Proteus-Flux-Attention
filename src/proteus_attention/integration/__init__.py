"""Integration helpers for adapting external checkpoints to Proteus Attention."""

from .hf_adapter import load_and_prepare_model, ModelAdapter

__all__ = ["load_and_prepare_model", "ModelAdapter"]
