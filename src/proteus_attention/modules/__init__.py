"""
High-level modules intended for drop-in integration with PyTorch models.
"""

from .aspa_multihead import CausalASPAMultiheadAttention
from .sparse_ctl import SparseHeadController, SparseCtlSnapshot
from .transformer_block import CausalASPATransformerBlock

__all__ = [
    "CausalASPAMultiheadAttention",
    "SparseHeadController",
    "SparseCtlSnapshot",
    "CausalASPATransformerBlock",
]
