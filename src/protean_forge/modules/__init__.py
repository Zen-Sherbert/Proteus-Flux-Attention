"""
High-level modules intended for drop-in integration with PyTorch models.
"""

from .genetic_multihead import CausalGeneticMultiheadAttention
from .sparse_ctl import SparseHeadController, SparseCtlSnapshot

__all__ = [
    "CausalGeneticMultiheadAttention",
    "SparseHeadController",
    "SparseCtlSnapshot",
]
