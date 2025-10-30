"""
Shared utilities for the Proteus Attention example scripts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

_DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "tiny_ctx_excerpt.txt"

try:
    import tiktoken  # type: ignore

    _ENCODER = tiktoken.get_encoding("gpt2")
except Exception:  # pragma: no cover - optional dependency
    _ENCODER = None


class ByteFallbackTokenizer:
    """Simple byte-level tokenizer used when `tiktoken` is unavailable."""

    n_vocab: int = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: Sequence[int]) -> str:
        return bytes(int(t) % 256 for t in tokens).decode("utf-8", errors="ignore")


class TikTokenWrapper:
    """Thin wrapper to unify the encode/decode API with the fallback tokenizer."""

    def __init__(self) -> None:
        self.n_vocab = int(getattr(_ENCODER, "n_vocab", 50257))

    def encode(self, text: str) -> List[int]:
        return _ENCODER.encode(text)  # type: ignore[operator]

    def decode(self, tokens: Sequence[int]) -> str:
        return _ENCODER.decode(tokens)  # type: ignore[operator]


def get_tokenizer() -> TikTokenWrapper | ByteFallbackTokenizer:
    if _ENCODER is None:
        return ByteFallbackTokenizer()
    return TikTokenWrapper()


def load_corpus(path: Optional[Path] = None) -> str:
    target = path or _DEFAULT_DATA
    if not target.is_file():
        return (
            "In the beginning there was Proteus Attention. "
            "It discovered new mixtures of heads and tokens, "
            "unlocking astonishing context windows on modest devices."
        )
    return target.read_text(encoding="utf-8")


def build_batches(
    tokens: Sequence[int],
    seq_len: int,
    batch_size: int,
    *,
    device: torch.device,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    total = len(tokens)
    if total <= seq_len:
        repeat = math.ceil((seq_len + 1) / total)
        tokens = list(tokens) * repeat
        total = len(tokens)

    import random

    while True:
        batch_inputs = torch.empty(batch_size, seq_len, dtype=torch.long)
        batch_targets = torch.empty(batch_size, seq_len, dtype=torch.long)
        for b in range(batch_size):
            start = random.randint(0, total - seq_len - 1)
            chunk = tokens[start : start + seq_len + 1]
            batch_inputs[b] = torch.tensor(chunk[:-1], dtype=torch.long)
            batch_targets[b] = torch.tensor(chunk[1:], dtype=torch.long)
        yield batch_inputs.to(device), batch_targets.to(device)


@dataclass
class TrainingConfig:
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    vocab_size: int = 50304  # GPT-2 tokenizer size
    max_seq_len: int = 4096
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_checkpoint_dir(name: str) -> Path:
    root = Path(__file__).resolve().parent / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    return target
