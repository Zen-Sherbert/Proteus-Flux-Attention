"""Compact language-model head built on Proteus Attention blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from proteus_attention.modules import CausalGeneticTransformerBlock


@dataclass
class ModelConfig:
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    max_seq_len: int


class MiniProteusLM(nn.Module):
    """Lightweight GPT-style language model using Proteus Attention blocks."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        blocks = []
        for _ in range(cfg.num_layers):
            block = CausalGeneticTransformerBlock(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                attention_kwargs={
                    "max_seq_len": cfg.max_seq_len,
                    "return_stats_default": False,
                },
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, *, return_attn: bool = False):
        B, T = x.shape
        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence length exceeds configured maximum.")
        pos = torch.arange(T, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)[None, :, :]
        h = self.dropout(h)
        attn_outputs = []
        for block in self.blocks:
            if return_attn:
                h, attn = block(h, return_attn=True)
                attn_outputs.append(attn)
            else:
                h = block(h, return_attn=False)
        h = self.norm(h)
        logits = self.lm_head(h)
        if return_attn:
            return logits, attn_outputs
        return logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        seq = input_ids.to(device)
        for _ in range(max_new_tokens):
            seq_trim = seq[:, -self.cfg.max_seq_len :]
            logits = self(seq_trim)
            next_token = torch.distributions.Categorical(
                logits=(logits[:, -1, :] / max(1e-6, temperature))
            ).sample()
            seq = torch.cat([seq, next_token.unsqueeze(-1)], dim=1)
        return seq

    def set_flux_alpha(self, value: float) -> None:
        """Override the Flux alpha on every attention block."""
        for block in self.blocks:
            block.self_attn.attention.set_flux_alpha(value)
