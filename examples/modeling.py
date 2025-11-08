"""Compact language-model head built on Proteus Attention blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from proteus_attention.modules import CausalASPATransformerBlock


class DenseTransformerBlock(nn.Module):
    """Mirror of nn.TransformerEncoderLayer using standard MultiheadAttention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.activation = nn.GELU()
        self.norm_first = bool(norm_first)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.norm_first:
            x = self.norm1(x)
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_out, _ = self.attn(x, x, x, need_weights=False, attn_mask=causal_mask)
        x = residual + self.dropout1(attn_out)
        if not self.norm_first:
            x = self.norm1(x)

        residual = x
        if self.norm_first:
            x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout2(x)
        if not self.norm_first:
            x = self.norm2(x)
        return x


@dataclass
class ModelConfig:
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    max_seq_len: int
    attn_proto_enable: bool = True
    attn_memory_enable: bool = True
    attn_memory_slots: int = 32
    attn_memory_decay: float = 0.9
    attn_router_top_p: float = 0.9


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
            attn_kwargs = {
                "max_seq_len": cfg.max_seq_len,
                "return_stats_default": False,
                "attn_proto_enable": cfg.attn_proto_enable,
                "attn_memory_enable": cfg.attn_memory_enable,
                "attn_memory_slots": cfg.attn_memory_slots,
                "attn_memory_decay": cfg.attn_memory_decay,
                "attn_router_top_p": cfg.attn_router_top_p,
            }
            block = CausalASPATransformerBlock(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                attention_kwargs=attn_kwargs,
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

    def last_head_consensus(self) -> Optional[torch.Tensor]:
        """Return the most recent head-level consensus from the top block, if available."""
        for block in reversed(self.blocks):
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            consensus = getattr(attn, "last_head_consensus", None)
            if consensus is not None:
                return consensus
        return None

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

    def set_shortlist_alpha(self, value: float) -> None:
        """Override the Shortlist alpha on every attention block."""
        for block in self.blocks:
            block.self_attn.attention.set_shortlist_alpha(value)


class MiniDenseLM(nn.Module):
    """Baseline dense Transformer language model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        blocks = []
        for _ in range(cfg.num_layers):
            blocks.append(
                DenseTransformerBlock(
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence length exceeds configured maximum.")
        pos = torch.arange(T, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)[None, :, :]
        h = self.dropout(h)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.lm_head(h)

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
