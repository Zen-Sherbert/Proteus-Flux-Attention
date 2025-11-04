import math
from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import proteus_attention.kernels.sparse_attn as sparse_attn
from proteus_attention.kernels.sparse_attn import (
    build_flux_candidates,
    build_packed_flux_candidates,
    dmoah_sparse_attention,
)
from proteus_attention.models.dmoah import (
    AttentionBlock,
    CausalDynamicAttention,
    ModelConfig,
    MLP,
    _build_norm,
)

_prepare_training_configs = None  # CLI helpers intentionally omitted in the minimal build.


def _causal_mask(length: int, device, dtype) -> torch.Tensor:
    mask = torch.full((length, length), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


def _reference_shortlist(
    rows: torch.Tensor,
    seq_len: int,
    linear_L: int,
    linear_window: int,
    anchor_stride: int,
    use_local: bool,
    use_anchors: bool,
    dna_scores: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rows.numel() == 0:
        empty = rows.new_empty(rows.shape + (0,))
        lengths = rows.new_zeros(rows.shape, dtype=torch.long)
        return empty, lengths
    device = rows.device
    pieces: list[torch.Tensor] = []

    if use_local:
        window = min(linear_window, linear_L)
        steps = torch.arange(window, device=device, dtype=torch.long)
        lengths_local = torch.clamp(rows + 1, max=window)
        start = torch.clamp(rows - lengths_local + 1, min=0)
        local = start.unsqueeze(1) + steps.unsqueeze(0)
        valid = steps.unsqueeze(0) < lengths_local.unsqueeze(1)
        local = torch.where(valid, local, rows.unsqueeze(1))
        pieces.append(local)

    if use_anchors and anchor_stride > 0:
        slots = min(linear_L, max(1, seq_len))
        offsets = torch.arange(1, slots + 1, device=device, dtype=torch.long) * anchor_stride
        anchors = torch.clamp(rows.unsqueeze(1) - offsets.unsqueeze(0), min=0)
        pieces.append(anchors)

    if pieces:
        candidates = torch.cat(pieces, dim=1)
    else:
        candidates = rows.unsqueeze(1)

    if dna_scores is not None and dna_scores.numel() > 0:
        sims = dna_scores.to(device=device)
        dna = torch.empty(rows.size(0), linear_L, device=device, dtype=torch.long)
        for i, row_val in enumerate(rows.tolist()):
            upto = int(row_val) + 1
            if upto <= 0:
                dna[i].fill_(0)
                continue
            dna_k = min(linear_L, upto)
            tmp = torch.topk(sims[:upto], k=dna_k).indices.to(device=device, dtype=torch.long)
            if tmp.numel() < linear_L:
                pad = rows.new_full((linear_L - tmp.numel(),), int(row_val))
                tmp = torch.cat([pad, tmp], dim=0)
            dna[i] = tmp[-linear_L:]
        candidates = torch.cat([candidates, dna], dim=1)

    candidates = torch.minimum(candidates, rows.unsqueeze(1))
    candidates = torch.clamp(candidates, min=0, max=seq_len - 1)
    candidates = torch.cat([candidates, rows.unsqueeze(1)], dim=1)
    candidates, _ = torch.sort(candidates, dim=1)
    keep = torch.ones_like(candidates, dtype=torch.bool)
    if candidates.size(1) > 1:
        keep[:, 1:] = candidates[:, 1:] != candidates[:, :-1]
    unique_index = keep.to(torch.long).cumsum(dim=1) - 1
    keep = keep & (unique_index < linear_L)
    lengths = keep.sum(dim=1, dtype=torch.long)
    lengths = torch.clamp(lengths, min=1)
    output = rows.unsqueeze(1).expand(-1, linear_L).to(torch.long).clone()
    if keep.any():
        r_idx, c_idx = keep.nonzero(as_tuple=True)
        dst = unique_index[r_idx, c_idx]
        output[r_idx, dst] = candidates[r_idx, c_idx]
    return output.contiguous(), lengths.contiguous()


class StandardAttention(nn.Module):
    """Dense Multi-Head Attention used for the baseline GPT."""

    def __init__(self, d_model: int, n_head: int, p_dropout: float, bias: bool) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head for StandardAttention")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = nn.Dropout(p_dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        mask = _causal_mask(T, x.device, x.dtype).unsqueeze(0)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(attn))


class StandardAttentionBlock(nn.Module):
    """Transformer block that mirrors AttentionBlock but with dense attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = _build_norm(config)
        self.attn = StandardAttention(
            d_model=config.d_model,
            n_head=int(getattr(config, "n_head", 1) or 1),
            p_dropout=float(getattr(config, "p_dropout", 0.0) or 0.0),
            bias=bool(getattr(config, "bias", False)),
        )
        self.ln2 = _build_norm(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Small GPT-style model used for behavioural regression tests."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.n_ctx, config.d_model)
        attn_variant = str(getattr(config, "attn_variant", "standard")).lower()
        self.use_dmoah = attn_variant == "dmoah"

        blocks = []
        for _ in range(config.n_layer):
            if self.use_dmoah:
                blocks.append(AttentionBlock(config))
            else:
                blocks.append(StandardAttentionBlock(config))
        self.blocks = nn.ModuleList(blocks)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.p_dropout)
        self.last_attn_stats = None

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(tok + pos)

        attn_stats: list[dict] = []
        for block in self.blocks:
            x = block(x)
            if self.use_dmoah:
                stats = getattr(block, "last_head_stats", None)
                if stats is not None:
                    attn_stats.append(stats)

        self.last_attn_stats = attn_stats if attn_stats else None

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

def test_dmoah_sparse_attention_matches_manual_reference():
    torch.manual_seed(0)
    batch_heads, tokens, head_dim = 5, 7, 8
    q = torch.randn(batch_heads, tokens, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    active_mask = torch.zeros(batch_heads, tokens, 1, dtype=q.dtype)
    # Leave the first head fully inactive to exercise the zero path.
    for head in range(1, batch_heads):
        mask = (torch.rand(tokens) > 0.35)
        if not mask.any():
            mask[0] = True
        active_mask[head, mask, 0] = 1.0
    causal_mask = torch.triu(torch.full((tokens, tokens), float("-inf"), dtype=q.dtype), diagonal=1)

    out_sparse = dmoah_sparse_attention(
        q,
        k,
        v,
        active_mask=active_mask,
        causal_mask=causal_mask,
        training=False,
    )

    reference = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(head_dim)
    mask_bool = active_mask.squeeze(-1) > 0
    for head_idx in range(batch_heads):
        token_idx = mask_bool[head_idx].nonzero(as_tuple=True)[0]
        if token_idx.numel() == 0:
            continue
        q_sel = q[head_idx].index_select(0, token_idx)
        att_scores = torch.matmul(q_sel, k[head_idx].transpose(0, 1)) * scale
        att_scores = att_scores + causal_mask.index_select(0, token_idx)
        att_probs = torch.softmax(att_scores, dim=-1)
        reference[head_idx].index_copy_(0, token_idx, torch.matmul(att_probs, v[head_idx]))

    assert torch.allclose(out_sparse, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("percentile", [1.0, 0.999])
def test_dmoah_sparse_attention_quantized_int8_close_to_fp(percentile):
    torch.manual_seed(42)
    batch_heads, tokens, head_dim = 4, 10, 16
    q = torch.randn(batch_heads, tokens, head_dim, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    active_mask = torch.zeros(batch_heads, tokens, 1, dtype=torch.float32)
    for head in range(batch_heads):
        mask = torch.rand(tokens) > 0.4
        if not mask.any():
            mask[0] = True
        active_mask[head, mask, 0] = 1.0
    causal_mask = torch.triu(torch.full((tokens, tokens), float("-inf"), dtype=torch.float32), diagonal=1)

    def quantize_per_head(tensor):
        flat = tensor.view(tensor.size(0), -1)
        max_vals = flat.abs().amax(dim=1, keepdim=True)
        if percentile < 1.0:
            qtile = torch.quantile(flat.abs(), percentile, dim=1, keepdim=True)
            base = torch.where(qtile > 0, qtile, max_vals)
        else:
            base = max_vals
        scales = torch.clamp(base / 127.0, min=1e-6)
        quant = torch.clamp((flat / scales).round(), -127.0, 127.0).to(torch.int8)
        return quant.view_as(tensor), scales.squeeze(1)

    q_int8, q_scales = quantize_per_head(q)
    k_int8, k_scales = quantize_per_head(k)
    v_int8, v_scales = quantize_per_head(v)

    out_float = dmoah_sparse_attention(
        q,
        k,
        v,
        active_mask=active_mask,
        causal_mask=causal_mask,
        training=False,
    )
    out_quant = dmoah_sparse_attention(
        q_int8,
        k_int8,
        v_int8,
        active_mask=active_mask,
        causal_mask=causal_mask,
        training=False,
        q_scale=q_scales,
        k_scale=k_scales,
        v_scale=v_scales,
        out_dtype=q.dtype,
    )

    diff = (out_float - out_quant).abs()
    assert diff.mean() < 0.05
    assert diff.max() < 0.2


@pytest.mark.parametrize(
    ("use_local", "use_anchors", "use_dna"),
    [
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ],
)
def test_flux_candidates_match_reference_cpu(use_local, use_anchors, use_dna):
    torch.manual_seed(42)
    seq_len = 64
    rows = torch.randint(0, seq_len, (16,), dtype=torch.long)
    linear_L = 16
    linear_window = 12
    anchor_stride = 4
    dna_scores = torch.rand(seq_len) if use_dna else None

    expected_candidates, expected_lengths = _reference_shortlist(
        rows,
        seq_len,
        linear_L,
        linear_window,
        anchor_stride,
        use_local,
        use_anchors,
        dna_scores,
    )
    flux_candidates, flux_lengths = build_flux_candidates(
        rows,
        seq_len,
        max_candidates=linear_L,
        linear_window=linear_window,
        anchor_stride=anchor_stride,
        use_local=use_local,
        use_anchors=use_anchors,
        dna_scores=dna_scores,
    )
    assert flux_candidates.shape == expected_candidates.shape
    assert torch.equal(flux_candidates, expected_candidates)
    assert torch.equal(flux_lengths, expected_lengths)


@pytest.mark.parametrize("use_dna", [False, True])
def test_build_packed_flux_candidates_matches_per_head(use_dna: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    active_heads = 3
    rows_per_head = torch.tensor([5, 3, 4], dtype=torch.long, device=device)
    row_offsets = torch.zeros(active_heads + 1, dtype=torch.int32, device=device)
    torch.cumsum(rows_per_head.to(torch.int32), dim=0, out=row_offsets[1:])

    seq_len = 32
    token_chunks = []
    for count in rows_per_head.tolist():
        tokens = torch.randint(0, seq_len, (count,), device=device)
        tokens, _ = torch.sort(tokens)
        token_chunks.append(tokens)
    token_idx = torch.cat(token_chunks, dim=0)

    h_total = 8
    head_indices = torch.tensor([1, 9, 10], dtype=torch.long, device=device)

    runtime_kwargs = dict(
        max_candidates=8,
        linear_window=6,
        anchor_stride=3,
        use_local=True,
        use_anchors=True,
        local_cap=5,
        anchor_cap=6,
        dna_cap=6,
    )

    if use_dna:
        dna_scores = torch.rand(2, seq_len, device=device)
        row_counts = rows_per_head.to(torch.long)
        head_range = torch.arange(active_heads, device=device, dtype=torch.long)
        row_head_ids = torch.repeat_interleave(head_range, row_counts)
        row_batch_ids = (head_indices[row_head_ids] // h_total).to(
            device=device, dtype=torch.long
        )
    else:
        dna_scores = None
        row_batch_ids = None

    packed_candidates, packed_lengths = build_packed_flux_candidates(
        token_idx,
        row_batch_ids,
        seq_len,
        dna_scores=dna_scores,
        **runtime_kwargs,
    )

    baseline_candidates = token_idx.unsqueeze(1).expand(-1, runtime_kwargs["max_candidates"]).clone()
    baseline_lengths = torch.ones(token_idx.size(0), dtype=torch.long, device=device)

    for head in range(active_heads):
        start = int(row_offsets[head].item())
        end = int(row_offsets[head + 1].item())
        rows = token_idx[start:end]
        dna_vec = None
        if use_dna:
            batch_index = int((head_indices[head] // h_total).item())
            dna_vec = dna_scores[batch_index]
        candidates, lengths = build_flux_candidates(
            rows,
            seq_len,
            dna_scores=dna_vec,
            **runtime_kwargs,
        )
        baseline_candidates[start:end] = candidates
        baseline_lengths[start:end] = lengths

    assert torch.equal(packed_candidates, baseline_candidates)
    assert torch.equal(packed_lengths, baseline_lengths)


def test_dmoah_sparse_attention_matches_sdpa_without_mask():
    torch.manual_seed(1)
    batch_heads, tokens, head_dim = 3, 6, 5
    q = torch.randn(batch_heads, tokens, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    causal_mask = torch.triu(torch.full((tokens, tokens), float("-inf"), dtype=q.dtype), diagonal=1)

    out_sparse = dmoah_sparse_attention(
        q,
        k,
        v,
        causal_mask=causal_mask,
        training=False,
    )
    out_dense = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=causal_mask.unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
    )

    assert torch.allclose(out_sparse, out_dense, atol=1e-6, rtol=1e-5)


def test_causal_dynamic_attention_linear_matches_dense_when_full_window():
    torch.manual_seed(123)
    base_kwargs = dict(
        vocab_size=32,
        n_ctx=128,
        n_layer=1,
        n_head=4,
        d_model=64,
        p_dropout=0.0,
        bias=False,
        attn_variant="dmoah",
        attn_h_total=4,
        attn_h_active=4,
        attn_h_active_min=4,
        attn_h_active_max=4,
        attn_active_seq_low=1,
        attn_active_seq_high=2,
        attn_force_dense_threshold=None,
        attn_token_sparse=False,
    )

    cfg_subquad = ModelConfig(attn_mode="subquad", **base_kwargs)
    cfg_linear = ModelConfig(
        attn_mode="linear",
        attn_linear_L=128,
        attn_linear_window=128,
        attn_linear_anchor_stride=0,
        attn_linear_switch_ctx=1,
        attn_linear_head_k=4,
        **base_kwargs,
    )

    torch.manual_seed(123)
    attn_subquad = CausalDynamicAttention(cfg_subquad)
    torch.manual_seed(123)
    attn_linear = CausalDynamicAttention(cfg_linear)
    attn_linear.load_state_dict(attn_subquad.state_dict())

    x = torch.randn(2, 32, cfg_linear.d_model)
    attn_subquad.eval()
    attn_linear.eval()
    with torch.no_grad():
        out_sub = attn_subquad(x)
        out_lin = attn_linear(x)

    assert torch.allclose(out_lin, out_sub, atol=1e-5, rtol=1e-4)
    mode_label = attn_linear.last_head_stats.get("mode")
    sparse_mode = attn_linear._last_sparse_state.get("mode")
    assert mode_label in {"linear", "flux"}
    assert sparse_mode in {"linear", "flux"}


@pytest.mark.parametrize(
    ("policy", "anchor_stride", "dna_enabled", "expect_close"),
    [
        ("local", 0, False, True),
        ("local+anchors", 4, False, True),
        ("local+dna", 0, True, True),
    ],
)
def test_causal_dynamic_attention_linear_policy_variants(policy, anchor_stride, dna_enabled, expect_close):
    torch.manual_seed(7)
    base_kwargs = dict(
        vocab_size=40,
        n_ctx=96,
        n_layer=1,
        n_head=4,
        d_model=64,
        p_dropout=0.0,
        bias=False,
        attn_variant="dmoah",
        attn_h_total=4,
        attn_h_active=4,
        attn_h_active_min=4,
        attn_h_active_max=4,
        attn_active_seq_low=1,
        attn_active_seq_high=2,
        attn_force_dense_threshold=None,
        attn_token_sparse=False,
    )

    cfg_subquad = ModelConfig(
        attn_mode="subquad",
        attn_dna_enable=dna_enabled,
        attn_dna_threshold=0.0,
        attn_dna_blend=1.0 if dna_enabled else 0.0,
        **base_kwargs,
    )
    cfg_linear = ModelConfig(
        attn_mode="linear",
        attn_linear_L=96,
        attn_linear_window=96,
        attn_linear_anchor_stride=anchor_stride,
        attn_linear_policy=policy,
        attn_linear_head_k=4,
        attn_dna_enable=dna_enabled,
        attn_dna_threshold=0.0,
        attn_dna_blend=1.0 if dna_enabled else 0.0,
        **base_kwargs,
    )

    torch.manual_seed(7)
    attn_subquad = CausalDynamicAttention(cfg_subquad)
    torch.manual_seed(7)
    attn_linear = CausalDynamicAttention(cfg_linear)
    attn_linear.load_state_dict(attn_subquad.state_dict())

    x = torch.randn(2, 48, cfg_linear.d_model)
    attn_subquad.eval()
    attn_linear.eval()
    with torch.no_grad():
        out_sub = attn_subquad(x)
        out_lin = attn_linear(x)

    if expect_close:
        assert torch.allclose(out_lin, out_sub, atol=1e-5, rtol=1e-4)
    else:
        assert not torch.allclose(out_lin, out_sub, atol=1e-5, rtol=1e-4)
    mode_label = attn_linear.last_head_stats.get("mode")
    sparse_mode = attn_linear._last_sparse_state.get("mode")
    assert mode_label in {"linear", "flux"}
    assert sparse_mode in {"linear", "flux"}
    backend = attn_linear.last_head_stats.get("linear_shortlist_backend")
    assert backend in {"torch", "triton", "fallback_dense"}

def test_causal_dynamic_attention_latency_budget_auto_switch():
    cfg = ModelConfig(
        vocab_size=16,
        n_ctx=128,
        n_layer=1,
        n_head=4,
        d_model=32,
        p_dropout=0.0,
        bias=False,
        attn_variant="dmoah",
        attn_h_total=4,
        attn_h_active=4,
        attn_h_active_min=2,
        attn_h_active_max=4,
        attn_mode="auto",
        attn_linear_switch_ctx=1024,
        attn_linear_latency_budget_ms=0.0,
        attn_token_sparse=False,
    )

    attn = CausalDynamicAttention(cfg)
    attn._latency_ema = 5.0  # simulate high latency
    assert attn._select_mode(128) == "linear"
    attn._latency_ema = 0.0
    attn._current_mode = "linear"
    assert attn._select_mode(64) == "subquad"


def test_gpt_dmoah_matches_dense_when_all_heads_active():
    torch.manual_seed(321)
    cfg_dense = ModelConfig(
        vocab_size=48,
        n_ctx=16,
        n_layer=2,
        n_head=4,
        d_model=64,
        p_dropout=0.0,
        bias=False,
        attn_variant="standard",
    )
    torch.manual_seed(321)
    model_dense = GPT(cfg_dense)

    cfg_sparse = ModelConfig(
        vocab_size=48,
        n_ctx=16,
        n_layer=2,
        n_head=4,
        d_model=64,
        p_dropout=0.0,
        bias=False,
        attn_variant="dmoah",
        attn_h_total=4,
        attn_h_active=4,
        attn_gates=8,
        attn_router_noise_std=0.0,
        attn_use_rope=False,
    )
    torch.manual_seed(321)
    model_sparse = GPT(cfg_sparse)

    load_res = model_sparse.load_state_dict(model_dense.state_dict(), strict=False)
    assert not load_res.unexpected_keys
    assert all("head_router" in key for key in load_res.missing_keys)

    model_dense.eval()
    model_sparse.eval()

    idx = torch.randint(0, cfg_dense.vocab_size, (2, 12))
    targets = torch.randint(0, cfg_dense.vocab_size, (2, 12))

    with torch.no_grad():
        logits_dense, loss_dense = model_dense(idx, targets)
        logits_sparse, loss_sparse = model_sparse(idx, targets)

    assert torch.allclose(logits_sparse, logits_dense, atol=1e-5, rtol=1e-4)
    assert torch.allclose(loss_sparse, loss_dense, atol=1e-5, rtol=1e-4)
    assert model_sparse.last_attn_stats is not None
    assert len(model_sparse.last_attn_stats) == cfg_sparse.n_layer


@pytest.mark.parametrize(
    "mode,percentile",
    [
        ("max", 1.0),
        ("percentile", 0.999),
        ("ema_percentile", 0.999),
    ],
)
def test_causal_dynamic_attention_quantized_path_matches_fp(mode, percentile):
    torch.manual_seed(123)
    base_cfg = ModelConfig(
        vocab_size=32,
        n_ctx=256,
        n_layer=1,
        n_head=4,
        d_model=64,
        p_dropout=0.0,
        bias=False,
        attn_variant="dmoah",
        attn_h_total=8,
        attn_h_active=4,
        attn_h_active_min=2,
        attn_h_active_max=4,
        attn_active_seq_low=16,
        attn_active_seq_high=256,
        attn_small_seq_dense=0,
        attn_force_dense_threshold=None,
        attn_router_noise_std=0.0,
    )
    quant_kwargs = dict(vars(base_cfg))
    quant_kwargs["attn_quantize_int8"] = True
    quant_kwargs["attn_quantize_int8_mode"] = mode
    quant_kwargs["attn_quantize_int8_percentile"] = percentile
    quant_cfg = ModelConfig(**quant_kwargs)

    attn_fp = CausalDynamicAttention(base_cfg)
    attn_q = CausalDynamicAttention(quant_cfg)
    attn_q.load_state_dict(attn_fp.state_dict())

    attn_fp.eval()
    attn_q.eval()

    x = torch.randn(2, 128, base_cfg.d_model)
    with torch.no_grad():
        out_fp = attn_fp(x)
        out_q = attn_q(x)
        if "ema" in mode:
            ema_before = attn_q._quant_ema_q.clone()
            _ = attn_q(x * 1.1)
            assert not torch.equal(ema_before, attn_q._quant_ema_q)

    diff = (out_fp - out_q).abs()
    assert diff.mean() < 0.05
    assert diff.max() < 0.2


def test_dmoah_active_head_schedule_respects_min_max_bounds():
    cfg = ModelConfig(
        vocab_size=32,
        n_ctx=4096,
        n_layer=1,
        n_head=8,
        d_model=64,
        p_dropout=0.0,
        bias=False,
        attn_variant="dmoah",
        attn_h_total=32,
        attn_h_active=8,
        attn_h_active_min=2,
        attn_h_active_max=16,
        attn_active_seq_low=512,
        attn_active_seq_high=4096,
    )
    attn = CausalDynamicAttention(cfg)

    assert attn.h_active_curve == pytest.approx(0.25)
    assert attn._choose_active_k(64) == attn.h_active_max
    assert attn._choose_active_k(cfg.attn_active_seq_low) == attn.h_active_max

    assert attn._choose_active_k(1024) == 7
    assert attn._choose_active_k(2048) == 4
    assert attn._choose_active_k(attn.seq_high) == attn.h_active_min
    assert attn._choose_active_k(attn.seq_high * 2) == attn.h_active_min

    samples = [attn._choose_active_k(s) for s in (attn.seq_low, 1024, 2048, attn.seq_high, attn.seq_high * 2)]
    assert samples == sorted(samples, reverse=True)


def test_cli_preset_dmoah_sets_variant():
    if _prepare_training_configs is None:
        pytest.skip("Typer/CLI dependencies not available")
    mc, _tc, _oc, _dc = _prepare_training_configs(preset="dmoah", config_files=None, overrides=None)
    assert mc.attn_variant == "dmoah"
    assert mc.attn_h_total > 0
    assert 0 < mc.attn_h_active <= mc.attn_h_total


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device is required for Triton path test")
def test_dmoah_sparse_attention_triton_gpu_matches_cpu():
    if not getattr(sparse_attn, "TRITON_AVAILABLE", False):
        pytest.skip("Triton runtime not available")

    device = torch.device("cuda")
    torch.manual_seed(7)

    batch_heads, tokens, head_dim = 6, 9, 32
    q = torch.randn(batch_heads, tokens, head_dim, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Build a mask with sparse structure: some rows inactive, others partially active.
    active_mask = torch.ones(batch_heads, tokens, 1, device=device, dtype=torch.float32)
    active_mask[0, :, :] = 0.0
    active_mask[2, 4:, :] = 0.0
    active_mask[4, :3, :] = 0.0

    causal_mask = torch.triu(torch.full((tokens, tokens), float("-inf"), device=device, dtype=torch.float32), diagonal=1)

    assert sparse_attn._should_try_triton(
        q,
        k,
        v,
        active_mask=active_mask.squeeze(-1),
        dropout_p=0.0,
        training=False,
        causal_mask=causal_mask,
        flux_candidates=None,
        flux_lengths=None,
    )

    out_gpu = dmoah_sparse_attention(
        q,
        k,
        v,
        active_mask=active_mask,
        causal_mask=causal_mask,
        dropout_p=0.0,
        training=False,
    ).cpu()

    out_cpu = dmoah_sparse_attention(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        active_mask=active_mask.cpu(),
        causal_mask=causal_mask.cpu(),
        dropout_p=0.0,
        training=False,
    )

    assert torch.allclose(out_gpu, out_cpu, atol=5e-4, rtol=5e-4)
