import pytest
import torch
import torch.nn as nn

from proteus_attention.modules import CausalASPAMultiheadAttention, SparseHeadController
from proteus_attention.modules import CausalASPATransformerBlock


@pytest.mark.parametrize("batch_first", [True, False])
def test_causal_aspa_multihead_shapes(batch_first: bool):
    torch.manual_seed(0)
    mha = CausalASPAMultiheadAttention(
        embed_dim=32,
        num_heads=4,
        dropout=0.1,
        batch_first=batch_first,
        max_seq_len=512,
    )
    if batch_first:
        x = torch.randn(2, 16, 32)
    else:
        x = torch.randn(16, 2, 32)

    out, weights = mha(x, need_weights=True)
    expected_shape = x.shape
    assert out.shape == expected_shape
    assert weights is not None
    if batch_first:
        assert weights.shape == (2, 16, 16)
    else:
        assert weights.shape == (2, 16, 16)


def test_causal_aspa_multihead_key_padding_mask():
    torch.manual_seed(0)
    mha = CausalASPAMultiheadAttention(
        embed_dim=16,
        num_heads=2,
        batch_first=True,
    )
    x = torch.randn(3, 8, 16)
    mask = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    out, _ = mha(x, key_padding_mask=mask, need_weights=False)
    # Entirely masked batch should output zeros due to zeroed inputs.
    assert torch.allclose(out[2], torch.zeros_like(out[2]), atol=1e-5)


def test_causal_aspa_multihead_cross_attn_matches_torch():
    torch.manual_seed(1)
    embed_dim, num_heads = 16, 4
    mha = CausalASPAMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.0)
    reference = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.0)
    reference.load_state_dict(mha.fallback_attention.state_dict())

    q = torch.randn(2, 5, embed_dim)
    k = torch.randn(2, 7, embed_dim)
    v = torch.randn(2, 7, embed_dim)

    expected_out, expected_weights = reference(q, k, v, need_weights=True)
    out, weights = mha(q, key=k, value=v, need_weights=True)

    assert torch.allclose(out, expected_out, atol=1e-6)
    assert weights is not None and torch.allclose(weights, expected_weights, atol=1e-6)


def test_causal_aspa_multihead_accepts_standard_causal_mask():
    mha = CausalASPAMultiheadAttention(embed_dim=8, num_heads=1)
    x = torch.randn(2, 4, 8)
    causal_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    out, _ = mha(x, attn_mask=causal_mask, need_weights=False)
    assert out.shape == x.shape


def test_causal_aspa_multihead_handles_custom_mask_with_fallback():
    torch.manual_seed(2)
    embed_dim = 12
    mha = CausalASPAMultiheadAttention(embed_dim=embed_dim, num_heads=3, batch_first=True, dropout=0.0)

    x = torch.randn(3, 6, embed_dim)
    custom_mask = torch.full((6, 6), float("-inf"))
    custom_mask.triu_(1)
    custom_mask[0, 3] = 0.0  # allow a non-causal interaction

    out, weights = mha(
        x,
        attn_mask=custom_mask,
        need_weights=True,
        average_attn_weights=False,
        is_causal=False,
    )

    ref = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=3, batch_first=True, dropout=0.0)
    ref.load_state_dict(mha.fallback_attention.state_dict())
    ref_out, ref_w = ref(
        x,
        x,
        x,
        attn_mask=custom_mask,
        need_weights=True,
        average_attn_weights=False,
        is_causal=False,
    )

    assert torch.allclose(out, ref_out, atol=1e-6)
    assert weights is not None and torch.allclose(weights, ref_w, atol=1e-6)


def test_causal_aspa_multihead_runtime_stats_toggle():
    mha = CausalASPAMultiheadAttention(embed_dim=8, num_heads=1, return_stats_default=False)
    x = torch.randn(2, 4, 8)
    out, _ = mha(x, need_weights=False, return_stats=True)
    assert out.shape == x.shape
    stats = mha.last_runtime_stats
    assert stats is not None
    assert "head_stats" in stats


def test_sparse_head_controller_decrease_path():
    class DummyAttention:
        def __init__(self):
            self.h_total = 4
            self.h_active_min = 1
            self.h_active_max = 4
            self.h_active = 4

    layer = DummyAttention()
    ctl = SparseHeadController(layer, target_density=0.0, tolerance=0.0, cooldown=1, verbose=False)
    snapshot = ctl.observe({"max_active_density": 0.75})
    assert snapshot is not None
    assert snapshot.action == "decrease"
    assert layer.h_active_max == 3


def test_multihead_with_sparse_controller_collects_stats():
    torch.manual_seed(0)
    mha = CausalASPAMultiheadAttention(
        embed_dim=32,
        num_heads=4,
        batch_first=True,
        sparse_ctl_config={"target_density": 0.05, "tolerance": 0.0, "cooldown": 1, "verbose": False},
        return_stats_default=True,
    )
    assert mha.sparse_controller is not None
    x = torch.randn(2, 8, 32)
    mha(x, need_weights=False)
    stats = mha.last_runtime_stats
    assert stats is not None
    assert "sparse_ctl" in stats
    if stats["sparse_ctl"] is not None:
        assert "action" in stats["sparse_ctl"]


def test_aspa_attention_random_forward_outputs_finite():
    torch.manual_seed(42)
    mha = CausalASPAMultiheadAttention(
        embed_dim=48,
        num_heads=6,
        dropout=0.1,
        batch_first=True,
        max_seq_len=256,
        return_stats_default=True,
    )
    x = torch.randn(3, 12, 48)
    out, weights = mha(x, need_weights=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert weights is not None
    assert weights.shape == (3, 12, 12)
    stats = mha.last_runtime_stats
    assert stats is not None
    assert "head_stats" in stats


def test_aspa_attention_backward_propagates_gradients():
    torch.manual_seed(7)
    mha = CausalASPAMultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
    x = torch.randn(2, 10, 32, requires_grad=True)
    out, _ = mha(x, need_weights=False)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_causal_aspa_multihead_fallback_populates_stats():
    mha = CausalASPAMultiheadAttention(embed_dim=8, num_heads=1, return_stats_default=True)
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 3, 8)
    out, _ = mha(q, key=k, value=k, need_weights=False, is_causal=False)
    assert out.shape == q.shape
    stats = mha.last_runtime_stats
    assert stats is not None
    assert stats["shortlist_backend"]["name"] == "torch.nn.MultiheadAttention"


def test_causal_aspa_transformer_block_forward_backward():
    torch.manual_seed(0)
    block = CausalASPATransformerBlock(
        embed_dim=32,
        num_heads=4,
        dim_feedforward=64,
        dropout=0.0,
        attention_kwargs={"attn_small_seq_dense": 64},
    )
    x = torch.randn(2, 12, 32, requires_grad=True)

    out, weights = block(x, return_attn=True)
    assert out.shape == x.shape
    assert weights is not None and weights.shape == (2, 12, 12)

    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
