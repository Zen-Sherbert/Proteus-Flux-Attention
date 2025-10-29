import pytest
import torch

from protean_forge.modules import (
    CausalGeneticMultiheadAttention,
    SparseHeadController,
)


@pytest.mark.parametrize("batch_first", [True, False])
def test_causal_genetic_multihead_shapes(batch_first: bool):
    torch.manual_seed(0)
    mha = CausalGeneticMultiheadAttention(
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


def test_causal_genetic_multihead_key_padding_mask():
    torch.manual_seed(0)
    mha = CausalGeneticMultiheadAttention(
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


def test_causal_genetic_multihead_rejects_cross_attn():
    mha = CausalGeneticMultiheadAttention(embed_dim=8, num_heads=1)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    with pytest.raises(NotImplementedError):
        _ = mha(q, key=k)


def test_causal_genetic_multihead_accepts_standard_causal_mask():
    mha = CausalGeneticMultiheadAttention(embed_dim=8, num_heads=1)
    x = torch.randn(2, 4, 8)
    causal_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    out, _ = mha(x, attn_mask=causal_mask, need_weights=False)
    assert out.shape == x.shape


def test_causal_genetic_multihead_rejects_custom_mask():
    mha = CausalGeneticMultiheadAttention(embed_dim=8, num_heads=1)
    x = torch.randn(2, 4, 8)
    custom_mask = torch.zeros(4, 4, dtype=torch.bool)
    custom_mask[0, 2] = True  # non-causal pattern
    with pytest.raises(NotImplementedError):
        _ = mha(x, attn_mask=custom_mask, need_weights=False)


def test_causal_genetic_multihead_runtime_stats_toggle():
    mha = CausalGeneticMultiheadAttention(embed_dim=8, num_heads=1, return_stats_default=False)
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
    mha = CausalGeneticMultiheadAttention(
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


def test_genetic_attention_random_forward_outputs_finite():
    torch.manual_seed(42)
    mha = CausalGeneticMultiheadAttention(
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


def test_genetic_attention_backward_propagates_gradients():
    torch.manual_seed(7)
    mha = CausalGeneticMultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
    x = torch.randn(2, 10, 32, requires_grad=True)
    out, _ = mha(x, need_weights=False)
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
