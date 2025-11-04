import torch

from proteus_attention.models.dmoah import CausalDynamicAttention, ModelConfig


def test_rope_checkpoint_roundtrip(tmp_path):
    config = ModelConfig(
        vocab_size=32,
        n_ctx=64,
        n_layer=1,
        n_head=4,
        d_model=128,
        attn_variant="dmoah",
        attn_use_rope=True,
    )
    model = CausalDynamicAttention(config)
    payload = {
        "config": config.to_dict(),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    ckpt_path = tmp_path / "rope.pt"
    torch.save(payload, ckpt_path)

    loaded = torch.load(ckpt_path, map_location="cpu")
    restored = CausalDynamicAttention(ModelConfig(**loaded["config"]))
    restored.load_state_dict(loaded["state_dict"])

    assert restored.use_rope
    assert restored._rope_cache == {}

    x = torch.randn(2, 32, restored.d_model)
    with torch.no_grad():
        y = restored(x)

    assert y.shape == x.shape
    assert restored._rope_cache
