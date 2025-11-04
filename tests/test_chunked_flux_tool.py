import torch

from proteus_attention.tools.chunked_flux import (
    ChunkedFluxConfig,
    ChunkedFluxRunner,
)


def test_chunked_flux_cpu_small_run():
    device = torch.device("cpu")
    config = ChunkedFluxConfig(
        seq_len=64,
        d_model=32,
        chunk_len=16,
        buffer_tokens=12,
        per_chunk_budget=6,
        device=device,
        heads=4,
        chunk_sparse_ratio=0.25,
        final_sparse_ratio=0.5,
        seed=123,
        report_latency=False,
        progress=False,
        run_final_pass=True,
    )
    runner = ChunkedFluxRunner(config)
    result = runner.run()

    assert result.keep_indices.numel() <= config.buffer_tokens
    assert result.reduced_sequence.size(1) == result.keep_indices.numel()
    assert result.metrics.original_tokens == config.seq_len
    assert result.metrics.retained_tokens == result.keep_indices.numel()
    assert result.final_output is not None
    assert isinstance(result.metrics.used_fallback, bool)
    assert result.metrics.total_time_ms and result.metrics.total_time_ms > 0
    assert result.metrics.chunk_tokens_per_s and result.metrics.chunk_tokens_per_s > 0
    assert result.metrics.final_tokens_per_s and result.metrics.final_tokens_per_s > 0
    assert result.metrics.total_tokens_per_s and result.metrics.total_tokens_per_s > 0
    if result.final_stats is not None:
        assert isinstance(result.final_stats, dict)
    if result.backend_info is not None:
        assert isinstance(result.backend_info, dict)


def test_chunked_flux_custom_sequence_respects_input():
    device = torch.device("cpu")
    seq_len = 48
    d_model = 16
    sequence = torch.randn(1, seq_len, d_model)
    config = ChunkedFluxConfig(
        seq_len=seq_len,
        d_model=d_model,
        chunk_len=12,
        buffer_tokens=10,
        per_chunk_budget=5,
        device=device,
        heads=2,
        chunk_sparse_ratio=0.3,
        final_sparse_ratio=0.6,
        seed=None,
        report_latency=False,
        progress=False,
        run_final_pass=False,
    )
    runner = ChunkedFluxRunner(config)
    result = runner.run(sequence=sequence)

    gathered = sequence[:, result.keep_indices]
    assert torch.allclose(result.reduced_sequence.cpu(), gathered, atol=1e-6, rtol=1e-5)
    assert result.final_output is None
    assert result.metrics.used_fallback is True
    assert result.metrics.chunk_tokens_per_s and result.metrics.chunk_tokens_per_s > 0
    assert result.metrics.final_tokens_per_s is None
