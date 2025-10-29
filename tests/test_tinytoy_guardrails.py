import torch

from protean_forge.kernels import tinytoy


def test_build_sequence_lengths_cpu_respects_cap():
    device = torch.device("cpu")
    seqs = tinytoy.build_sequence_lengths(device=device, max_seq_len=1024)
    assert seqs == [128, 256, 512, 1024]


def test_build_sequence_lengths_cpu_defaults_when_filtered_out():
    device = torch.device("cpu")
    seqs = tinytoy.build_sequence_lengths(device=device, max_seq_len=64)
    assert seqs == [128]


def test_build_sequence_lengths_gpu_growth_cap():
    device = torch.device("cuda")
    seqs = tinytoy.build_sequence_lengths(device=device, max_seq_len=4096, gpu_start=128, gpu_cap=16384)
    assert seqs == [128, 256, 512, 1024, 2048, 4096]
