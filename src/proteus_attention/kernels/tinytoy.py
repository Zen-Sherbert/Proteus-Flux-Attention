# benchmark.py
from proteus_attention.models.dmoah import CausalDynamicAttention
from proteus_attention.kernels.sparse_attn import get_last_backend_info
import argparse
import json
import os
import sys
import time
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

# Ensure the repo's `src` tree is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Import your custom modules ---

# Ensure CUDA allocator uses expandable segments to reduce fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# --- Helper Functions ---


def get_device():
    """Gets the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Add other devices here if needed, e.g., 'mps' for Apple Silicon
    return torch.device("cpu")


@torch.inference_mode()
def benchmark_forward_pass(model, x, num_runs=50):
    """
    Runs a forward pass benchmark on a given model.
    Includes a warmup phase.
    """
    model.eval()
    device = get_device()
    model.to(device)
    x = x.to(device)

    seq_len = int(x.shape[1]) if x.dim() >= 2 else 0
    if seq_len >= 16384:
        warmup_runs = 0
    elif seq_len >= 8192:
        warmup_runs = 1
    else:
        warmup_runs = 10

    # Warmup runs
    for _ in range(warmup_runs):
        _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    if seq_len >= 524288:
        eval_runs = 1
    elif seq_len >= 131072:
        eval_runs = min(num_runs, 2)
    elif seq_len >= 65536:
        eval_runs = min(num_runs, 2)
    elif seq_len >= 32768:
        eval_runs = min(num_runs, 3)
    elif seq_len >= 16384:
        eval_runs = min(num_runs, 5)
    elif seq_len >= 8192:
        eval_runs = min(num_runs, 5)
    else:
        eval_runs = num_runs

    start_time = time.perf_counter()
    for _ in range(eval_runs):
        _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / eval_runs) * 1000
    return avg_latency_ms


@torch.inference_mode()
def get_peak_memory_mb(model, x):
    """Measures the peak GPU memory allocated during a forward pass."""
    device = get_device()
    if device.type != 'cuda':
        return 0.0

    model.to(device)
    x = x.to(device)

    torch.cuda.reset_peak_memory_stats(device)
    _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    return peak_memory_mb


DEFAULT_GPU_SEQUENCE_CAP = 1_000_000


def build_sequence_lengths(
    *,
    device: torch.device,
    max_seq_len: int | None,
    base_lengths: list[int] | None = None,
    gpu_start: int = 128,
    gpu_cap: int = DEFAULT_GPU_SEQUENCE_CAP,
) -> list[int]:
    """
    Build the benchmarking sequence schedule with optional guardrails.
    """
    base_lengths = base_lengths or [128, 256, 512, 1024, 2048, 4096]
    if device.type == "cuda":
        seq_lengths: list[int] = []
        current = max(gpu_start, 1)
        limit = gpu_cap if max_seq_len is None else min(max_seq_len, gpu_cap)
        while current <= limit:
            seq_lengths.append(current)
            current *= 2
        if not seq_lengths:
            seq_lengths = [min(base_lengths)]
    else:
        seq_lengths = [
            length for length in base_lengths if max_seq_len is None or length <= max_seq_len]
        if not seq_lengths:
            seq_lengths = [min(base_lengths)]
    return seq_lengths


def save_summary_to_disk(summary: dict) -> Path:
    """Persist benchmark summary JSON and return the destination path."""
    reports_dir = PROJECT_ROOT / "reports" / "tinytoy"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = reports_dir / f"tinytoy_summary_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return output_path


def build_dmoah_config(seq_len: int, seq_high: int, *, d_model: int, quantize: bool) -> SimpleNamespace:
    if seq_len <= 256:
        token_keep_ratio = 1.0
        token_keep_min = seq_len
        token_threshold = 0.0
        attn_force_dense_threshold = 0.20
        attn_h_active = 8
    elif seq_len <= 1024:
        token_keep_ratio = 0.8
        token_keep_min = 160
        token_threshold = 0.04
        attn_force_dense_threshold = 0.40
        attn_h_active = 6
    elif seq_len <= 4096:
        token_keep_ratio = 0.6
        token_keep_min = 192
        token_threshold = 0.06
        attn_force_dense_threshold = 0.60
        attn_h_active = 5
    else:
        token_keep_ratio = 0.45
        token_keep_min = 224
        token_threshold = 0.08
        attn_force_dense_threshold = 0.90
        attn_h_active = 4
    token_guard = max(4, seq_len // 128)

    return SimpleNamespace(
        d_model=d_model,
        p_dropout=0.0,
        use_sdpa=True,
        attn_h_total=32,
        attn_h_active=attn_h_active,
        attn_h_active_min=max(2, attn_h_active // 2),
        attn_h_active_max=max(attn_h_active, 8),
        attn_active_seq_low=256,
        attn_active_seq_high=seq_high,
        attn_small_seq_dense=0,
        n_ctx=seq_high,
        bias=False,
        attn_router_lambda=0.0,
        attn_router_beta=1.0,
        attn_router_reg_mode="entropy",
        attn_gates=64,
        attn_router_noise_std=0.0,
        attn_force_dense_threshold=attn_force_dense_threshold,
        attn_quantize_int8=quantize,
        attn_dna_enable=True,
        attn_dna_threshold=0.3,
        attn_dna_blend=0.6,
        attn_dna_temp=0.2,
        attn_dna_usage_boost=0.15,
        attn_dna_decay=0.97,
        attn_token_sparse=True,
        attn_token_keep_ratio=token_keep_ratio,
        attn_token_keep_min=token_keep_min,
        attn_token_keep_threshold=token_threshold,
        attn_token_keep_guard=token_guard,
        attn_linear_L_base=512,
        attn_linear_L_min=128,
        attn_linear_L_max=2048,
        attn_linear_L_schedule="sqrt",
        attn_linear_L_scale=1.0,
        attn_linear_L_scale_max=1.5,
        attn_linear_token_keep_schedule="sqrt",
        attn_linear_token_keep_min_ratio=0.2,
    )


def _apply_memory_saving_overrides(config: SimpleNamespace, seq_len: int) -> SimpleNamespace:
    """
    Tighten sparse attention settings to reduce memory footprint at long context lengths.
    """
    base_keep = getattr(config, "attn_token_keep_ratio", 1.0)
    target_keep_ratio = base_keep
    if seq_len >= 1_048_576:
        target_keep_ratio = min(target_keep_ratio, 0.0015)
    elif seq_len >= 524_288:
        target_keep_ratio = min(target_keep_ratio, 0.0025)
    elif seq_len >= 131072:
        target_keep_ratio = min(target_keep_ratio, 0.005)
    elif seq_len >= 32768:
        target_keep_ratio = min(target_keep_ratio, 0.01)
    elif seq_len >= 16384:
        target_keep_ratio = min(target_keep_ratio, 0.04)
    elif seq_len >= 8192:
        target_keep_ratio = min(target_keep_ratio, 0.10)
    elif seq_len >= 4096:
        target_keep_ratio = min(target_keep_ratio, 0.15)
    elif seq_len >= 2048:
        target_keep_ratio = min(target_keep_ratio, 0.20)
    elif seq_len >= 1024:
        target_keep_ratio = min(target_keep_ratio, 0.30)
    config.attn_token_keep_ratio = min(
        getattr(config, "attn_token_keep_ratio", 1.0), target_keep_ratio)
    min_keep_cfg = min(seq_len, 12 if seq_len >=
                       32768 else 24 if seq_len >= 8192 else 64)
    guard_val = max(int(getattr(config, "attn_token_keep_guard", 1) or 1), 1)
    if seq_len >= 32768:
        guard_val = min(guard_val, 64)
    elif seq_len >= 8192:
        guard_val = min(guard_val, 80)
    guard = guard_val
    config.attn_token_keep_min = max(min_keep_cfg, guard)
    config.attn_token_keep_guard = guard
    if seq_len >= 32768:
        active_heads = 1
        config.attn_h_active_min = 1
        config.attn_h_active_max = 1
    elif seq_len >= 8192:
        active_heads = max(1, min(getattr(config, "attn_h_active", 2), 2))
        config.attn_h_active_min = 1
        config.attn_h_active_max = 2
    elif seq_len >= 4096:
        active_heads = max(1, min(getattr(config, "attn_h_active", 2), 2))
        config.attn_h_active_min = 1
        config.attn_h_active_max = 2
    elif seq_len >= 2048:
        active_heads = max(1, min(getattr(config, "attn_h_active", 2), 2))
        config.attn_h_active_min = 1
        config.attn_h_active_max = 2
    else:
        active_heads = max(1, min(getattr(config, "attn_h_active", 4), 4))
        config.attn_h_active_min = min(2, active_heads)
        config.attn_h_active_max = 4
    config.attn_h_active = active_heads
    linear_cap = 1024
    if seq_len >= 1_048_576:
        linear_cap = 24
    elif seq_len >= 32768:
        linear_cap = 48
    elif seq_len >= 16384:
        linear_cap = 80
    elif seq_len >= 8192:
        linear_cap = 160
    elif seq_len >= 4096:
        linear_cap = 256
    config.attn_linear_L_max = min(
        getattr(config, "attn_linear_L_max", 2048), linear_cap)
    config.attn_linear_L_base = min(getattr(
        config, "attn_linear_L_base", config.attn_linear_L_max), config.attn_linear_L_max)
    min_cap = 64 if seq_len >= 16384 else 128
    min_cap = min(min_cap, linear_cap)
    config.attn_linear_L_min = min(
        getattr(config, "attn_linear_L_min", min_cap), min_cap)
    config.attn_linear_L = min(getattr(
        config, "attn_linear_L", config.attn_linear_L_base), config.attn_linear_L_max)
    if seq_len >= 262144:
        config.attn_dna_enable = False
    if seq_len >= 4096:
        if seq_len >= 1_048_576:
            target_budget = 12.0
        elif seq_len >= 32768:
            target_budget = 24.0
        elif seq_len >= 8192:
            target_budget = 140.0
        else:
            target_budget = 192.0
        current_budget = getattr(config, "attn_linear_mem_budget_mb", None)
        config.attn_linear_mem_budget_mb = float(
            current_budget) if current_budget not in (None, 0) else target_budget
        config.attn_linear_mem_budget_mb = min(
            config.attn_linear_mem_budget_mb, target_budget)
        config.attn_linear_L_schedule = "mem_cap"
    else:
        config.attn_linear_L_schedule = "log"
    config.attn_linear_switch_ctx = max(
        2048,
        min(int(getattr(config, "attn_linear_switch_ctx", 20000) or 20000), 8192),
    )
    config.attn_linear_policy = "local"
    return config


def _prepare_dmoah_config(
    seq_len: int,
    seq_high: int,
    *,
    d_model: int,
    quantize: bool,
    memory_guard: bool,
) -> SimpleNamespace:
    config = build_dmoah_config(
        seq_len, seq_high, d_model=d_model, quantize=quantize)
    if memory_guard:
        config = _apply_memory_saving_overrides(config, seq_len)
    return config


def _flux_alpha_from_seq(seq_len: int, *, seq_low: int, switch_ctx: int) -> float:
    seq_len = int(max(1, seq_len))
    low = max(1, seq_low)
    high = max(low + 1, switch_ctx) if switch_ctx > 0 else max(low + 1, seq_len)
    if seq_len <= low:
        return 0.0
    if seq_len >= high:
        return 1.0
    return float((seq_len - low) / (high - low))


def _effective_batch_size(seq_len: int, base_batch_size: int) -> int:
    """
    Reduce batch size as sequence length grows to avoid exhausting GPU memory.
    """
    if seq_len >= 16384:
        return 1
    if seq_len >= 8192:
        return 1
    return max(1, min(base_batch_size, 4))


def _instantiate_dmoah_model(
    seq_len: int,
    seq_high: int,
    *,
    d_model: int,
    quantize: bool,
    memory_guard: bool,
) -> tuple[CausalDynamicAttention, SimpleNamespace, float | None]:
    config = _prepare_dmoah_config(
        seq_len,
        seq_high,
        d_model=d_model,
        quantize=quantize,
        memory_guard=memory_guard,
    )
    model = CausalDynamicAttention(config)
    alpha: float | None = None
    if hasattr(model, "set_flux_alpha"):
        seq_low_cfg = int(getattr(config, "attn_active_seq_low", 256) or 256)
        switch_ctx_cfg = int(
            getattr(config, "attn_linear_switch_ctx", 8192) or 8192)
        alpha = _flux_alpha_from_seq(
            seq_len=seq_len,
            seq_low=seq_low_cfg,
            switch_ctx=switch_ctx_cfg,
        )
        model.set_flux_alpha(alpha)
    return model, config, alpha


_LINEAR_WARM_STATE: set[tuple[int, bool]] = set()


def _prime_linear_mode(
    *,
    device: torch.device,
    seq_target: int,
    seq_high: int,
    d_model: int,
    quantize: bool,
    memory_guard: bool,
) -> None:
    """
    Run a lightweight forward pass in linear mode to trigger Triton compilation ahead of time.
    """
    if device.type != "cuda":
        return
    device_index = int(device.index) if device.index is not None else 0
    key = (device_index, bool(quantize))
    if key in _LINEAR_WARM_STATE:
        return
    prime_len = min(2048, max(1024, seq_target))
    config = _prepare_dmoah_config(
        prime_len,
        max(seq_high, prime_len),
        d_model=d_model,
        quantize=quantize,
        memory_guard=memory_guard,
    )
    config.attn_mode = "linear"
    config.attn_linear_switch_ctx = 1
    model = CausalDynamicAttention(config)
    if hasattr(model, "set_flux_alpha"):
        model.set_flux_alpha(1.0)
    model.to(device)
    input_tensor = torch.randn(
        1, prime_len, d_model, device=device, dtype=torch.get_default_dtype())
    with torch.inference_mode():
        _ = model(input_tensor)
    torch.cuda.synchronize()
    del model
    torch.cuda.empty_cache()
    _LINEAR_WARM_STATE.add(key)


def estimate_safe_length(label: str,
                         sequence_lengths: list[int],
                         batch_size: int,
                         d_model: int,
                         model_factory) -> int | None:
    device = get_device()
    if device.type != 'cuda':
        return max(sequence_lengths)

    safe: int | None = None
    for seq_len in sequence_lengths:
        batch_eff = _effective_batch_size(seq_len, batch_size)
        input_tensor = torch.randn(batch_eff, seq_len, d_model)
        model = model_factory(seq_len)
        if device.type == 'cuda':
            if seq_len >= 16384:
                probe_runs = 1
            elif seq_len >= 8192:
                probe_runs = 1
            else:
                probe_runs = 3
        else:
            probe_runs = min(3, max(1, 4096 // max(1, seq_len)))
        print(
            f"[Probe] {label} testing seq_len={seq_len} batch={batch_eff} runs={probe_runs}", flush=True)
        try:
            benchmark_forward_pass(model, input_tensor, num_runs=probe_runs)
            safe = seq_len
        except RuntimeError as exc:
            message = str(exc)
            lower = message.lower()
            if "cuda" in message or "out of memory" in lower:
                print(
                    f"[Probe] {label} failed at sequence length {seq_len}: {message.splitlines()[-1]}")
                torch.cuda.empty_cache()
                break
            raise
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if safe is not None:
        print(
            f"[Probe] {label} safe up to sequence length {safe} on this device.")
    else:
        print(f"[Probe] {label} did not complete any probed lengths safely.")
    return safe


# --- Baseline Standard Attention Model ---

class StandardAttention(nn.Module):
    """A wrapper for a standard multi-head attention layer."""

    def __init__(self, config):
        super().__init__()
        # Use torch's built-in MHA which is highly optimized
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_head,
            dropout=0.0,
            batch_first=True
        )
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        # MultiheadAttention expects (query, key, value)
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        return self.proj(attn_output)

# --- Main Benchmark Execution ---


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DMoAH (Dynamic Mixture-of-Attention-Heads) kernels vs. standard attention."
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto", help="Force device selection.")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Limit the maximum sequence length.")
    parser.add_argument("--max-seq-count", type=int, default=None,
                        help="Limit the number of sequence lengths to evaluate.")
    parser.add_argument(
        "--gpu-cap",
        type=int,
        default=DEFAULT_GPU_SEQUENCE_CAP,
        help="Maximum sequence length explored on GPU when --max-seq-len is not provided (default: 10,000,000).",
    )
    parser.add_argument("--disable-standard", action="store_true",
                        help="Skip the dense attention baseline.")
    parser.add_argument("--disable-int8", action="store_true",
                        help="Disable the INT8 variant.")
    parser.add_argument("--skip-probe", action="store_true",
                        help="Skip allocation probes that search for safe lengths.")
    parser.add_argument("--report-dir", type=Path, default=None,
                        help="Override the directory where JSON summaries are written.")
    parser.add_argument("--no-save", action="store_true",
                        help="Disable writing JSON summaries to disk.")
    parser.add_argument(
        "--memory-guard",
        action="store_true",
        help="Enable conservative memory-saving overrides (legacy behaviour).",
    )
    parser.add_argument(
        "--cuda-bf16",
        action="store_true",
        help="Switch CUDA default dtype to bfloat16 for the benchmark run (default behaviour).",
    )
    parser.add_argument(
        "--cuda-fp32",
        action="store_true",
        help="Force CUDA default dtype to float32 (overrides the default bfloat16).",
    )
    return parser.parse_args(argv)


def resolve_device_choice(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available.")
        return torch.device("cuda")
    return get_device()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.max_seq_len is not None and args.max_seq_len < 1:
        raise ValueError("max-seq-len must be a positive integer.")
    if args.max_seq_count is not None and args.max_seq_count < 1:
        raise ValueError("max-seq-count must be a positive integer.")

    device = resolve_device_choice(args.device)

    if args.cuda_bf16 and args.cuda_fp32:
        raise ValueError("Specify at most one of --cuda-bf16 or --cuda-fp32.")

    use_bf16 = False
    if device.type == "cuda":
        use_bf16 = not args.cuda_fp32
        if args.cuda_bf16:
            use_bf16 = True
        torch.set_default_dtype(torch.bfloat16 if use_bf16 else torch.float32)

    # --- Configuration ---
    # We will use the same d_model but vary the head configurations
    D_MODEL = 512
    BASE_BATCH_SIZE = 4

    # Config for the standard model
    config_standard = SimpleNamespace(
        d_model=D_MODEL,
        n_head=8  # A typical number of heads
    )

    # Sequence lengths to test
    base_sequence_lengths = [128, 256, 512, 1024, 2048, 4096]
    sequence_lengths = build_sequence_lengths(
        device=device,
        max_seq_len=args.max_seq_len,
        base_lengths=base_sequence_lengths,
        gpu_cap=args.gpu_cap,
    )
    if args.max_seq_count is not None:
        sequence_lengths = sequence_lengths[: args.max_seq_count]

    seq_high = max(sequence_lengths)

    print("--- Starting Attention Benchmark ---")
    print(f"Device: {device}")
    print(f"Sequence lengths: {sequence_lengths}\n")

    probe_lengths = sequence_lengths if not args.skip_probe else []
    if device.type == 'cuda' and probe_lengths:
        safe_std = None
        safe_dmoah = None
        if not args.disable_standard:
            safe_std = estimate_safe_length(
                label="Standard",
                sequence_lengths=probe_lengths,
                batch_size=BASE_BATCH_SIZE,
                d_model=D_MODEL,
                model_factory=lambda _seq: StandardAttention(config_standard),
            )

        def _factory(seq: int) -> CausalDynamicAttention:
            model, _, _ = _instantiate_dmoah_model(
                seq,
                seq_high,
                d_model=D_MODEL,
                quantize=False,
                memory_guard=args.memory_guard,
            )
            return model

        safe_dmoah = estimate_safe_length(
            label="DMoAH",
            sequence_lengths=probe_lengths,
            batch_size=BASE_BATCH_SIZE,
            d_model=D_MODEL,
            model_factory=_factory,
        )
    else:
        safe_std = max(sequence_lengths) if not args.disable_standard else None
        safe_dmoah = max(sequence_lengths)

    runs_summary: list[dict[str, object]] = []

    print(f"{'Seq Len':<10} | {'Model':<24} | {'Latency (ms)':<15} | {'Seq/s':<12} | {'Tok/s':<15} | {'Memory (MB)':<15} | {'Active K':<10} | {'Tokens':<10} | {'Mode':<10} | {'Backend':<36}")
    print("-" * 212)

    std_limit_announced = False
    last_std_success = None
    dtype_label = "BF16" if use_bf16 else "FP32"
    dtype_key = "bf16" if use_bf16 else "fp32"
    variant_configs = [
        (f"DMoAH ({dtype_label})", False, dtype_key),
        ("DMoAH (INT8)", True, "int8"),
    ]
    if args.disable_int8:
        variant_configs = [entry for entry in variant_configs if not entry[1]]
    linear_trigger = min(
        (seq for seq in sequence_lengths if seq >= 8192), default=None)
    if device.type == 'cuda' and linear_trigger is not None:
        print(
            f"[Warmup] Priming flux kernel at seq_len={linear_trigger}...", flush=True)
        for _, quantize, _ in variant_configs:
            _prime_linear_mode(
                device=device,
                seq_target=linear_trigger,
                seq_high=seq_high,
                d_model=D_MODEL,
                quantize=quantize,
                memory_guard=args.memory_guard,
            )
    dmoah_limit_announced = {key: False for _, _, key in variant_configs}
    variant_active = {key: True for _, _, key in variant_configs}
    variant_last_success = {key: None for _, _, key in variant_configs}
    auto_logs: list[dict[str, object]] = []
    for seq_len in sequence_lengths:
        batch_eff = _effective_batch_size(seq_len, BASE_BATCH_SIZE)
        input_tensor = torch.randn(batch_eff, seq_len, D_MODEL)

        run_standard = (not args.disable_standard) and (
            safe_std is None or seq_len <= safe_std or device.type != 'cuda')
        std_record = {
            "seq_len": seq_len,
            "model": "standard",
            "batch_size": batch_eff,
        }
        if run_standard:
            model_std = StandardAttention(config_standard)
            latency_std = benchmark_forward_pass(model_std, input_tensor)
            throughput_std = (1000 / latency_std) * batch_eff
            mem_std = get_peak_memory_mb(model_std, input_tensor)
            del model_std
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tokens_std = throughput_std * seq_len
            std_latency = f"{latency_std:<15.2f}"
            std_throughput = f"{throughput_std:<12.2f}"
            std_mem = f"{mem_std:<15.2f}"
            std_backend = "torch.nn"
            std_mode = "-"
            last_std_success = seq_len
            std_record.update({
                "latency_ms": latency_std,
                "throughput_seq_s": throughput_std,
                "tokens_per_s": tokens_std,
                "memory_mb": mem_std,
                "status": "ok",
            })
        else:
            std_latency = f"{'OOM':<15}"
            std_throughput = f"{'-':<12}"
            std_tokens = f"{'-':<15}"
            std_mem = f"{'-':<15}"
            std_backend = "std-limit"
            std_mode = "-"
            std_record.update({
                "latency_ms": None,
                "throughput_seq_s": None,
                "tokens_per_s": None,
                "memory_mb": None,
                "status": "limit",
            })
            if not std_limit_announced:
                hit_at = seq_len
                prior = last_std_success if last_std_success is not None else "<unknown>"
                print(
                    f"[Notice] Standard attention hit its limit at seq_len={hit_at} (last safe ~{prior}).")
                std_limit_announced = True
        if run_standard:
            std_tokens = f"{tokens_std:<15.2f}"
            std_mode = "-"
        print(
            f"{seq_len:<10} | {'Standard Attention':<24} | {std_latency} | {std_throughput} | {std_tokens} | {std_mem} | {'-':<10} | {'-':<10} | {std_mode:<10} | {std_backend:<36}"
        )
        runs_summary.append(std_record)

        for variant_label, quantize, variant_key in variant_configs:
            run_variant = variant_active[variant_key] and (
                safe_dmoah is None or seq_len <= safe_dmoah or device.type != 'cuda')
            record = {
                "seq_len": seq_len,
                "model": f"dmoah_{variant_key}",
                "variant": variant_label,
                "quantized": quantize,
                "batch_size": batch_eff,
            }
            mode_setting = "auto"
            flux_alpha: float | None = None
            flux_backend = None
            last_stats: dict[str, object] | dict = {}
            token_keep = None
            target_k = None
            tokens_per_s = None
            latency = None
            throughput = None
            mem_use = None
            status = "skipped"
            backend_display = "dmoah-skip"
            active_k_display = "-"
            token_keep_display = "-"
            mode_display = mode_setting
            if run_variant:
                model_dmoah, config_dmoah, alpha_override = _instantiate_dmoah_model(
                    seq_len,
                    seq_high,
                    d_model=D_MODEL,
                    quantize=quantize,
                    memory_guard=args.memory_guard,
                )
                mode_setting = getattr(config_dmoah, "attn_mode", "auto")
                flux_alpha = alpha_override
                if flux_alpha is None:
                    seq_low_cfg = int(
                        getattr(config_dmoah, "attn_active_seq_low", 256) or 256)
                    switch_ctx_cfg = int(
                        getattr(config_dmoah, "attn_linear_switch_ctx", 8192) or 8192)
                    flux_alpha = _flux_alpha_from_seq(
                        seq_len=seq_len, seq_low=seq_low_cfg, switch_ctx=switch_ctx_cfg)

                model_dmoah.to(device)
                model_dmoah.eval()

                try:
                    latency = benchmark_forward_pass(model_dmoah, input_tensor)
                    throughput = (1000 / latency) * batch_eff
                    mem_use = get_peak_memory_mb(model_dmoah, input_tensor)
                    status = "ok"
                except RuntimeError as exc:
                    message = str(exc)
                    lower = message.lower()
                    if device.type == 'cuda' and ("out of memory" in lower or "cuda" in message or "hip" in lower):
                        variant_active[variant_key] = False
                        status = "limit"
                        latency = None
                        throughput = None
                        mem_use = None
                        if not dmoah_limit_announced[variant_key]:
                            prior = variant_last_success[variant_key]
                            info = prior if prior is not None else "<unknown>"
                            print(
                                f"[Notice] {variant_label} hit its limit at seq_len={seq_len} (last safe ~{info}).")
                            dmoah_limit_announced[variant_key] = True
                    else:
                        raise

                if status == "ok":
                    backend_info = get_last_backend_info()
                    backend_name = str(backend_info.get('name', 'unknown'))
                    details_raw = backend_info.get('details') or {}
                    details = dict(details_raw) if isinstance(
                        details_raw, dict) else {}
                    last_stats = getattr(
                        model_dmoah, 'last_head_stats', {}) or {}
                    sparse_state = getattr(
                        model_dmoah, '_last_sparse_state', {}) or {}
                    backend_override = sparse_state.get('backend')
                    if backend_override:
                        backend_name = str(backend_override)
                    if sparse_state:
                        if 'max_rows' not in details and 'max_rows' in sparse_state:
                            details['max_rows'] = sparse_state['max_rows']
                        if 'density' not in details and 'density' in sparse_state:
                            details['density'] = sparse_state['density']
                    max_rows = details.get('max_rows') if isinstance(
                        details, dict) else None
                    if max_rows is None:
                        max_rows = last_stats.get('max_active_rows')
                    target_k = last_stats.get(
                        'target_k') or last_stats.get('top_k')
                    density = last_stats.get('max_active_density')
                    unique_heads = last_stats.get('unique_heads')
                    quantized_flag = sparse_state.get(
                        'quantized') if sparse_state else quantize
                    dna_stats = last_stats.get('dna') if isinstance(
                        last_stats, dict) else None
                    token_keep = last_stats.get('token_keep_fraction') if isinstance(
                        last_stats, dict) else None
                    flux_alpha_val = last_stats.get("flux_alpha", flux_alpha)
                    parts: list[str] = []
                    if max_rows is not None:
                        parts.append(f"max_rows={int(max_rows)}")
                    if density is not None:
                        parts.append(f"density={density:.2f}")
                    if unique_heads is not None:
                        parts.append(f"unique={int(unique_heads)}")
                    if quantized_flag:
                        parts.append("int8")
                    if token_keep is not None:
                        parts.append(f"tokens={token_keep:.2f}")
                    if dna_stats:
                        blend_mean = dna_stats.get('blend_mean')
                        if blend_mean is not None:
                            parts.append(f"dna_blend={blend_mean:.2f}")
                        updated = dna_stats.get('updated_gates')
                        if updated is not None:
                            parts.append(f"dna_updates={updated}")
                    if flux_alpha_val is not None:
                        parts.append(f"alpha={float(flux_alpha_val):.2f}")
                    detail_suffix = f" ({', '.join(parts)})" if parts else ''
                    flux_backend = last_stats.get("linear_shortlist_backend")
                    backend_display = f"{backend_name}{detail_suffix}" if detail_suffix else backend_name
                    active_k_display = "-" if target_k is None else f"{int(target_k)}"
                    token_keep_display = "-" if token_keep is None else f"{token_keep:.2f}"
                    variant_last_success[variant_key] = seq_len
                    mode_display = str(last_stats.get('mode_selected') or last_stats.get(
                        'mode') or mode_setting or 'unknown')
                    tokens_per_s = throughput * seq_len if throughput is not None else None
                else:
                    backend_display = "dmoah-limit"
                    active_k_display = "-"
                    token_keep_display = "-"
                    token_keep = None
                    target_k = None
                    mode_display = mode_setting if run_variant else "-"
                    tokens_per_s = None
                del model_dmoah
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                flux_backend = None

            display_latency = f"{latency:<15.2f}" if latency is not None else f"{'OOM':<15}"
            display_throughput = f"{throughput:<12.2f}" if throughput is not None else f"{'-':<12}"
            display_tokens = f"{tokens_per_s:<15.2f}" if tokens_per_s is not None else f"{'-':<15}"
            display_memory = f"{mem_use:<15.2f}" if mem_use is not None else f"{'-':<15}"

            print(
                f"{seq_len:<10} | {variant_label:<24} | {display_latency} | "
                f"{display_throughput} | {display_tokens} | {display_memory} | {active_k_display:<10} | {token_keep_display:<10} | {mode_display:<10} | {backend_display:<36}"
            )
            record.update({
                "latency_ms": latency,
                "throughput_seq_s": throughput,
                "tokens_per_s": tokens_per_s,
                "memory_mb": mem_use,
                "active_k": target_k,
                "token_fraction": token_keep,
                "backend": backend_display,
                "flux_backend": flux_backend,
                "mode": mode_display,
                "status": status,
                "linear_L_effective": last_stats.get("linear_L_effective"),
                "linear_L_local_cap": last_stats.get("linear_L_local_cap"),
                "linear_L_anchor_cap": last_stats.get("linear_L_anchor_cap"),
                "linear_L_dna_cap": last_stats.get("linear_L_dna_cap"),
                "flux_alpha": None if flux_alpha is None else float(flux_alpha),
            })
            if status == "ok":
                auto_log = {
                    "model": record["model"],
                    "variant": variant_key,
                    "label": variant_label,
                    "seq_len": seq_len,
                    "batch_size": batch_eff,
                    "latency_ms": latency,
                    "tokens_per_s": tokens_per_s,
                    "throughput_seq_s": throughput,
                    "mode_setting": mode_setting,
                    "mode_selected": last_stats.get("mode_selected") or last_stats.get("mode") or mode_setting,
                    "backend": backend_name,
                    "flux_backend": flux_backend,
                    "token_fraction": token_keep,
                    "active_k": target_k,
                    "quantized": quantize,
                    "linear_L_effective": last_stats.get("linear_L_effective"),
                    "linear_L_local_cap": last_stats.get("linear_L_local_cap"),
                    "linear_L_anchor_cap": last_stats.get("linear_L_anchor_cap"),
                    "linear_L_dna_cap": last_stats.get("linear_L_dna_cap"),
                    "flux_alpha": float(flux_alpha),
                }
                auto_logs.append(auto_log)
                print(
                    f"[AutoLog] captured {variant_label} seq_len={seq_len} backend={backend_name}")
            else:
                if device.type == 'cuda' and not dmoah_limit_announced[variant_key]:
                    print(
                        f"[Notice] {variant_label} skipping seq_len={seq_len} due to limit.")
                    dmoah_limit_announced[variant_key] = True
                print(
                    f"{seq_len:<10} | {variant_label:<24} | {'OOM':<15} | {'-':<12} | {'-':<15} | {'-':<15} | {'-':<10} | {'-':<10} | {'-':<10} | {'dmoah-limit':<36}"
                )
                record.update({
                    "latency_ms": None,
                    "throughput_seq_s": None,
                    "tokens_per_s": None,
                    "memory_mb": None,
                    "active_k": None,
                    "token_fraction": None,
                    "backend": "dmoah-limit",
                    "mode": mode_setting if run_variant else None,
                    "status": "limit",
                })

            runs_summary.append(record)

        print("-" * 212)

    if device.type == 'cuda':
        std_msg = safe_std if safe_std is not None else "none"
        dmoah_msg = safe_dmoah if safe_dmoah is not None else "none"
        print(
            f"\n[Summary] Standard safe length: {std_msg}; DMoAH safe length: {dmoah_msg}.")
    else:
        std_msg = dmoah_msg = "cpu"

    std_last_safe = max((entry["seq_len"] for entry in runs_summary if entry.get(
        "model") == "standard" and entry.get("status") == "ok"), default=None)
    last_safe_variants = {
        key: max(
            (
                entry["seq_len"]
                for entry in runs_summary
                if entry.get("model") == f"dmoah_{key}" and entry.get("status") == "ok"
            ),
            default=None,
        )
        for _, _, key in variant_configs
    }

    safe_record = {
        "device": str(device),
        "safe_standard": std_msg,
        "safe_dmoah": dmoah_msg,
        "last_safe_standard": std_last_safe,
        "last_safe_variants": last_safe_variants,
        "sequence_lengths": list(sequence_lengths),
        "base_batch_size": BASE_BATCH_SIZE,
        "d_model": D_MODEL,
        "variant_labels": {key: label for label, _, key in variant_configs},
        "runs": runs_summary,
    }
    safe_record["auto_logs"] = auto_logs

    def _estimate_alpha(model_key: str) -> float | None:
        points: list[tuple[float, float]] = []
        for entry in runs_summary:
            if entry.get("model") != model_key:
                continue
            tokens_per_s = entry.get("tokens_per_s")
            seq_len = entry.get("seq_len")
            if tokens_per_s and tokens_per_s > 0 and seq_len and seq_len > 0:
                points.append((float(seq_len), float(tokens_per_s)))
        if len(points) < 2:
            return None
        points = points[-3:]
        xs = torch.tensor([math.log(p[0])
                          for p in points], dtype=torch.float64)
        ys = torch.tensor([math.log(p[1])
                          for p in points], dtype=torch.float64)
        denom = xs.var(unbiased=False)
        if denom <= 0:
            return None
        cov = ((xs - xs.mean()) * (ys - ys.mean())).mean()
        return float(cov / denom)

    safe_record["scaling_alpha"] = {
        "standard": _estimate_alpha("standard"),
        **{
            model: _estimate_alpha(model)
            for model in {entry["model"] for entry in runs_summary if entry["model"].startswith("dmoah_")}
        },
    }
    summary_path: Path | None = None
    if not args.no_save:
        if args.report_dir is not None:
            reports_dir = args.report_dir
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            summary_path = reports_dir / f"tinytoy_summary_{timestamp}.json"
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(safe_record, handle, indent=2)
                handle.write("\n")
        else:
            summary_path = save_summary_to_disk(safe_record)

    if summary_path is not None:
        try:
            relative_summary_path = summary_path.relative_to(PROJECT_ROOT)
        except ValueError:
            relative_summary_path = summary_path
        print(
            f"\n[Summary] Detailed JSON report saved to {relative_summary_path}")
    else:
        print("\n[Summary] JSON report saving disabled for this run.")
    if auto_logs:
        print(f"[Summary] Captured {len(auto_logs)} AutoLog entries.")


if __name__ == "__main__":
    main()
