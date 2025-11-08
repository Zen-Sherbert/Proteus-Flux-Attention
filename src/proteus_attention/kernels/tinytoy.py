# benchmark.py
import argparse
import json
import os
import sys
import time
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

# Ensure the repo's `src` tree is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proteus_attention.models.aspa import AdaptiveSparseAttention
from proteus_attention.kernels.sparse_attn import get_last_backend_info
from proteus_attention.tools.chunked_shortlist import ChunkedShortlistConfig, ChunkedShortlistRunner

# --- Import your custom modules ---

# Ensure CUDA allocator uses expandable segments to reduce fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class _ShortlistAutotuneGuard:
    """Context manager to toggle Triton autotune for Shortlist chunk runs only."""

    def __init__(self, disable: bool) -> None:
        self.disable = disable
        self._prev: Optional[str] = None

    def __enter__(self) -> None:
        if not self.disable:
            return
        self._prev = os.environ.get("PROTEUS_TUNE_DISABLE")
        os.environ["PROTEUS_TUNE_DISABLE"] = "1"

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.disable:
            return
        if self._prev is None:
            os.environ.pop("PROTEUS_TUNE_DISABLE", None)
        else:
            os.environ["PROTEUS_TUNE_DISABLE"] = self._prev

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
    if seq_len >= 524288:
        warmup_runs = 1
    elif seq_len >= 131072:
        warmup_runs = 1
    elif seq_len >= 65536:
        warmup_runs = 1
    elif seq_len >= 32768:
        warmup_runs = 1
    elif seq_len >= 16384:
        warmup_runs = 2
    elif seq_len >= 8192:
        warmup_runs = 5
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

    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    x = x.to(device)

    _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    return peak_memory_mb


DEFAULT_GPU_SEQUENCE_CAP = 2_000_000


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


def _maybe_generate_plot(
    runs_summary: list[dict[str, object]],
    *,
    plot_requested: bool,
    plot_path: Path | None,
    summary_path: Path | None,
    device_label: str,
) -> Path | None:
    if not plot_requested and plot_path is None:
        return None

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[Plot] matplotlib unavailable ({exc}); skipping plot.")
        return None

    series: dict[str, list[tuple[int, float]]] = {}
    for entry in runs_summary:
        if entry.get("status") != "ok":
            continue
        latency = entry.get("latency_ms")
        seq_len = entry.get("seq_len")
        if latency is None or seq_len is None:
            continue
        try:
            latency_val = float(latency)
            seq_val = int(seq_len)
        except (TypeError, ValueError):
            continue
        label = entry.get("variant")
        if isinstance(label, str):
            key = label
        else:
            key = "Standard Attention" if entry.get("model") == "standard" else str(entry.get("model"))
        series.setdefault(key, []).append((seq_val, latency_val))

    if not series:
        print("[Plot] No successful runs to plot.")
        return None

    target_path = plot_path
    if target_path is None:
        if summary_path is not None:
            target_path = summary_path.with_suffix(".png")
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            default_dir = PROJECT_ROOT / "reports" / "tinytoy"
            default_dir.mkdir(parents=True, exist_ok=True)
            target_path = default_dir / f"tinytoy_plot_{timestamp}.png"
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    max_x = 0
    min_x = None
    max_y = 0.0
    min_y = None
    for label, points in sorted(series.items()):
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        max_x = max(max_x, max(xs))
        min_x = min(xs) if min_x is None else min(min_x, min(xs))
        max_y = max(max_y, max(ys))
        min_y = min(ys) if min_y is None else min(min_y, min(ys))
        plt.plot(xs, ys, marker="o", label=label)

    if min_x and max_x and max_x > min_x * 2:
        plt.xscale("log", base=2)
    if min_y and max_y and max_y > min_y * 4:
        plt.yscale("log")

    plt.xlabel("Sequence Length")
    plt.ylabel("Latency (ms)")
    plt.title(f"Proteus TinyToy Latency — Device: {device_label}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved latency plot to {target_path.relative_to(PROJECT_ROOT) if target_path.is_relative_to(PROJECT_ROOT) else target_path}")
    return target_path


def build_aspa_config(
    seq_len: int,
    seq_high: int,
    *,
    d_model: int,
    quantize: bool,
    allow_sdpa_fastpath: bool,
) -> SimpleNamespace:
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
        use_sdpa=allow_sdpa_fastpath,
        attn_h_total=32,
        attn_h_active=attn_h_active,
        attn_h_active_min=max(2, attn_h_active // 2),
        attn_h_active_max=max(attn_h_active, 8),
        attn_active_seq_low=256,
        attn_active_seq_high=seq_high,
        attn_small_seq_dense=max(128, min(seq_len, 1024)) if allow_sdpa_fastpath else 0,
        n_ctx=seq_high,
        bias=False,
        attn_router_lambda=0.0,
        attn_router_beta=1.0,
        attn_router_reg_mode="entropy",
        attn_gates=64,
        attn_router_noise_std=0.0,
        attn_force_dense_threshold=attn_force_dense_threshold,
        attn_quantize_int8=quantize,
        attn_proto_enable=True,
        attn_proto_threshold=0.3,
        attn_proto_blend=0.6,
        attn_proto_temp=0.2,
        attn_proto_usage_boost=0.15,
        attn_proto_decay=0.97,
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


def _prepare_aspa_config(
    seq_len: int,
    seq_high: int,
    *,
    d_model: int,
    quantize: bool,
    memory_guard: bool,
    allow_sdpa_fastpath: bool,
) -> SimpleNamespace:
    config = build_aspa_config(
        seq_len,
        seq_high,
        d_model=d_model,
        quantize=quantize,
        allow_sdpa_fastpath=allow_sdpa_fastpath,
    )
    if memory_guard:
        config = _apply_memory_saving_overrides(config, seq_len)
    dense_cut = 0.35 if seq_len <= 2048 else 0.20
    setattr(config, "attn_dense_alpha_cutoff", dense_cut)
    return config


def _shortlist_alpha_from_seq(seq_len: int, *, seq_low: int, switch_ctx: int) -> float:
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


def _parse_mode_label(value: object) -> tuple[str, Optional[float]]:
    """
    Extract human-readable mode and optional CVT alpha from the telemetry string.
    """
    if value is None:
        return "-", None
    text = str(value).strip()
    if not text:
        return "-", None
    alpha: Optional[float] = None
    mode = text
    if "::" in text:
        prefix, mode = text.split("::", 1)
        mode = mode.strip() or mode
        prefix = prefix.strip()
        if prefix.startswith("cvt(") and "alpha=" in prefix:
            start = prefix.find("alpha=") + len("alpha=")
            end = prefix.find(")", start)
            snippet = prefix[start:end if end != -1 else None].strip()
            try:
                alpha = float(snippet)
            except (TypeError, ValueError):
                alpha = None
    return mode or "-", alpha


def _instantiate_aspa_model(
    seq_len: int,
    seq_high: int,
    *,
    d_model: int,
    quantize: bool,
    memory_guard: bool,
    allow_sdpa_fastpath: bool,
    device: torch.device | None = None,
) -> tuple[AdaptiveSparseAttention, SimpleNamespace, float | None]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _prepare_aspa_config(
        seq_len,
        seq_high,
        d_model=d_model,
        quantize=quantize,
        memory_guard=memory_guard,
        allow_sdpa_fastpath=allow_sdpa_fastpath and device.type == "cuda",
    )
    setattr(config, "attn_track_latency", False)
    model = AdaptiveSparseAttention(config)
    alpha: float | None = None
    if hasattr(model, "set_shortlist_alpha"):
        seq_low_cfg = int(getattr(config, "attn_active_seq_low", 256) or 256)
        switch_ctx_cfg = int(
            getattr(config, "attn_linear_switch_ctx", 8192) or 8192)
        alpha = _shortlist_alpha_from_seq(
            seq_len=seq_len,
            seq_low=seq_low_cfg,
            switch_ctx=switch_ctx_cfg,
        )
        model.set_shortlist_alpha(alpha)
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
    allow_sdpa_fastpath: bool,
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
    config = _prepare_aspa_config(
        prime_len,
        max(seq_high, prime_len),
        d_model=d_model,
        quantize=quantize,
        memory_guard=memory_guard,
        allow_sdpa_fastpath=allow_sdpa_fastpath,
    )
    setattr(config, "attn_track_latency", False)
    model = AdaptiveSparseAttention(config)
    if hasattr(model, "set_shortlist_alpha"):
        model.set_shortlist_alpha(1.0)
    model.to(device)
    input_tensor = torch.randn(
        1, prime_len, d_model, device=device, dtype=torch.get_default_dtype())
    with torch.inference_mode():
        _ = model(input_tensor)
    torch.cuda.synchronize()
    del model
    _LINEAR_WARM_STATE.add(key)


def _prewarm_shortlist_kernels(
    *,
    device: torch.device,
    sequence_lengths: list[int],
    seq_high: int,
    d_model: int,
    base_batch_size: int,
    variant_configs: list[tuple[str, bool, str]],
    memory_guard: bool,
    allow_sdpa_fastpath: bool,
) -> None:
    """
    Trigger Triton JIT/autotune for every (seq_len, variant) combination before timing.
    """
    if device.type != "cuda":
        return

    for _, quantize, _ in variant_configs:
        for seq_len in sequence_lengths:
            batch_eff = _effective_batch_size(seq_len, base_batch_size)
            input_tensor = torch.randn(
                batch_eff,
                seq_len,
                d_model,
                device=device,
                dtype=torch.get_default_dtype(),
            )
            config = _prepare_aspa_config(
                seq_len,
                seq_high,
                d_model=d_model,
                quantize=quantize,
                memory_guard=memory_guard,
                allow_sdpa_fastpath=allow_sdpa_fastpath,
            )
            setattr(config, "attn_track_latency", False)
            model = AdaptiveSparseAttention(config)
            if hasattr(model, "set_shortlist_alpha"):
                seq_low_cfg = int(getattr(config, "attn_active_seq_low", 256) or 256)
                switch_ctx_cfg = int(getattr(config, "attn_linear_switch_ctx", 8192) or 8192)
                alpha = _shortlist_alpha_from_seq(
                    seq_len=seq_len,
                    seq_low=seq_low_cfg,
                    switch_ctx=switch_ctx_cfg,
                )
                model.set_shortlist_alpha(alpha)
            model.to(device)
            with torch.inference_mode():
                _ = model(input_tensor)
            del model
            torch.cuda.synchronize(device)

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
        tensor_device = device if device.type == "cuda" else torch.device("cpu")
        input_tensor = torch.randn(
            batch_eff,
            seq_len,
            d_model,
            device=tensor_device,
            dtype=torch.get_default_dtype(),
        )
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
                break
            raise
        finally:
            del model

    if safe is not None:
        print(
            f"[Probe] {label} safe up to sequence length {safe} on this device.")
    else:
        print(f"[Probe] {label} did not complete any probed lengths safely.")
    return safe


def estimate_shortlist_chunk_safe_length(
    *,
    sequence_lengths: list[int],
    base_batch_size: int,
    d_model: int,
    device: torch.device,
    shortlist_args: argparse.Namespace,
) -> int | None:
    if not shortlist_args.enable_shortlist_chunk:
        return None

    safe: int | None = None
    for seq_len in sequence_lengths:
        batch_eff = _effective_batch_size(seq_len, base_batch_size)
        if batch_eff != 1:
            continue
        buffer_tokens = max(1, int(seq_len * shortlist_args.shortlist_chunk_buffer_ratio))
        shortlist_config = ChunkedShortlistConfig(
            seq_len=seq_len,
            d_model=d_model,
            chunk_len=max(1, shortlist_args.shortlist_chunk_len),
            buffer_tokens=buffer_tokens,
            per_chunk_budget=max(1, shortlist_args.shortlist_chunk_budget),
            device=device,
            heads=8,
            chunk_sparse_ratio=max(1e-4, shortlist_args.shortlist_chunk_sparse_ratio),
            final_sparse_ratio=max(1e-4, shortlist_args.shortlist_chunk_final_ratio),
            shortlist_alpha=shortlist_args.shortlist_chunk_alpha,
            seed=123,
            report_latency=False,
            progress=False,
            run_final_pass=True,
        )
        sequence = torch.randn(1, seq_len, d_model)
        print(f"[Probe] Chunked Shortlist testing seq_len={seq_len}", flush=True)
        guard = _ShortlistAutotuneGuard(disable=(device.type == "cuda"))
        with guard:
            runner = ChunkedShortlistRunner(shortlist_config)
            try:
                runner.run(sequence=sequence)
                safe = seq_len
            except RuntimeError as exc:
                message = str(exc)
                lower = message.lower()
                if device.type == "cuda" and ("out of memory" in lower or "cuda" in message or "hip" in lower):
                    print(f"[Probe] Chunked Shortlist failed at sequence length {seq_len}: {message.splitlines()[-1]}")
                    break
                raise
            finally:
                del runner
    if safe is not None:
        print(f"[Probe] Chunked Shortlist safe up to sequence length {safe} on this device.")
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

    def forward(self, x):
        # MultiheadAttention expects (query, key, value)
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        return attn_output

# --- Main Benchmark Execution ---


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ASPA (Dynamic Mixture-of-Attention-Heads) kernels vs. standard attention."
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
    parser.add_argument(
        "--bruteforce-blocks",
        action="store_true",
        help="Enable exhaustive block-size autotuning (PROTEUS_TUNE_BRUTE_FORCE=1).",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="store_true",
        default=True,
        help="Generate a latency plot (default; requires matplotlib).",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Disable latency plot generation.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional path for the latency plot image (implies --plot).",
    )
    parser.add_argument(
        "--enable-shortlist-chunk",
        action="store_true",
        help="Enable Chunked Shortlist pipeline for extreme sequence lengths.",
    )
    parser.add_argument(
        "--shortlist-chunk-threshold",
        type=int,
        default=1_000_000,
        help="Minimum sequence length that triggers Chunked Shortlist when enabled.",
    )
    parser.add_argument(
        "--shortlist-chunk-len",
        type=int,
        default=65_536,
        help="Streaming chunk length used by Chunked Shortlist (capped at 65k tokens to avoid ROCm compiler asserts).",
    )
    parser.add_argument(
        "--shortlist-chunk-buffer-ratio",
        type=float,
        default=0.05,
        help="Fraction of tokens retained after Chunked Shortlist streaming (buffer size = ratio * seq_len).",
    )
    parser.add_argument(
        "--shortlist-chunk-budget",
        type=int,
        default=4_096,
        help="Per-chunk promotion budget for Chunked Shortlist streaming.",
    )
    parser.add_argument(
        "--shortlist-chunk-sparse-ratio",
        type=float,
        default=0.05,
        help="Sparse keep ratio used during Chunked Shortlist streaming stage.",
    )
    parser.add_argument(
        "--shortlist-chunk-final-ratio",
        type=float,
        default=0.5,
        help="Sparse keep ratio used during Chunked Shortlist final pass.",
    )
    parser.add_argument(
        "--shortlist-chunk-alpha",
        type=float,
        default=1.0,
        help="Shortlist alpha slider (0=dense, 1=fully sparse linear shortlist).",
    )
    parser.add_argument(
        "--shortlist-chunk-report-latency",
        action="store_true",
        help="Capture CUDA latency metrics inside the Chunked Shortlist pipeline.",
    )
    parser.add_argument(
        "--shortlist-chunk-storage",
        choices=["cpu", "disk"],
        default="cpu",
        help="Stage chunk data in system RAM ('cpu') or spill to temporary disk ('disk').",
    )
    parser.add_argument(
        "--shortlist-chunk-temp-dir",
        type=Path,
        default=None,
        help="Optional directory to use when --shortlist-chunk-storage=disk (defaults to system tmp).",
    )
    parser.add_argument(
        "--allow-sdpa-fastpath",
        action="store_true",
        help="Permit PyTorch SDPA fast-path for very short sequences (defaults to disabled so Shortlist kernels are benchmarked).",
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

    chunked_shortlist_enabled = args.enable_shortlist_chunk
    if chunked_shortlist_enabled:
        if args.shortlist_chunk_threshold < 1:
            raise ValueError("shortlist-chunk-threshold must be a positive integer.")
        if args.shortlist_chunk_len <= 0:
            raise ValueError("shortlist-chunk-len must be positive.")
        if args.shortlist_chunk_buffer_ratio <= 0.0:
            raise ValueError("shortlist-chunk-buffer-ratio must be positive.")
        if args.shortlist_chunk_budget <= 0:
            raise ValueError("shortlist-chunk-budget must be positive.")
        if args.shortlist_chunk_sparse_ratio <= 0.0:
            raise ValueError("shortlist-chunk-sparse-ratio must be positive.")
        if args.shortlist_chunk_final_ratio <= 0.0:
            raise ValueError("shortlist-chunk-final-ratio must be positive.")
        if not (0.0 <= args.shortlist_chunk_alpha <= 1.0):
            raise ValueError("shortlist-chunk-alpha must lie within [0, 1].")

    if args.bruteforce_blocks:
        os.environ["PROTEUS_TUNE_BRUTE_FORCE"] = "1"
        from proteus_attention.kernels import sparse_attn as _sparse_module

        _sparse_module._BRUTE_FORCE_ENABLED = True
        _sparse_module._BRUTE_FORCE_WARNED = False

    if args.max_seq_len is not None and args.max_seq_len < 1:
        raise ValueError("max-seq-len must be a positive integer.")
    if args.max_seq_count is not None and args.max_seq_count < 1:
        raise ValueError("max-seq-count must be a positive integer.")

    device = resolve_device_choice(args.device)
    allow_sdpa_fastpath = bool(args.allow_sdpa_fastpath and device.type == "cuda")
    if device.type == "cuda" and not allow_sdpa_fastpath:
        print("[Config] SDPA fast-path disabled; benchmarking Shortlist kernels only.")

    if args.cuda_bf16 and args.cuda_fp32:
        raise ValueError("Specify at most one of --cuda-bf16 or --cuda-fp32.")

    use_bf16 = False
    if device.type == "cuda":
        use_bf16 = not args.cuda_fp32
        if args.cuda_bf16:
            use_bf16 = True
        torch.set_default_dtype(torch.bfloat16 if use_bf16 else torch.float32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except AttributeError:
            pass

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
    safe_chunked_shortlist = None
    if device.type == 'cuda' and probe_lengths:
        safe_std = None
        safe_aspa = None
        if not args.disable_standard:
            safe_std = estimate_safe_length(
                label="Standard",
                sequence_lengths=probe_lengths,
                batch_size=BASE_BATCH_SIZE,
                d_model=D_MODEL,
                model_factory=lambda _seq: StandardAttention(config_standard),
            )

        def _factory(seq: int) -> AdaptiveSparseAttention:
            model, _, _ = _instantiate_aspa_model(
                seq,
                seq_high,
                d_model=D_MODEL,
                quantize=False,
                memory_guard=args.memory_guard,
                allow_sdpa_fastpath=allow_sdpa_fastpath,
                device=device,
            )
            return model

        safe_aspa = estimate_safe_length(
            label="ASPA",
            sequence_lengths=probe_lengths,
            batch_size=BASE_BATCH_SIZE,
            d_model=D_MODEL,
            model_factory=_factory,
        )
        if chunked_shortlist_enabled:
            safe_chunked_shortlist = estimate_shortlist_chunk_safe_length(
                sequence_lengths=probe_lengths,
                base_batch_size=BASE_BATCH_SIZE,
                d_model=D_MODEL,
                device=device,
                shortlist_args=args,
            )
    else:
        safe_std = max(sequence_lengths) if not args.disable_standard else None
        safe_aspa = max(sequence_lengths)
        if chunked_shortlist_enabled:
            safe_chunked_shortlist = max(sequence_lengths)

    if not chunked_shortlist_enabled:
        sequence_lengths = [seq for seq in sequence_lengths if seq <= 524288]

    runs_summary: list[dict[str, object]] = []

    header = (
        f"{'Seq Len':<10} | {'Model':<24} | {'Latency (ms)':<15} | {'Seq/s':<12} | "
        f"{'Tok/s':<15} | {'Memory (MB)':<15} | {'Active K':<10} | {'Tokens':<10} | "
        f"{'Alpha':<7} | {'Mode':<16} | {'Backend':<36}"
    )
    print(header)
    print("-" * len(header))

    std_limit_announced = False
    last_std_success = None
    dtype_label = "BF16" if use_bf16 else "FP32"
    dtype_key = "bf16" if use_bf16 else "fp32"
    variant_configs = [
        (f"ASPA ({dtype_label})", False, dtype_key),
        ("ASPA (INT8)", True, "int8"),
    ]
    if args.disable_int8:
        variant_configs = [entry for entry in variant_configs if not entry[1]]
    linear_trigger = min(
        (seq for seq in sequence_lengths if seq >= 8192), default=None)
    if device.type == 'cuda' and linear_trigger is not None:
        print(
            f"[Warmup] Priming shortlist kernel at seq_len={linear_trigger}...", flush=True)
        for _, quantize, _ in variant_configs:
            _prime_linear_mode(
                device=device,
                seq_target=linear_trigger,
                seq_high=seq_high,
                d_model=D_MODEL,
                quantize=quantize,
                memory_guard=args.memory_guard,
                allow_sdpa_fastpath=allow_sdpa_fastpath,
            )
    if device.type == 'cuda':
        _prewarm_shortlist_kernels(
            device=device,
            sequence_lengths=sequence_lengths,
            seq_high=seq_high,
            d_model=D_MODEL,
            base_batch_size=BASE_BATCH_SIZE,
            variant_configs=variant_configs,
            memory_guard=args.memory_guard,
            allow_sdpa_fastpath=allow_sdpa_fastpath,
        )
    aspa_limit_announced = {key: False for _, _, key in variant_configs}
    variant_active = {key: True for _, _, key in variant_configs}
    variant_last_success = {key: None for _, _, key in variant_configs}
    chunked_shortlist_active = True
    chunked_shortlist_limit_announced = False
    chunked_shortlist_last_success: int | None = None
    auto_logs: list[dict[str, object]] = []
    input_cache: dict[tuple[int, int, torch.dtype, str], torch.Tensor] = {}

    def _get_input(batch: int, length: int) -> torch.Tensor:
        current_dtype = torch.get_default_dtype()
        device_label = device.type if device.type != "cuda" else f"{device.type}:{device.index or 0}"
        key = (batch, length, current_dtype, device_label)
        cached = input_cache.get(key)
        if cached is None:
            tensor = torch.randn(
                batch,
                length,
                D_MODEL,
                device=device if device.type == "cuda" else torch.device("cpu"),
                dtype=current_dtype,
            )
            input_cache[key] = tensor
            return tensor
        return cached

    for seq_len in sequence_lengths:
        batch_eff = _effective_batch_size(seq_len, BASE_BATCH_SIZE)
        input_tensor = _get_input(batch_eff, seq_len)

        run_standard = (not args.disable_standard) and (
            safe_std is None or seq_len <= safe_std or device.type != 'cuda')
        std_record = {
            "seq_len": seq_len,
            "model": "standard",
            "batch_size": batch_eff,
        }
        baseline_success = False
        std_alpha = "-"
        if run_standard:
            model_std = StandardAttention(config_standard)
            latency_std = benchmark_forward_pass(model_std, input_tensor)
            throughput_std = (1000 / latency_std) * batch_eff
            mem_std = get_peak_memory_mb(model_std, input_tensor)
            del model_std
            tokens_std = throughput_std * seq_len
            std_latency = f"{latency_std:<15.2f}"
            std_throughput = f"{throughput_std:<12.2f}"
            std_mem = f"{mem_std:<15.2f}"
            std_backend = "torch.nn"
            std_mode = "-"
            last_std_success = seq_len
            baseline_success = True
            std_record.update({
                "latency_ms": latency_std,
                "throughput_seq_s": throughput_std,
                "tokens_per_s": tokens_std,
                "memory_mb": mem_std,
                "status": "ok",
                "mode": "-",
                "shortlist_alpha": None,
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
                "mode": "-",
                "shortlist_alpha": None,
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
            f"{seq_len:<10} | {'Standard Attention':<24} | {std_latency} | "
            f"{std_throughput} | {std_tokens} | {std_mem} | {'-':<10} | {'-':<10} | "
            f"{std_alpha:<7} | {std_mode:<16} | {std_backend:<36}"
        )
        runs_summary.append(std_record)

        variant_status_map: dict[str, str] = {}
        for variant_label, quantize, variant_key in variant_configs:
            run_variant = variant_active[variant_key] and (
                safe_aspa is None or seq_len <= safe_aspa or device.type != 'cuda')
            record = {
                "seq_len": seq_len,
                "model": f"aspa_{variant_key}",
                "variant": variant_label,
                "quantized": quantize,
                "batch_size": batch_eff,
            }
            mode_setting = "auto"
            shortlist_alpha: float | None = None
            shortlist_backend = None
            last_stats: dict[str, object] | dict = {}
            token_keep = None
            target_k = None
            tokens_per_s = None
            latency = None
            throughput = None
            mem_use = None
            status = "skipped"
            backend_display = "aspa-skip"
            active_k_display = "-"
            token_keep_display = "-"
            alpha_display = "-"
            mode_display = "-"
            alpha_value: float | None = None
            if run_variant:
                model_aspa, config_aspa, alpha_override = _instantiate_aspa_model(
                    seq_len,
                    seq_high,
                    d_model=D_MODEL,
                    quantize=quantize,
                    memory_guard=args.memory_guard,
                    allow_sdpa_fastpath=allow_sdpa_fastpath,
                    device=device,
                )
                mode_setting = getattr(config_aspa, "attn_mode", "auto")
                mode_display = mode_setting
                shortlist_alpha = alpha_override
                if shortlist_alpha is None:
                    seq_low_cfg = int(
                        getattr(config_aspa, "attn_active_seq_low", 256) or 256)
                    switch_ctx_cfg = int(
                        getattr(config_aspa, "attn_linear_switch_ctx", 8192) or 8192)
                    shortlist_alpha = _shortlist_alpha_from_seq(
                        seq_len=seq_len, seq_low=seq_low_cfg, switch_ctx=switch_ctx_cfg)

                model_aspa.to(device)
                model_aspa.eval()

                try:
                    latency = benchmark_forward_pass(model_aspa, input_tensor)
                    throughput = (1000 / latency) * batch_eff
                    mem_use = get_peak_memory_mb(model_aspa, input_tensor)
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
                        if not aspa_limit_announced[variant_key]:
                            prior = variant_last_success[variant_key]
                            info = prior if prior is not None else "<unknown>"
                            print(
                                f"[Notice] {variant_label} hit its limit at seq_len={seq_len} (last safe ~{info}).")
                            aspa_limit_announced[variant_key] = True
                    else:
                        raise
                shortlist_alpha_val = shortlist_alpha
                if status == "ok":
                    backend_info = get_last_backend_info()
                    backend_name = str(backend_info.get('name', 'unknown'))
                    details_raw = backend_info.get('details') or {}
                    details = dict(details_raw) if isinstance(
                        details_raw, dict) else {}
                    last_stats = getattr(
                        model_aspa, 'last_head_stats', {}) or {}
                    sparse_state = getattr(
                        model_aspa, '_last_sparse_state', {}) or {}
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
                    proto_stats = None
                    if isinstance(last_stats, dict):
                        proto_stats = last_stats.get('proto') or last_stats.get('dna')
                    token_keep = last_stats.get('token_keep_fraction') if isinstance(
                        last_stats, dict) else None
                    shortlist_alpha_val = last_stats.get("shortlist_alpha", shortlist_alpha_val)
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
                    if proto_stats:
                        blend_mean = proto_stats.get('blend_mean')
                        if blend_mean is not None:
                            parts.append(f"proto_blend={blend_mean:.2f}")
                        updated = proto_stats.get('updated_gates')
                        if updated is not None:
                            parts.append(f"proto_updates={updated}")
                    if shortlist_alpha_val is not None:
                        parts.append(f"alpha={float(shortlist_alpha_val):.2f}")
                    detail_suffix = f" ({', '.join(parts)})" if parts else ''
                    shortlist_backend = last_stats.get("linear_shortlist_backend")
                    backend_display = f"{backend_name}{detail_suffix}" if detail_suffix else backend_name
                    active_k_display = "-" if target_k is None else f"{int(target_k)}"
                    token_keep_display = "-" if token_keep is None else f"{token_keep:.2f}"
                    variant_last_success[variant_key] = seq_len
                    mode_raw = last_stats.get('mode_selected') or last_stats.get(
                        'mode') or mode_setting or 'unknown'
                    mode_display, alpha_from_mode = _parse_mode_label(mode_raw)
                    if shortlist_alpha_val is None:
                        shortlist_alpha_val = alpha_from_mode
                    alpha_value = shortlist_alpha_val
                    tokens_per_s = throughput * seq_len if throughput is not None else None
                else:
                    backend_display = "aspa-limit"
                    active_k_display = "-"
                    token_keep_display = "-"
                    token_keep = None
                    target_k = None
                    alpha_value = shortlist_alpha
                    tokens_per_s = None
                del model_aspa
            else:
                shortlist_backend = None
                alpha_value = None

            alpha_display = "-" if alpha_value is None else f"{alpha_value:.2f}"

            display_latency = f"{latency:<15.2f}" if latency is not None else f"{'OOM':<15}"
            display_throughput = f"{throughput:<12.2f}" if throughput is not None else f"{'-':<12}"
            display_tokens = f"{tokens_per_s:<15.2f}" if tokens_per_s is not None else f"{'-':<15}"
            display_memory = f"{mem_use:<15.2f}" if mem_use is not None else f"{'-':<15}"

            print(
                f"{seq_len:<10} | {variant_label:<24} | {display_latency} | "
                f"{display_throughput} | {display_tokens} | {display_memory} | {active_k_display:<10} | {token_keep_display:<10} | "
                f"{alpha_display:<7} | {mode_display:<16} | {backend_display:<36}"
            )
            record.update({
                "latency_ms": latency,
                "throughput_seq_s": throughput,
                "tokens_per_s": tokens_per_s,
                "memory_mb": mem_use,
                "active_k": target_k,
                "token_fraction": token_keep,
                "backend": backend_display,
                "shortlist_backend": shortlist_backend,
                "mode": mode_display,
                "status": status,
                "linear_L_effective": last_stats.get("linear_L_effective"),
                "linear_L_local_cap": last_stats.get("linear_L_local_cap"),
                "linear_L_anchor_cap": last_stats.get("linear_L_anchor_cap"),
                "linear_L_proto_cap": last_stats.get("linear_L_proto_cap")
                or last_stats.get("linear_L_dna_cap"),
                "shortlist_alpha": None if alpha_value is None else float(alpha_value),
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
                    "shortlist_backend": shortlist_backend,
                    "token_fraction": token_keep,
                    "active_k": target_k,
                    "quantized": quantize,
                    "linear_L_effective": last_stats.get("linear_L_effective"),
                    "linear_L_local_cap": last_stats.get("linear_L_local_cap"),
                    "linear_L_anchor_cap": last_stats.get("linear_L_anchor_cap"),
                    "linear_L_proto_cap": last_stats.get("linear_L_proto_cap")
                    or last_stats.get("linear_L_dna_cap"),
                    "shortlist_alpha": None if alpha_value is None else float(alpha_value),
                }
                auto_logs.append(auto_log)
                print(
                    f"[AutoLog] captured {variant_label} seq_len={seq_len} backend={backend_name}")
            else:
                if device.type == 'cuda' and not aspa_limit_announced[variant_key]:
                    print(
                        f"[Notice] {variant_label} skipping seq_len={seq_len} due to limit.")
                    aspa_limit_announced[variant_key] = True
                print(
                    f"{seq_len:<10} | {variant_label:<24} | {'OOM':<15} | {'-':<12} | {'-':<15} | {'-':<15} | "
                    f"{'-':<10} | {'-':<10} | {'-':<7} | {'-':<16} | {'aspa-limit':<36}"
                )
                record.update({
                    "latency_ms": None,
                    "throughput_seq_s": None,
                    "tokens_per_s": None,
                    "memory_mb": None,
                    "active_k": None,
                    "token_fraction": None,
                    "backend": "aspa-limit",
                    "mode": "-" if run_variant else None,
                    "status": "limit",
                    "shortlist_alpha": None,
                })

            variant_status_map[variant_key] = status
            runs_summary.append(record)

        if chunked_shortlist_enabled:
            has_aspa_success = any(
                status == "ok" for status in variant_status_map.values()
            )
            should_run_chunked_shortlist = chunked_shortlist_active and (
                seq_len >= args.shortlist_chunk_threshold
                or not has_aspa_success
            )
            if should_run_chunked_shortlist:
                buffer_tokens = max(1, int(seq_len * args.shortlist_chunk_buffer_ratio))
                shortlist_config = ChunkedShortlistConfig(
                    seq_len=seq_len,
                    d_model=D_MODEL,
                    chunk_len=args.shortlist_chunk_len,
                    buffer_tokens=buffer_tokens,
                    per_chunk_budget=args.shortlist_chunk_budget,
                    device=device,
                    heads=config_standard.n_head,
                    chunk_sparse_ratio=args.shortlist_chunk_sparse_ratio,
                    final_sparse_ratio=args.shortlist_chunk_final_ratio,
                    shortlist_alpha=args.shortlist_chunk_alpha,
                    seed=None,
                    report_latency=args.shortlist_chunk_report_latency and device.type == "cuda",
                    progress=False,
                    run_final_pass=True,
                    storage=args.shortlist_chunk_storage,
                    temp_dir=args.shortlist_chunk_temp_dir,
                )
                record_shortlist = {
                    "seq_len": seq_len,
                    "model": "aspa_chunked_shortlist",
                    "variant": f"Chunked Shortlist ({dtype_label})",
                    "quantized": False,
                    "batch_size": batch_eff,
                    "shortlist_alpha": args.shortlist_chunk_alpha,
                }
                sequence_cpu = input_tensor[:1].detach().to("cpu", dtype=torch.float32).contiguous()
                guard = _ShortlistAutotuneGuard(disable=(device.type == "cuda"))
                try:
                    with guard:
                        shortlist_runner = ChunkedShortlistRunner(shortlist_config)
                        shortlist_result = shortlist_runner.run(sequence=sequence_cpu)
                except RuntimeError as exc:
                    message = str(exc)
                    lower = message.lower()
                    if device.type == 'cuda' and ("out of memory" in lower or "cuda" in message or "hip" in lower):
                        chunked_shortlist_active = False
                        record_shortlist.update({
                            "latency_ms": None,
                            "throughput_seq_s": None,
                            "tokens_per_s": None,
                            "memory_mb": None,
                            "status": "limit",
                        })
                        info = chunked_shortlist_last_success if chunked_shortlist_last_success is not None else "<unknown>"
                        if not chunked_shortlist_limit_announced:
                            print(f"[Notice] Chunked Shortlist hit its limit at seq_len={seq_len} (last safe ~{info}).")
                            chunked_shortlist_limit_announced = True
                        print(
                            f"{seq_len:<10} | {'ASPA + Chunked Shortlist':<24} | {'OOM':<15} | {'-':<12} | {'-':<15} | {'-':<15} | "
                            f"{'-':<10} | {'-':<10} | {'-':<7} | {'-':<16} | {'chunked_shortlist-limit':<36}"
                        )
                    else:
                        raise
                else:
                    metrics = shortlist_result.metrics
                    latency = metrics.total_time_ms or 0.0
                    throughput = (1000 / latency) * batch_eff if latency > 0 else None
                    tokens_per_s = throughput * seq_len if throughput is not None else None
                    mem_use = metrics.peak_memory_mb
                    retention = metrics.retention_ratio
                    final_stats = shortlist_result.final_stats or {}
                    backend_info = shortlist_result.backend_info or {}
                    backend_name = str(backend_info.get("name", "chunked_shortlist"))
                    detail_parts: list[str] = []
                    if metrics.chunk_time_ms is not None:
                        detail_parts.append(f"chunk={metrics.chunk_time_ms:.1f}ms")
                    if metrics.final_time_ms is not None:
                        detail_parts.append(f"final={metrics.final_time_ms:.1f}ms")
                    if retention is not None:
                        detail_parts.append(f"retain={retention:.4f}")
                    if metrics.used_fallback:
                        detail_parts.append("fallback")
                    backend_display = f"chunked_shortlist/{backend_name}"
                    if detail_parts:
                        backend_display = f"{backend_display} ({', '.join(detail_parts)})"
                    target_k = None
                    if isinstance(final_stats, dict):
                        target_k = final_stats.get("target_k") or final_stats.get("top_k")
                    active_k_display = "-" if target_k is None else f"{int(target_k)}"
                    token_keep_display = f"{retention:.2f}" if retention is not None else "-"
                    mode_raw = final_stats.get("mode_selected") or final_stats.get("mode") or "chunked_shortlist"
                    mode_display, alpha_from_mode = _parse_mode_label(mode_raw)
                    alpha_value = alpha_from_mode
                    if alpha_value is None:
                        alpha_value = float(args.shortlist_chunk_alpha) if args.shortlist_chunk_alpha is not None else None
                    alpha_display = "-" if alpha_value is None else f"{alpha_value:.2f}"
                    display_latency = f"{latency:<15.2f}" if latency else f"{'-':<15}"
                    display_throughput = f"{throughput:<12.2f}" if throughput is not None else f"{'-':<12}"
                    display_tokens = f"{tokens_per_s:<15.2f}" if tokens_per_s is not None else f"{'-':<15}"
                    display_memory = f"{mem_use:<15.2f}" if mem_use is not None else f"{'-':<15}"
                    backend_display_print = backend_display
                    if len(backend_display_print) > 36:
                        backend_display_print = backend_display_print[:33] + "..."
                    print(
                        f"{seq_len:<10} | {'ASPA + Chunked Shortlist':<24} | {display_latency} | {display_throughput} | {display_tokens} | "
                        f"{display_memory} | {active_k_display:<10} | {token_keep_display:<10} | {alpha_display:<7} | {mode_display:<16} | {backend_display_print:<36}"
                    )
                    record_shortlist.update({
                        "latency_ms": latency,
                        "throughput_seq_s": throughput,
                        "tokens_per_s": tokens_per_s,
                        "memory_mb": mem_use,
                        "active_k": target_k,
                        "token_fraction": retention,
                        "backend": backend_display,
                        "chunked_shortlist_backend": backend_name,
                        "mode": mode_display,
                        "status": "ok",
                        "shortlist_alpha": alpha_value,
                        "chunked_shortlist_chunk_time_ms": metrics.chunk_time_ms,
                        "chunked_shortlist_final_time_ms": metrics.final_time_ms,
                        "chunked_shortlist_retained_tokens": metrics.retained_tokens,
                    })
                    auto_logs.append({
                        "model": record_shortlist["model"],
                        "variant": "chunked_shortlist",
                        "label": f"Chunked Shortlist ({dtype_label})",
                        "seq_len": seq_len,
                        "batch_size": batch_eff,
                        "latency_ms": latency,
                        "tokens_per_s": tokens_per_s,
                        "throughput_seq_s": throughput,
                        "mode_selected": mode_display,
                        "backend": backend_name,
                        "backend_info": backend_info,
                        "token_fraction": retention,
                        "active_k": target_k,
                        "chunk_time_ms": metrics.chunk_time_ms,
                        "final_time_ms": metrics.final_time_ms,
                        "retained_tokens": metrics.retained_tokens,
                        "used_fallback": metrics.used_fallback,
                        "shortlist_alpha": alpha_value,
                    })
                    chunked_shortlist_last_success = seq_len
                    del shortlist_result
                    del shortlist_runner
                runs_summary.append(record_shortlist)

        print("-" * len(header))

    if device.type == 'cuda':
        std_msg = safe_std if safe_std is not None else "none"
        aspa_msg = safe_aspa if safe_aspa is not None else "none"
        shortlist_msg = safe_chunked_shortlist if safe_chunked_shortlist is not None else ("disabled" if not chunked_shortlist_enabled else "none")
        print(
            f"\n[Summary] Standard safe length: {std_msg}; ASPA safe length: {aspa_msg}; Chunked Shortlist safe length: {shortlist_msg}.")
    else:
        std_msg = aspa_msg = "cpu"
        shortlist_msg = "cpu" if chunked_shortlist_enabled else "disabled"

    std_last_safe = max((entry["seq_len"] for entry in runs_summary if entry.get(
        "model") == "standard" and entry.get("status") == "ok"), default=None)
    last_safe_variants = {
        key: max(
            (
                entry["seq_len"]
                for entry in runs_summary
                if entry.get("model") == f"aspa_{key}" and entry.get("status") == "ok"
            ),
            default=None,
        )
        for _, _, key in variant_configs
    }

    safe_record = {
        "device": str(device),
        "safe_standard": std_msg,
        "safe_aspa": aspa_msg,
        "safe_chunked_shortlist": shortlist_msg,
        "last_safe_standard": std_last_safe,
        "last_safe_variants": last_safe_variants,
        "last_safe_chunked_shortlist": chunked_shortlist_last_success,
        "sequence_lengths": list(sequence_lengths),
        "base_batch_size": BASE_BATCH_SIZE,
        "d_model": D_MODEL,
        "variant_labels": {key: label for label, _, key in variant_configs},
        "runs": runs_summary,
    }
    if chunked_shortlist_enabled:
        safe_record["variant_labels"]["chunked_shortlist"] = f"Chunked Shortlist ({dtype_label})"
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
            for model in {entry["model"] for entry in runs_summary if entry["model"].startswith("aspa_")}
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

    plot_requested = args.plot or args.plot_path is not None
    plot_generated: Path | None = None
    if plot_requested:
        plot_generated = _maybe_generate_plot(
            runs_summary,
            plot_requested=plot_requested,
            plot_path=args.plot_path,
            summary_path=summary_path,
            device_label=str(device),
        )

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
    if plot_generated is None and plot_requested:
        print("[Plot] Plot generation skipped.")


if __name__ == "__main__":
    main()
