"""
Reusable Chunked Flux pipeline utilities.

This module promotes the previous demo script to a production-oriented tool:

* Encapsulates configuration via dataclasses.
* Supports optional external sequences (e.g. pre-computed embeddings) in
  addition to synthetic random tensors.
* Provides structured metrics and chunk-level diagnostics for downstream use.
* Handles both CUDA/ROCm and CPU execution paths gracefully.
* Exposes a ``ChunkedFluxRunner`` class that callers can use directly or wrap in
  their own automation.
"""
from __future__ import annotations

import contextlib
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

from proteus_attention.kernels.sparse_attn import get_last_backend_info
from proteus_attention.models.dmoah import CausalDynamicAttention, ModelConfig

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[attr-defined]

# IMPORTANT: The sparse Triton backend autotunes the Flux kernel the first time
# it sees a new (chunk_len, head_dim, device) combination. For large chunks this
# benchmark sweep can take minutes and looks like a hang. Setting
# PROTEUS_TUNE_DISABLE=1 (or pre-populating the cache) skips autotune and makes
# smoke tests responsive.


# ROCm LLVM currently asserts when compiling Flux-style linear kernels above ~64K
# tokens per chunk. Keep the hard limit conservative until upstream fixes land.
MAX_FLUX_CHUNK_TOKENS = 65_536
BYTES_PER_MB = 1024.0 * 1024.0


def _system_available_ram() -> Optional[int]:
    """Best-effort estimate of available system RAM in bytes."""

    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:  # pragma: no cover - psutil failure
            pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")  # type: ignore[attr-defined]
        av_pages = os.sysconf("SC_AVPHYS_PAGES")  # type: ignore[attr-defined]
        return int(page_size * av_pages)
    except (AttributeError, ValueError, TypeError, OSError):  # pragma: no cover
        return None


def _sanitize_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                result[key] = value.detach().cpu().item()
            else:
                result[key] = value.detach().cpu().tolist()
        elif isinstance(value, dict):
            result[key] = _sanitize_mapping(value)
        else:
            result[key] = value
    return result


def _build_model_config(
    seq_len: int,
    d_model: int,
    *,
    heads: int,
    sparse_ratio: float,
    flux_alpha: float,
) -> ModelConfig:
    """Return a minimal ``ModelConfig`` tuned for chunked sparsity."""

    return ModelConfig(
        vocab_size=0,
        n_ctx=seq_len,
        n_layer=1,
        n_head=heads,
        d_model=d_model,
        p_dropout=0.0,
        bias=False,
        attn_variant="dmoah",
        attn_h_total=heads,
        attn_h_active=heads,
        attn_h_active_min=heads,
        attn_h_active_max=heads,
        attn_active_seq_low=max(128, seq_len // 4),
        attn_active_seq_high=seq_len,
        attn_mode="linear",
        attn_token_sparse=True,
        attn_token_keep_ratio=max(0.0, min(1.0, sparse_ratio)),
        attn_token_keep_min=max(4, int(seq_len * sparse_ratio * 0.5)),
        attn_token_keep_guard=4,
        attn_token_keep_threshold=0.0,
        attn_linear_L=256,
        attn_linear_L_base=256,
        attn_linear_L_min=64,
        attn_linear_L_max=256,
        attn_linear_switch_ctx=max(512, min(seq_len, 4096)),
        attn_linear_policy="local+anchors",
        attn_dna_enable=True,
        attn_dna_threshold=0.2,
        attn_dna_blend=0.5,
        attn_dna_temp=0.3,
        attn_linear_token_keep_schedule="sqrt",
        attn_linear_token_keep_min_ratio=0.05,
        attn_track_latency=False,
        attn_small_seq_dense=0,
        use_sdpa=False,
        attn_flux_alpha=flux_alpha,
    )


def _select_top_tokens(
    model: CausalDynamicAttention,
    global_offset: int,
    *,
    per_chunk_budget: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(scores, indices)`` tensors for the top tokens in the chunk."""

    importance = getattr(model, "_last_token_importance", None)
    if importance is None:
        return None
    scores = importance.squeeze(0).to(torch.float32)
    mask = getattr(model, "_last_token_mask", None)
    if mask is not None:
        scores = torch.where(
            mask.squeeze(0).to(torch.bool),
            scores,
            torch.full_like(scores, float("-inf")),
        )
    top_k = min(per_chunk_budget, int(scores.size(-1)))
    if top_k <= 0:
        return None
    values, indices = torch.topk(scores, k=top_k)
    finite_mask = torch.isfinite(values)
    if not torch.any(finite_mask):
        return None
    values = values[finite_mask]
    indices = indices[finite_mask] + int(global_offset)
    return values, indices


def _finalise_indices(
    scores: torch.Tensor,
    indices: torch.Tensor,
    *,
    max_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the highest-scoring global indices across all chunks.

    Returns
    -------
    keep_indices:
        Sorted tensor of unique indices promoted to the final buffer.
    keep_scores:
        Scores associated with ``keep_indices`` in ascending index order.
    """

    if scores.numel() == 0 or indices.numel() == 0:
        raise RuntimeError("No candidate tokens captured during chunking.")
    device = scores.device
    order = torch.argsort(scores, descending=True)
    ordered_scores = scores[order]
    ordered_indices = indices[order]
    if ordered_indices.numel() == 0:
        raise RuntimeError("No unique tokens available after deduplication.")

    positions = torch.arange(ordered_indices.size(0), device=device, dtype=torch.long)
    stride = positions.size(0) + 1
    composite = ordered_indices.to(torch.long) * stride - positions
    order_by_index = torch.argsort(composite)
    sorted_indices_by_index = ordered_indices[order_by_index]
    unique_mask = torch.ones_like(sorted_indices_by_index, dtype=torch.bool)
    if unique_mask.numel() > 1:
        unique_mask[1:] = sorted_indices_by_index[1:] != sorted_indices_by_index[:-1]
    first_positions = order_by_index[unique_mask]
    if first_positions.numel() == 0:
        raise RuntimeError("No unique tokens available after deduplication.")
    first_positions = torch.sort(first_positions)[0]
    selected_positions = first_positions[:max_tokens]
    selected_scores = ordered_scores[selected_positions]
    selected_indices = ordered_indices[selected_positions]
    sort_by_index = torch.argsort(selected_indices)
    keep_indices = selected_indices[sort_by_index].to(dtype=torch.long, device=device)
    keep_scores = selected_scores[sort_by_index].to(dtype=scores.dtype, device=device)
    return keep_indices.to("cpu"), keep_scores.to("cpu")


def _chunk_iter(total_len: int, chunk_len: int) -> Iterable[Tuple[int, int]]:
    """Yield ``(start, end)`` pairs that partition the sequence."""

    for start in range(0, total_len, chunk_len):
        end = min(total_len, start + chunk_len)
        yield start, end


@dataclass
class ChunkSummary:
    """Diagnostics collected for each processed chunk."""

    index: int
    start: int
    end: int
    promoted: Sequence[int] = field(default_factory=list)
    scores: Sequence[float] = field(default_factory=list)


@dataclass
class ChunkedFluxMetrics:
    """Aggregate metrics describing a Chunked Flux run."""

    original_tokens: int
    retained_tokens: int
    retention_ratio: float
    chunk_count: int
    device: str
    chunk_time_ms: Optional[float] = None
    final_time_ms: Optional[float] = None
    total_time_ms: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    used_fallback: bool = False
    chunk_tokens_per_s: Optional[float] = None
    final_tokens_per_s: Optional[float] = None
    total_tokens_per_s: Optional[float] = None
    storage_mode: str = "cpu"
    host_required_mb: Optional[float] = None
    host_allocated_mb: Optional[float] = None
    host_limit_mb: Optional[float] = None
    storage_reason: Optional[str] = None


@dataclass
class ChunkedFluxResult:
    """Structured output from :class:`ChunkedFluxRunner`."""

    keep_indices: torch.Tensor
    keep_scores: torch.Tensor
    reduced_sequence: torch.Tensor
    metrics: ChunkedFluxMetrics
    chunks: Sequence[ChunkSummary]
    final_output: Optional[torch.Tensor] = None
    final_stats: Optional[Dict[str, Any]] = None
    backend_info: Optional[Dict[str, Any]] = None


@dataclass
class ChunkedFluxConfig:
    """Configuration for the Chunked Flux pipeline."""

    seq_len: int
    d_model: int
    chunk_len: int
    buffer_tokens: int
    per_chunk_budget: int
    device: torch.device
    heads: int = 8
    chunk_sparse_ratio: float = 0.05
    final_sparse_ratio: float = 0.5
    seed: Optional[int] = 7
    report_latency: bool = False
    progress: bool = False
    run_final_pass: bool = True
    flux_alpha: float = 1.0
    storage: Literal["cpu", "disk", "auto"] = "auto"
    temp_dir: Optional[Path] = None
    ram_limit_bytes: Optional[int] = None

    def validate(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.chunk_len <= 0:
            raise ValueError("chunk_len must be positive.")
        if self.chunk_len > MAX_FLUX_CHUNK_TOKENS:
            raise ValueError(
                f"chunk_len={self.chunk_len} exceeds the conservative "
                f"MAX_FLUX_CHUNK_TOKENS={MAX_FLUX_CHUNK_TOKENS}. Larger values "
                "currently trigger ROCm/LLVM register allocation asserts; do "
                "not increase this cap until upstream fixes land."
            )
        if self.buffer_tokens <= 0:
            raise ValueError("buffer_tokens must be positive.")
        if self.per_chunk_budget <= 0:
            raise ValueError("per_chunk_budget must be positive.")
        if self.chunk_sparse_ratio <= 0.0:
            raise ValueError("chunk_sparse_ratio must be positive.")
        if self.final_sparse_ratio <= 0.0:
            raise ValueError("final_sparse_ratio must be positive.")
        if not (0.0 <= self.flux_alpha <= 1.0):
            raise ValueError("flux_alpha must be within [0, 1].")
        if self.storage not in {"cpu", "disk", "auto"}:
            raise ValueError("storage must be 'cpu', 'disk', or 'auto'.")
        if self.ram_limit_bytes is not None and self.ram_limit_bytes <= 0:
            raise ValueError("ram_limit_bytes must be positive when provided.")


class ChunkedFluxRunner:
    """Execute the Chunked Flux pipeline with structured diagnostics."""

    def __init__(
        self,
        config: ChunkedFluxConfig,
        *,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.config.validate()
        self.logger = logger or LOGGER

    def run(self, sequence: Optional[torch.Tensor] = None) -> ChunkedFluxResult:
        """
        Run the chunked pipeline.

        Parameters
        ----------
        sequence:
            Optional tensor containing the full sequence with shape ``(B, T, D)``.
            When ``None`` a synthetic random sequence is generated on CPU and
            streamed to the target device.
        """

        cfg = self.config
        device = cfg.device
        batch = 1

        if sequence is not None:
            if sequence.dim() != 3:
                raise ValueError("sequence tensor must have shape (B, T, D)")
            if sequence.size(1) < cfg.seq_len:
                raise ValueError(
                    f"sequence length {sequence.size(1)} smaller than configured seq_len {cfg.seq_len}"
                )
            if sequence.size(2) != cfg.d_model:
                raise ValueError(
                    f"sequence hidden size {sequence.size(2)} does not match d_model={cfg.d_model}"
                )
            if sequence.size(0) != 1:
                raise ValueError("Chunked Flux runner currently expects batch size of 1.")
            source_sequence = sequence[:, : cfg.seq_len].detach()
            base_dtype = source_sequence.dtype
        else:
            source_sequence = None
            base_dtype = torch.get_default_dtype()

        elem_size = torch.tensor([], dtype=base_dtype).element_size()
        host_required_bytes = int(batch * cfg.seq_len * cfg.d_model * elem_size)
        host_dataset_bytes = host_required_bytes
        available_bytes = _system_available_ram()
        limit_bytes = cfg.ram_limit_bytes
        host_limit_bytes: Optional[int] = limit_bytes if limit_bytes is not None else available_bytes
        storage_mode = cfg.storage
        storage_reason: Optional[str] = None

        if storage_mode == "auto":
            limit = host_limit_bytes
            if limit is not None and host_required_bytes > limit:
                storage_mode = "disk"
                storage_reason = (
                    f"required={host_required_bytes:,}B exceeds limit={limit:,}B; staging on disk"
                )
            else:
                storage_mode = "cpu"
                if limit is not None:
                    storage_reason = (
                        f"limit={limit:,}B, required={host_required_bytes:,}B"
                    )
                elif available_bytes is not None:
                    storage_reason = (
                        f"available≈{available_bytes:,}B, required={host_required_bytes:,}B"
                    )
        elif storage_mode == "cpu":
            if limit_bytes is not None and host_required_bytes > limit_bytes:
                storage_mode = "disk"
                storage_reason = (
                    f"required={host_required_bytes:,}B exceeds ram_limit={limit_bytes:,}B; staging on disk"
                )
        else:
            storage_mode = "disk"

        if storage_mode == "cpu" and available_bytes is not None:
            if host_required_bytes > int(available_bytes * 0.9):
                warn_msg = (
                    f"Host allocation {host_required_bytes / BYTES_PER_MB:.1f} MB is near "
                    f"available RAM {available_bytes / BYTES_PER_MB:.1f} MB."
                )
                self.logger.warning(warn_msg)
                if storage_reason is None:
                    storage_reason = warn_msg

        memmap_array: Optional[np.memmap] = None
        memmap_path: Optional[Path] = None
        base_cpu: Optional[torch.Tensor] = None
        non_blocking = False
        host_allocated_bytes: Optional[int] = None

        if storage_mode == "cpu":
            if source_sequence is not None:
                base_cpu = source_sequence.to("cpu", dtype=base_dtype)
            else:
                generator = torch.Generator(device="cpu")
                if cfg.seed is not None:
                    generator.manual_seed(int(cfg.seed))
                random_cpu = torch.randn(
                    1,
                    cfg.seq_len,
                    cfg.d_model,
                    generator=generator,
                    device="cpu",
                )
                base_cpu = random_cpu.to(base_dtype) if random_cpu.dtype != base_dtype else random_cpu
            host_allocated_bytes = int(base_cpu.numel() * base_cpu.element_size())
            batch = base_cpu.size(0)
            if device.type == "cuda" and hasattr(base_cpu, "pin_memory"):
                base_cpu = base_cpu.pin_memory()
            non_blocking = device.type == "cuda" and getattr(
                base_cpu, "is_pinned", lambda: False
            )()

            def fetch_chunk(start: int, end: int) -> torch.Tensor:
                return base_cpu[:, start:end]

            def gather_indices(indices: torch.Tensor) -> torch.Tensor:
                return base_cpu[:, indices]

        else:
            tmp_dir = cfg.temp_dir if cfg.temp_dir is not None else Path(tempfile.gettempdir())
            tmp_dir = Path(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                dir=tmp_dir, prefix="fluxchunk_", suffix=".mmap", delete=False
            )
            handle.close()
            memmap_path = Path(handle.name)
            memmap_array = np.memmap(
                memmap_path,
                dtype=np.float32,
                mode="w+",
                shape=(cfg.seq_len, cfg.d_model),
            )
            rows_per_block = max(
                1,
                min(
                    cfg.chunk_len,
                    int((64 * 1024 * 1024) / max(cfg.d_model * np.dtype(np.float32).itemsize, 1)),
                ),
            )
            if source_sequence is not None:
                for start in range(0, cfg.seq_len, rows_per_block):
                    end = min(cfg.seq_len, start + rows_per_block)
                    block = source_sequence[:, start:end].to("cpu", dtype=torch.float32)
                    memmap_array[start:end] = block.squeeze(0).numpy()
            else:
                generator = torch.Generator(device="cpu")
                if cfg.seed is not None:
                    generator.manual_seed(int(cfg.seed))
                for start in range(0, cfg.seq_len, rows_per_block):
                    end = min(cfg.seq_len, start + rows_per_block)
                    block = torch.randn(
                        end - start,
                        cfg.d_model,
                        generator=generator,
                        device="cpu",
                    )
                    memmap_array[start:end] = block.to(torch.float32).numpy()
            memmap_array.flush()
            non_blocking = False
            host_allocated_bytes = int(
                rows_per_block * cfg.d_model * np.dtype(np.float32).itemsize
            )
            host_dataset_bytes = int(
                cfg.seq_len * cfg.d_model * np.dtype(np.float32).itemsize
            )

            def fetch_chunk(start: int, end: int) -> torch.Tensor:
                view = np.asarray(memmap_array[start:end])
                tensor = torch.from_numpy(view).to(base_dtype)
                return tensor.unsqueeze(0)

            def gather_indices(indices: torch.Tensor) -> torch.Tensor:
                idx_np = indices.cpu().numpy()
                view = np.asarray(memmap_array[idx_np])
                tensor = torch.from_numpy(view).to(base_dtype)
                return tensor.unsqueeze(0)

        if storage_reason is not None:
            self.logger.info("Host staging: %s (%s)", storage_mode, storage_reason)
        else:
            self.logger.info("Host staging: %s", storage_mode)

        chunk_cfg = _build_model_config(
            cfg.chunk_len,
            cfg.d_model,
            heads=cfg.heads,
            sparse_ratio=cfg.chunk_sparse_ratio,
            flux_alpha=cfg.flux_alpha,
        )
        chunk_model = CausalDynamicAttention(chunk_cfg).to(
            device=device, dtype=base_dtype
        ).eval()
        if hasattr(chunk_model, "use_sdpa"):
            chunk_model.use_sdpa = False
        if hasattr(chunk_model, "set_flux_alpha"):
            chunk_model.set_flux_alpha(float(cfg.flux_alpha))

        chunk_tensor = torch.empty(
            batch, cfg.chunk_len, cfg.d_model, device=device, dtype=base_dtype
        )

        # Warm-up to ensure kernels are compiled before timing.
        dummy = torch.randn(
            batch, cfg.chunk_len, cfg.d_model, device=device, dtype=base_dtype
        )
        with torch.inference_mode():
            _ = chunk_model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device=None)

        chunk_scores: List[torch.Tensor] = []
        chunk_indices: List[torch.Tensor] = []
        chunk_summaries: List[ChunkSummary] = []
        used_fallback = False

        chunk_start_event = chunk_end_event = None
        if cfg.report_latency and device.type == "cuda":
            chunk_start_event = torch.cuda.Event(enable_timing=True)
            chunk_end_event = torch.cuda.Event(enable_timing=True)
            chunk_start_event.record()

        overall_start = time.perf_counter()
        chunk_complete_time = overall_start
        total_chunks = math.ceil(cfg.seq_len / cfg.chunk_len)
        if cfg.progress and tqdm is not None:
            iterable = enumerate(
                tqdm(
                    _chunk_iter(cfg.seq_len, cfg.chunk_len),
                    total=total_chunks,
                    desc="Chunked Flux",
                    unit="chunk",
                )
            )
        else:
            iterable = enumerate(_chunk_iter(cfg.seq_len, cfg.chunk_len))

        try:
            for chunk_idx, (start, end) in iterable:
                current_len = end - start
                chunk_source = fetch_chunk(start, end)
                chunk_tensor[:, :current_len].copy_(chunk_source, non_blocking=non_blocking)
                if current_len < cfg.chunk_len:
                    chunk_tensor[:, current_len:].zero_()
                with torch.inference_mode():
                    _ = chunk_model(chunk_tensor[:, :current_len])
                promoted = _select_top_tokens(
                    chunk_model,
                    start,
                    per_chunk_budget=cfg.per_chunk_budget,
                )
                if promoted is None:
                    # Fallback: derive importance from token norms when the router
                    # does not expose explicit scores (common on CPU back-ends).
                    token_slice = chunk_tensor[0, :current_len].detach()
                    fallback_scores = token_slice.norm(dim=-1).to(torch.float32)
                    top_k = min(cfg.per_chunk_budget, fallback_scores.numel())
                    if top_k == 0:
                        continue
                    values, rel_indices = torch.topk(fallback_scores, k=top_k)
                    indices_tensor = rel_indices.to(torch.long) + start
                    scores_tensor = values
                    used_fallback = True
                else:
                    scores_tensor, indices_tensor = promoted
                chunk_scores.append(scores_tensor)
                chunk_indices.append(indices_tensor)
                chunk_summaries.append(
                    ChunkSummary(
                        index=chunk_idx,
                        start=start,
                        end=end,
                        promoted=indices_tensor.tolist(),
                        scores=[float(s) for s in scores_tensor],
                    )
                )
                chunk_complete_time = time.perf_counter()
                if (
                    cfg.progress
                    and tqdm is None
                    and (chunk_idx + 1) % max(1, total_chunks // 10) == 0
                ):
                    self.logger.info(
                        "Processed %s/%s chunks (%0.2f%%)",
                        chunk_idx + 1,
                        total_chunks,
                        ((chunk_idx + 1) / max(total_chunks, 1)) * 100.0,
                    )

            if not chunk_scores:
                raise RuntimeError(
                    "Chunked pipeline selected zero tokens; adjust per_chunk_budget or sparse ratios."
                )
            if chunk_end_event is not None:
                chunk_end_event.record()

            all_scores = torch.cat(chunk_scores)
            all_indices = torch.cat(chunk_indices)
            keep_indices, keep_scores = _finalise_indices(
                all_scores,
                all_indices,
                max_tokens=cfg.buffer_tokens,
            )
            reduced_sequence_cpu = gather_indices(keep_indices.to(torch.long))
            reduced_sequence = reduced_sequence_cpu.to(device, non_blocking=non_blocking)
            chunk_time_ms: Optional[float] = None
            chunk_tokens_per_s: Optional[float] = None

            if cfg.report_latency and device.type == "cuda":
                torch.cuda.synchronize(device=None)
                if chunk_start_event is not None and chunk_end_event is not None:
                    chunk_time_ms = float(chunk_start_event.elapsed_time(chunk_end_event))
            if chunk_time_ms is None:
                chunk_time_ms = float((chunk_complete_time - overall_start) * 1_000.0)
            if chunk_time_ms > 0:
                chunk_tokens_per_s = float(cfg.seq_len * 1000.0 / chunk_time_ms)

            final_output: Optional[torch.Tensor] = None
            final_time_ms: Optional[float] = None
            peak_mb: Optional[float] = None
            final_stats: Optional[Dict[str, Any]] = None
            backend_info: Optional[Dict[str, Any]] = None
            final_end_clock = chunk_complete_time
            final_tokens_per_s: Optional[float] = None

            if cfg.run_final_pass:
                final_start_clock = time.perf_counter()
                final_cfg = _build_model_config(
                    reduced_sequence.size(1),
                    cfg.d_model,
                    heads=cfg.heads,
                    sparse_ratio=cfg.final_sparse_ratio,
                    flux_alpha=cfg.flux_alpha,
                )
                final_model = CausalDynamicAttention(final_cfg).to(
                    device=device, dtype=base_dtype
                ).eval()
                if hasattr(final_model, "use_sdpa"):
                    final_model.use_sdpa = False
                if hasattr(final_model, "set_flux_alpha"):
                    final_model.set_flux_alpha(float(cfg.flux_alpha))
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    final_start_event = final_end_event = None
                    if cfg.report_latency:
                        final_start_event = torch.cuda.Event(enable_timing=True)
                        final_end_event = torch.cuda.Event(enable_timing=True)
                        final_start_event.record()
                else:
                    final_start_event = final_end_event = None

                with torch.inference_mode():
                    output = final_model(reduced_sequence)
                final_output = output.detach()
                final_end_clock = time.perf_counter()
                backend_info = get_last_backend_info()
                raw_stats = getattr(final_model, "last_head_stats", {}) or {}
                sparse_state = getattr(final_model, "_last_sparse_state", {}) or {}
                if isinstance(raw_stats, dict):
                    final_stats = _sanitize_mapping(raw_stats)
                else:
                    final_stats = {}
                if isinstance(sparse_state, dict) and sparse_state:
                    final_stats = dict(final_stats or {})
                    final_stats["sparse_state"] = _sanitize_mapping(sparse_state)
                if cfg.report_latency and final_end_event is not None:
                    final_end_event.record()
                if device.type == "cuda":
                    torch.cuda.synchronize(device=None)
                    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                    if final_start_event is not None and final_end_event is not None:
                        final_time_ms = float(final_start_event.elapsed_time(final_end_event))
                else:
                    peak_mb = None
                if final_time_ms is None:
                    final_time_ms = float((final_end_clock - final_start_clock) * 1_000.0)
                if final_time_ms and final_time_ms > 0:
                    final_tokens_per_s = float(
                        keep_indices.numel() * 1000.0 / final_time_ms
                    )
            else:
                final_time_ms = None

            total_time = final_end_clock - overall_start
            total_tokens_per_s: Optional[float] = None
            if total_time > 0:
                total_tokens_per_s = float(cfg.seq_len / total_time)

            metrics = ChunkedFluxMetrics(
                original_tokens=cfg.seq_len,
                retained_tokens=int(keep_indices.numel()),
                retention_ratio=float(keep_indices.numel() / max(cfg.seq_len, 1)),
                chunk_count=len(chunk_summaries),
                device=str(device),
                chunk_time_ms=chunk_time_ms,
                final_time_ms=final_time_ms,
                total_time_ms=float(total_time * 1_000.0),
                peak_memory_mb=peak_mb,
                used_fallback=used_fallback,
                chunk_tokens_per_s=chunk_tokens_per_s,
                final_tokens_per_s=final_tokens_per_s,
                total_tokens_per_s=total_tokens_per_s,
                storage_mode=storage_mode,
                host_required_mb=host_dataset_bytes / BYTES_PER_MB,
                host_allocated_mb=(
                    host_allocated_bytes / BYTES_PER_MB if host_allocated_bytes is not None else None
                ),
                host_limit_mb=(
                    host_limit_bytes / BYTES_PER_MB if host_limit_bytes is not None else None
                ),
                storage_reason=storage_reason,
            )

            self.logger.info(
                "Chunked Flux retained %s/%s tokens (%.3f) across %s chunks",
                metrics.retained_tokens,
                metrics.original_tokens,
                metrics.retention_ratio,
                metrics.chunk_count,
            )
            if used_fallback:
                self.logger.warning(
                    "Router importance unavailable; used norm-based fallback scoring."
                )

            result = ChunkedFluxResult(
                keep_indices=keep_indices,
                keep_scores=keep_scores,
                reduced_sequence=reduced_sequence,
                metrics=metrics,
                chunks=chunk_summaries,
                final_output=final_output,
                final_stats=final_stats,
                backend_info=backend_info,
            )
            return result
        finally:
            if memmap_array is not None:
                memmap_array.flush()
            if memmap_path is not None:
                with contextlib.suppress(OSError):
                    os.remove(memmap_path)


__all__ = [
    "ChunkedFluxConfig",
    "ChunkedFluxMetrics",
    "ChunkedFluxResult",
    "ChunkedFluxRunner",
    "ChunkSummary",
    "MAX_FLUX_CHUNK_TOKENS",
]
