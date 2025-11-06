"""
Sparse attention kernels backing Genetic Attention, powered by the Dynamic
Mixture-of-Attention-Heads (DMoAH) architecture.

The module exposes a Triton implementation that gathers the active query rows
per head, runs a fused two-pass softmax with optional causal masking/dropout,
and scatters the result back into the dense output layout.  A PyTorch fallback
path mirrors the behaviour for CPU execution or environments without Triton.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TRITON_AVAILABLE = False
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

def _resolve_triton_seq_limit(default: int = 0) -> int:
    """Resolve the maximum sequence length for the Triton kernel."""

    env_value = os.getenv("DMOAH_TRITON_SEQ_LIMIT")
    if env_value is None:
        return default
    try:
        parsed = int(env_value)
    except ValueError:
        warnings.warn(
            f"Invalid DMOAH_TRITON_SEQ_LIMIT value '{env_value}', using default {default}",
            RuntimeWarning,
        )
        return default
    return parsed


TRITON_SEQ_LEN_LIMIT = _resolve_triton_seq_limit()

CUDA_BACKEND = None
if os.getenv("DMOAH_USE_CUDA_KERNEL"):
    try:  # pragma: no cover - optional back-end
        from . import dmoah_cuda as CUDA_BACKEND  # type: ignore
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"DMoAH CUDA kernel requested but unavailable ({exc}); falling back to Triton/PyTorch",
            RuntimeWarning,
        )
        CUDA_BACKEND = None


_LAST_BACKEND_INFO: Dict[str, object] = {"name": "uninitialized", "details": {}}


def _record_backend(name: str, **details: object) -> None:
    global _LAST_BACKEND_INFO
    _LAST_BACKEND_INFO = {"name": name, "details": details}


def get_last_backend() -> str:
    """Return the name of the backend used by the most recent call."""

    return str(_LAST_BACKEND_INFO.get("name", "uninitialized"))


def get_last_backend_info() -> Dict[str, object]:
    """Return a shallow copy of the most recent backend metadata."""

    info = dict(_LAST_BACKEND_INFO)
    details = info.get("details")
    if isinstance(details, dict):
        info["details"] = dict(details)
    return info


_BLOCK_DEFAULT_CONFIG: Tuple[int, int, int] = (32, 64, 32)
_BLOCK_CANDIDATES: List[Tuple[int, int, int]] = [
    (16, 32, 16),
    (32, 32, 16),
    (32, 64, 16),
    (32, 64, 32),
    (64, 64, 32),
    (64, 128, 32),
    (64, 128, 64),
    (128, 128, 64),
]

_BRUTE_FORCE_ENABLED = os.getenv("PROTEUS_TUNE_BRUTE_FORCE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
    "bruteforce",
}
_BRUTE_FORCE_WARNED = False

_CACHE_ENV_DIR = os.getenv("PROTEUS_CACHE_DIR")
if _CACHE_ENV_DIR:
    _CACHE_DIR = Path(_CACHE_ENV_DIR).expanduser()
else:
    try:
        _CACHE_DIR = Path.home() / ".cache" / "proteus_attention"
    except Exception:
        _CACHE_DIR = Path.cwd() / ".proteus_attention_cache"
_CACHE_FILE = _CACHE_DIR / "shortlist_block_config.json"
_BLOCK_CONFIG_CACHE: Dict[str, Tuple[int, int, int]] = {}


def _load_block_cache() -> None:
    global _BLOCK_CONFIG_CACHE
    if not _CACHE_FILE.exists():
        _BLOCK_CONFIG_CACHE = {}
        return
    try:
        with _CACHE_FILE.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Failed to read Proteus Attention block-size cache ({exc}); ignoring persisted values.",
            RuntimeWarning,
        )
        _BLOCK_CONFIG_CACHE = {}
        return
    cache: Dict[str, Tuple[int, int, int]] = {}
    for key, value in raw.items():
        if (
            isinstance(key, str)
            and isinstance(value, (list, tuple))
            and len(value) == 3
        ):
            try:
                cfg = tuple(int(v) for v in value)
            except (TypeError, ValueError):
                continue
            cache[key] = cfg  # type: ignore[assignment]
    _BLOCK_CONFIG_CACHE = cache


def _save_block_cache() -> None:
    if not _BLOCK_CONFIG_CACHE:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        serializable = {k: list(v) for k, v in _BLOCK_CONFIG_CACHE.items()}
        with _CACHE_FILE.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, sort_keys=True)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Unable to persist Proteus Attention block-size cache ({exc}).",
            RuntimeWarning,
        )


_load_block_cache()


def _block_cache_key(device: torch.device, head_dim: int) -> str:
    if device.type != "cuda":
        return f"{device.type}|{head_dim}"
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        cc_major, cc_minor = torch.cuda.get_device_capability(index)
        capability = f"{cc_major}.{cc_minor}"
    except Exception:
        name = "cuda"
        capability = "unknown"
    return f"{name}|{capability}|hd{head_dim}"


def _should_disable_autotune() -> bool:
    flag = os.getenv("PROTEUS_TUNE_DISABLE", "")
    return flag.strip().lower() in {"1", "true", "yes", "disable"}


def _should_force_autotune() -> bool:
    flag = os.getenv("PROTEUS_TUNE_FORCE", "")
    return flag.strip().lower() in {"1", "true", "yes", "force"}


def _candidate_block_configs(head_dim: int) -> List[Tuple[int, int, int]]:
    global _BRUTE_FORCE_WARNED
    if _BRUTE_FORCE_ENABLED:
        if not _BRUTE_FORCE_WARNED:
            warnings.warn(
                "PROTEUS_TUNE_BRUTE_FORCE enabled; exhaustive block search may take longer.",
                RuntimeWarning,
            )
            _BRUTE_FORCE_WARNED = True
        combos: List[Tuple[int, int, int]] = []
        for m in range(16, 129, 16):
            for n in range(16, 129, 16):
                for d in range(16, 129, 16):
                    if head_dim > 0 and d > head_dim:
                        continue
                    if head_dim > 0 and head_dim % d != 0:
                        continue
                    combos.append((m, n, d))
        if not combos:
            combos.append(_BLOCK_DEFAULT_CONFIG)
        return combos

    viable: List[Tuple[int, int, int]] = []
    for cfg in _BLOCK_CANDIDATES:
        m, n, d = cfg
        if d > head_dim and head_dim > 0:
            continue
        viable.append((m, n, d))
    if not viable:
        viable.append(_BLOCK_DEFAULT_CONFIG)
    return viable


def get_block_config_cache() -> Dict[str, Tuple[int, int, int]]:
    """Return the currently cached block-size configurations."""

    return dict(_BLOCK_CONFIG_CACHE)


def _shortlist_prepare_proto_candidates(
    rows: torch.Tensor,
    seq_len: int,
    max_candidates: int,
    proto_scores: Optional[torch.Tensor],
    proto_cap: int,
) -> Optional[torch.Tensor]:
    if proto_scores is None:
        return None
    if proto_scores.numel() == 0:
        return None
    if proto_cap <= 0 or max_candidates <= 0:
        return None
    device = rows.device
    proto = proto_scores.detach().to(device=device, dtype=torch.float32)
    proto = proto[:seq_len]
    if proto.dim() == 0:
        proto = proto.unsqueeze(0)
    if proto.dim() > 1:
        proto = proto.view(-1)
    if proto.numel() == 0:
        return None

    max_index = proto.size(0)
    cap = min(max_candidates, proto_cap, max_index)
    if cap <= 0:
        return None

    counts = torch.clamp(rows + 1, max=cap)
    indices = torch.arange(max_index, device=device)
    mask = indices.unsqueeze(0) <= rows.unsqueeze(1)
    scores = proto.unsqueeze(0).expand(rows.size(0), -1)
    scores = scores.masked_fill(~mask, float('-inf'))
    top_idx = torch.topk(scores, k=cap, dim=1).indices.to(rows.dtype)

    base = rows.unsqueeze(1).expand(-1, cap).to(rows.dtype)
    proto_candidates = base.clone()
    valid = torch.arange(cap, device=device).unsqueeze(0) < counts.unsqueeze(1)
    proto_candidates = torch.where(valid, top_idx.to(rows.dtype), proto_candidates)
    return proto_candidates


def _shortlist_prepare_proto_candidates_grouped(
    rows: torch.Tensor,
    row_batch_ids: Optional[torch.Tensor],
    seq_len: int,
    max_candidates: int,
    proto_scores: Optional[torch.Tensor],
    proto_cap: int,
) -> Optional[torch.Tensor]:
    """
    Vectorised helper that assembles prototype shortlist candidates for multiple batches at once.

    Parameters
    ----------
    rows:
        Tensor of active token indices with shape ``(N,)``.
    row_batch_ids:
        Tensor of shape ``(N,)`` mapping each row to the batch index whose prototype scores should be used.
    proto_scores:
        Optional tensor containing prototype similarity values. Expected shape ``(B, seq_len)`` or
        ``(seq_len,)`` when only a single batch is present.
    """

    if (
        proto_scores is None
        or proto_scores.numel() == 0
        or proto_cap <= 0
        or max_candidates <= 0
        or rows.numel() == 0
    ):
        return None

    if proto_scores.dim() == 1:
        proto_scores = proto_scores.unsqueeze(0)

    if row_batch_ids is None:
        if proto_scores.size(0) != 1:
            warnings.warn(
                "row_batch_ids not provided for grouped prototype candidates; using first prototype vector for all rows.",
                RuntimeWarning,
            )
        row_batch_ids = torch.zeros_like(rows, dtype=torch.long)
    else:
        row_batch_ids = row_batch_ids.to(device=rows.device).to(torch.long)

    cap = min(max_candidates, proto_cap, seq_len)
    if cap <= 0:
        return None

    device = rows.device
    dtype = rows.dtype
    result = rows.unsqueeze(1).expand(-1, cap).to(dtype).clone()

    unique_batches = torch.unique(row_batch_ids.to(torch.long))
    for batch_id in unique_batches.tolist():
        mask = row_batch_ids == batch_id
        if torch.count_nonzero(mask) == 0:
            continue
        subset_rows = rows[mask]
        proto_vec = proto_scores[batch_id]
        candidates = _shortlist_prepare_proto_candidates(
            subset_rows,
            seq_len,
            max_candidates=max_candidates,
            proto_scores=proto_vec,
            proto_cap=proto_cap,
        )
        if candidates is None or candidates.numel() == 0:
            continue
        width = min(candidates.size(1), result.size(1))
        result[mask, :width] = candidates[:, :width].to(device=device, dtype=dtype)

    return result


def build_shortlist_candidates(
    rows: torch.Tensor,
    seq_len: int,
    *,
    max_candidates: int,
    linear_window: int,
    anchor_stride: int,
    use_local: bool,
    use_anchors: bool,
    proto_scores: Optional[torch.Tensor] = None,
    local_cap: Optional[int] = None,
    anchor_cap: Optional[int] = None,
    proto_cap: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rows.numel() == 0 or max_candidates <= 0:
        empty = rows.new_empty(rows.shape + (0,))
        return empty, rows.new_zeros(rows.shape, dtype=torch.long)

    device = rows.device
    dtype = rows.dtype
    pieces: list[torch.Tensor] = []

    local_cap_val = max(0, int(local_cap)) if local_cap is not None else (max_candidates if use_local else 0)
    anchor_cap_val = max(0, int(anchor_cap)) if anchor_cap is not None else (max_candidates if use_anchors else 0)
    proto_cap_val = max(0, int(proto_cap)) if proto_cap is not None else (max_candidates if proto_scores is not None else 0)

    if use_local and local_cap_val > 0:
        window = min(linear_window, max_candidates, local_cap_val)
        if window > 0:
            steps = torch.arange(window, device=device, dtype=dtype)
            lengths = torch.clamp(rows + 1, max=window)
            start = rows - lengths + 1
            start = torch.clamp(start, min=0)
            local = start.unsqueeze(1) + steps.unsqueeze(0)
            valid = steps.unsqueeze(0) < lengths.unsqueeze(1)
            local = torch.where(valid, local, rows.unsqueeze(1))
            pieces.append(local)

    if use_anchors and anchor_stride > 0 and anchor_cap_val > 0:
        slots = min(anchor_cap_val, max_candidates, max(1, seq_len))
        if slots > 0:
            steps = torch.arange(1, slots + 1, device=device, dtype=dtype)
            anchors = rows.unsqueeze(1) - steps.unsqueeze(0) * anchor_stride
            anchors = torch.clamp(anchors, min=0)
            pieces.append(anchors)

    proto_candidates = _shortlist_prepare_proto_candidates(rows, seq_len, max_candidates, proto_scores, proto_cap_val)
    if proto_candidates is not None and proto_candidates.numel() > 0:
        pieces.append(proto_candidates.to(dtype))

    if pieces:
        candidates = torch.cat(pieces, dim=1)
    else:
        candidates = rows.unsqueeze(1)

    candidates = torch.clamp(candidates, min=0, max=max(seq_len - 1, 0))
    candidates = torch.minimum(candidates, rows.unsqueeze(1))
    candidates = torch.cat([candidates, rows.unsqueeze(1)], dim=1)

    candidates, _ = torch.sort(candidates, dim=1)
    keep_mask = torch.ones_like(candidates, dtype=torch.bool)
    if candidates.size(1) > 1:
        keep_mask[:, 1:] = candidates[:, 1:] != candidates[:, :-1]
    keep_mask[:, 0] = True

    unique_indices = keep_mask.to(torch.int64).cumsum(dim=1) - 1
    valid_keep = keep_mask & (unique_indices >= 0)
    valid_keep = valid_keep & (unique_indices < max_candidates)

    shortlist_lengths = valid_keep.sum(dim=1, dtype=torch.long)
    shortlist_lengths = torch.clamp(shortlist_lengths, min=1)

    output = rows.unsqueeze(1).expand(-1, max_candidates).to(dtype).clone()
    if valid_keep.any():
        row_idx, src_col = valid_keep.nonzero(as_tuple=True)
        dst_col = unique_indices[row_idx, src_col].to(torch.long)
        output[row_idx, dst_col] = candidates[row_idx, src_col]

    return output.contiguous(), shortlist_lengths.contiguous()


def build_packed_shortlist_candidates(
    rows: torch.Tensor,
    row_batch_ids: Optional[torch.Tensor],
    seq_len: int,
    *,
    max_candidates: int,
    linear_window: int,
    anchor_stride: int,
    use_local: bool,
    use_anchors: bool,
    proto_scores: Optional[torch.Tensor] = None,
    local_cap: Optional[int] = None,
    anchor_cap: Optional[int] = None,
    proto_cap: Optional[int] = None,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Assemble Shortlist shortlist candidates for a packed set of rows spanning multiple heads/batches.

    Parameters mirror :func:`build_shortlist_candidates`, but ``rows`` contains every active row in the
    packed structure and ``row_batch_ids`` provides a lookup into ``proto_scores`` so we can generate
    Prototype shortlist slots without looping head-by-head in Python.
    """

    if rows.numel() == 0 or max_candidates <= 0:
        empty = rows.new_empty(rows.shape + (0,))
        return empty, rows.new_zeros(rows.shape, dtype=torch.long)

    device = rows.device
    dtype = rows.dtype

    local_cap_val = (
        max(0, int(local_cap))
        if local_cap is not None
        else (max_candidates if use_local else 0)
    )
    anchor_cap_val = (
        max(0, int(anchor_cap))
        if anchor_cap is not None
        else (max_candidates if use_anchors else 0)
    )
    proto_cap_val = (
        max(0, int(proto_cap))
        if proto_cap is not None
        else (max_candidates if proto_scores is not None else 0)
    )

    chunk_size = max(1, int(chunk_size))
    total_rows = rows.size(0)
    output = rows.unsqueeze(1).expand(-1, max_candidates).to(dtype).clone()
    shortlist_lengths = torch.ones(total_rows, device=device, dtype=torch.long)

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        chunk_rows = rows[start:end]
        chunk_count = chunk_rows.size(0)

        chunk_pieces: list[torch.Tensor] = []

        if use_local and local_cap_val > 0:
            window = min(linear_window, max_candidates, local_cap_val)
            if window > 0:
                steps = torch.arange(window, device=device, dtype=dtype)
                lengths = torch.clamp(chunk_rows + 1, max=window)
                start_idx = torch.clamp(chunk_rows - lengths + 1, min=0)
                local = start_idx.unsqueeze(1) + steps.unsqueeze(0)
                valid = steps.unsqueeze(0) < lengths.unsqueeze(1)
                local = torch.where(valid, local, chunk_rows.unsqueeze(1))
                chunk_pieces.append(local)

        if use_anchors and anchor_stride > 0 and anchor_cap_val > 0:
            slots = min(anchor_cap_val, max_candidates, max(1, seq_len))
            if slots > 0:
                steps = torch.arange(1, slots + 1, device=device, dtype=dtype)
                anchors = chunk_rows.unsqueeze(1) - steps.unsqueeze(0) * anchor_stride
                anchors = torch.clamp(anchors, min=0)
                chunk_pieces.append(anchors)

        if proto_scores is not None and proto_cap_val > 0:
            chunk_batch_ids = None
            if row_batch_ids is not None:
                chunk_batch_ids = row_batch_ids[start:end]
            proto_chunk = _shortlist_prepare_proto_candidates_grouped(
                chunk_rows,
                chunk_batch_ids,
                seq_len,
                max_candidates=max_candidates,
                proto_scores=proto_scores,
                proto_cap=proto_cap_val,
            )
            if proto_chunk is not None and proto_chunk.numel() > 0:
                chunk_pieces.append(proto_chunk.to(dtype))

        if chunk_pieces:
            candidates = torch.cat(chunk_pieces, dim=1)
        else:
            candidates = chunk_rows.unsqueeze(1)

        candidates = torch.clamp(candidates, min=0, max=max(seq_len - 1, 0))
        candidates = torch.minimum(candidates, chunk_rows.unsqueeze(1))
        candidates = torch.cat([candidates, chunk_rows.unsqueeze(1)], dim=1)

        candidates, _ = torch.sort(candidates, dim=1)
        keep_mask = torch.ones_like(candidates, dtype=torch.bool)
        if candidates.size(1) > 1:
            keep_mask[:, 1:] = candidates[:, 1:] != candidates[:, :-1]
        keep_mask[:, 0] = True

        unique_indices = keep_mask.to(torch.int64).cumsum(dim=1) - 1
        valid_keep = keep_mask & (unique_indices >= 0)
        valid_keep = valid_keep & (unique_indices < max_candidates)

        chunk_lengths = valid_keep.sum(dim=1, dtype=torch.long)
        chunk_lengths = torch.clamp(chunk_lengths, min=1)

        chunk_output = chunk_rows.unsqueeze(1).expand(-1, max_candidates).to(dtype).clone()
        if valid_keep.any():
            row_idx, src_col = valid_keep.nonzero(as_tuple=True)
            dst_col = unique_indices[row_idx, src_col].to(torch.long)
            chunk_output[row_idx, dst_col] = candidates[row_idx, src_col]

        output[start:end] = chunk_output
        shortlist_lengths[start:end] = chunk_lengths

    return output.contiguous(), shortlist_lengths.contiguous()
if TRITON_AVAILABLE:
    @triton.jit  # pragma: no cover - executed on device
    def _shortlist_attention_kernel(
        Q_ACTIVE_PTR,
        K_PTR,
        V_PTR,
        OUT_PTR,
        TOKEN_PTR,
        ROW_OFFSET_PTR,
        MASK_PTR,
        CAND_PTR,
        ROW_LEN_PTR,
        STRIDE_QA_ROW,
        STRIDE_QA_D,
        STRIDE_KH,
        STRIDE_KT,
        STRIDE_KD,
        STRIDE_VH,
        STRIDE_VT,
        STRIDE_VD,
        STRIDE_OH,
        STRIDE_OT,
        STRIDE_OD,
        STRIDE_TOKEN,
        STRIDE_ROW_OFFSET,
        STRIDE_MASK_M,
        STRIDE_MASK_N,
        T,
        D,
        SCALE,
        DROP_SEED,
        DROP_OFFSET,
        DROP_P,
        DROP_SCALE,
        QSCALE_PTR,
        KSCALE_PTR,
        VSCALE_PTR,
        MAX_CAND,
        BLOCK_M: tl.constexpr,  # type: ignore[assignment]
        BLOCK_N: tl.constexpr,  # type: ignore[assignment]
        BLOCK_D: tl.constexpr,  # type: ignore[assignment]
        USE_CANDIDATES: tl.constexpr,  # type: ignore[assignment]
        HAS_MASK: tl.constexpr,  # type: ignore[assignment]
        HAS_DROPOUT: tl.constexpr,  # type: ignore[assignment]
        HAS_QSCALE: tl.constexpr,  # type: ignore[assignment]
        HAS_KSCALE: tl.constexpr,  # type: ignore[assignment]
        HAS_VSCALE: tl.constexpr,  # type: ignore[assignment]
    ):
        # One Kernel to Rule them All — the Shortlist kernel adapts on the fly.
        NEG_INF = float('-inf')
        pid_block = tl.program_id(0)
        pid_head = tl.program_id(1)

        head_row_start = tl.load(ROW_OFFSET_PTR + pid_head * STRIDE_ROW_OFFSET)
        head_row_end = tl.load(ROW_OFFSET_PTR + (pid_head + 1) * STRIDE_ROW_OFFSET)
        row_start = head_row_start + pid_block * BLOCK_M
        if row_start >= head_row_end:
            return

        row_arange = tl.arange(0, BLOCK_M)
        row_indices = row_start + row_arange
        row_mask = row_indices < head_row_end
        row_mask_f = tl.where(row_mask, 1.0, 0.0)

        token_idx = tl.load(
            TOKEN_PTR + row_indices * STRIDE_TOKEN,
            mask=row_mask,
            other=0,
        ).to(tl.int32)

        head_offset_k = pid_head * STRIDE_KH
        head_offset_v = pid_head * STRIDE_VH
        head_offset_o = pid_head * STRIDE_OH

        scale_q = tl.full((), 1.0, tl.float32)
        scale_k = tl.full((), 1.0, tl.float32)
        scale_v = tl.full((), 1.0, tl.float32)
        if HAS_QSCALE:
            scale_q = tl.load(QSCALE_PTR + pid_head).to(tl.float32)
        if HAS_KSCALE:
            scale_k = tl.load(KSCALE_PTR + pid_head).to(tl.float32)
        if HAS_VSCALE:
            scale_v = tl.load(VSCALE_PTR + pid_head).to(tl.float32)

        offs_d_init = tl.arange(0, BLOCK_D)
        offs_n_init = tl.arange(0, BLOCK_N)

        m_i = tl.full((BLOCK_M,), NEG_INF, tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)

        if USE_CANDIDATES:
            cand_lengths = tl.load(ROW_LEN_PTR + row_indices, mask=row_mask, other=0).to(tl.int32)
            cand_lengths = tl.where(row_mask, cand_lengths, 0)
            for start_c in range(0, MAX_CAND, BLOCK_N):
                offs_c = start_c + offs_n_init
                cand_valid = offs_c[None, :] < cand_lengths[:, None]
                cand_ptr = CAND_PTR + row_indices[:, None] * MAX_CAND + offs_c[None, :]
                cand_idx = tl.load(
                    cand_ptr,
                    mask=row_mask[:, None] & cand_valid,
                    other=0,
                ).to(tl.int32)
                cand_idx = tl.where(cand_valid, cand_idx, 0)

                att_block = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

                for start_d in range(0, D, BLOCK_D):
                    offs_d = start_d + offs_d_init
                    q_tile = tl.load(
                        Q_ACTIVE_PTR + row_indices[:, None] * STRIDE_QA_ROW + offs_d[None, :] * STRIDE_QA_D,
                        mask=row_mask[:, None] & (offs_d[None, :] < D),
                        other=0.0,
                    ).to(tl.float32)
                    q_tile = q_tile * scale_q

                    flat_valid = tl.reshape(cand_valid, (BLOCK_M * BLOCK_N,))
                    flat_idx = tl.reshape(cand_idx, (BLOCK_M * BLOCK_N,)).to(tl.int32)
                    k_tile = tl.load(
                        K_PTR
                        + head_offset_k
                        + flat_idx[:, None] * STRIDE_KT
                        + offs_d[None, :] * STRIDE_KD,
                        mask=flat_valid[:, None] & (offs_d[None, :] < D),
                        other=0.0,
                    ).to(tl.float32)
                    k_tile = k_tile * scale_k
                    k_tile = tl.reshape(k_tile, (BLOCK_M, BLOCK_N, BLOCK_D))

                    q_expanded = tl.reshape(q_tile, (BLOCK_M, 1, BLOCK_D))
                    att_block += tl.sum(q_expanded * k_tile, axis=2)

                att_block = att_block * SCALE
                if HAS_MASK:
                    mask_block = tl.load(
                        MASK_PTR + token_idx[:, None] * STRIDE_MASK_M + cand_idx * STRIDE_MASK_N,
                        mask=row_mask[:, None] & cand_valid,
                        other=0.0,
                    ).to(tl.float32)
                    att_block += mask_block

                valid = row_mask[:, None] & cand_valid
                att_block = tl.where(valid, att_block, NEG_INF)

                m_block = tl.max(att_block, axis=1)
                m_block = tl.where(row_mask, m_block, NEG_INF)
                m_new = tl.maximum(m_i, m_block)
                m_new = tl.where(row_mask, m_new, m_i)

                m_new_finite = m_new != NEG_INF
                m_new_safe = tl.where(m_new_finite, m_new, 0.0)
                logits = att_block - m_new_safe[:, None]
                logits = tl.where(valid & m_new_finite[:, None], logits, NEG_INF)
                p = tl.exp(logits)
                if HAS_DROPOUT:
                    rng_offsets = tl.cast(
                        DROP_OFFSET + token_idx[:, None] * T + cand_idx,
                        tl.int32,
                    )
                    keep = tl.rand(DROP_SEED, rng_offsets)
                    keep = keep > DROP_P
                    keep = tl.where(valid, keep, False)
                    p = p * keep.to(p.dtype) * DROP_SCALE
                p = p * row_mask_f[:, None]

                l_block = tl.sum(p, axis=1)
                l_block = tl.where(m_new_finite, l_block, 0.0)
                m_i_safe = tl.where(m_new_finite, m_i, 0.0)
                scale_prev = tl.exp(m_i_safe - m_new_safe)
                scale_prev = tl.where(m_new_finite, scale_prev, 0.0)
                l_i = tl.where(m_new_finite, l_i * scale_prev + l_block, l_i)
                m_i = tl.where(m_new_finite, m_new, m_i)

                p_expanded = tl.reshape(p, (BLOCK_M, BLOCK_N, 1))
                flat_valid = tl.reshape(valid, (BLOCK_M * BLOCK_N,))
                flat_idx = tl.reshape(cand_idx, (BLOCK_M * BLOCK_N,)).to(tl.int32)
                for start_d in range(0, D, BLOCK_D):
                    offs_d = start_d + offs_d_init
                    d_mask = offs_d < D
                    out_ptr = (
                        OUT_PTR
                        + head_offset_o
                        + token_idx[:, None] * STRIDE_OT
                        + offs_d[None, :] * STRIDE_OD
                    )
                    prev = tl.load(
                        out_ptr,
                        mask=row_mask[:, None] & d_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    prev = prev * scale_prev[:, None]

                    v_tile = tl.load(
                        V_PTR
                        + head_offset_v
                        + flat_idx[:, None] * STRIDE_VT
                        + offs_d[None, :] * STRIDE_VD,
                        mask=flat_valid[:, None] & (offs_d[None, :] < D),
                        other=0.0,
                    ).to(tl.float32)
                    v_tile = v_tile * scale_v
                    v_tile = tl.reshape(v_tile, (BLOCK_M, BLOCK_N, BLOCK_D))
                    contrib = tl.sum(p_expanded * v_tile, axis=1)
                    prev += contrib
                    tl.store(
                        out_ptr,
                        prev,
                        mask=row_mask[:, None] & d_mask[None, :],
                    )
        else:
            for start_n in range(0, T, BLOCK_N):
                offs_n = start_n + offs_n_init
                col_mask = offs_n < T
                att_block = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

                for start_d in range(0, D, BLOCK_D):
                    offs_d = start_d + offs_d_init
                    q_tile = tl.load(
                        Q_ACTIVE_PTR + row_indices[:, None] * STRIDE_QA_ROW + offs_d[None, :] * STRIDE_QA_D,
                        mask=row_mask[:, None] & (offs_d[None, :] < D),
                        other=0.0,
                    ).to(tl.float32)
                    q_tile = q_tile * scale_q
                    k_tile = tl.load(
                        K_PTR + head_offset_k + offs_n[:, None] * STRIDE_KT + offs_d[None, :] * STRIDE_KD,
                        mask=(offs_n[:, None] < T) & (offs_d[None, :] < D),
                        other=0.0,
                    ).to(tl.float32)
                    k_tile = k_tile * scale_k
                    att_block += tl.dot(q_tile, tl.trans(k_tile))

                att_block = att_block * SCALE
                if HAS_MASK:
                    mask_block = tl.load(
                        MASK_PTR + token_idx[:, None] * STRIDE_MASK_M + offs_n[None, :] * STRIDE_MASK_N,
                        mask=row_mask[:, None] & col_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    att_block += mask_block

                valid = row_mask[:, None] & col_mask[None, :]
                att_block = tl.where(valid, att_block, NEG_INF)

                m_block = tl.max(att_block, axis=1)
                m_block = tl.where(row_mask, m_block, NEG_INF)
                m_new = tl.maximum(m_i, m_block)
                m_new = tl.where(row_mask, m_new, m_i)

                m_new_finite = m_new != NEG_INF
                m_new_safe = tl.where(m_new_finite, m_new, 0.0)
                logits = att_block - m_new_safe[:, None]
                logits = tl.where(valid & m_new_finite[:, None], logits, NEG_INF)
                p = tl.exp(logits)
                if HAS_DROPOUT:
                    rng_offsets = tl.cast(
                        DROP_OFFSET + token_idx[:, None] * T + offs_n[None, :],
                        tl.int32,
                    )
                    keep = tl.rand(DROP_SEED, rng_offsets)
                    keep = keep > DROP_P
                    keep = tl.where(valid, keep, False)
                    p = p * keep.to(p.dtype) * DROP_SCALE
                p = p * row_mask_f[:, None]

                l_block = tl.sum(p, axis=1)
                l_block = tl.where(m_new_finite, l_block, 0.0)
                m_i_safe = tl.where(m_new_finite, m_i, 0.0)
                scale_prev = tl.exp(m_i_safe - m_new_safe)
                scale_prev = tl.where(m_new_finite, scale_prev, 0.0)
                l_i = tl.where(m_new_finite, l_i * scale_prev + l_block, l_i)
                m_i = tl.where(m_new_finite, m_new, m_i)

                for start_d in range(0, D, BLOCK_D):
                    offs_d = start_d + offs_d_init
                    d_mask = offs_d < D
                    out_ptr = (
                        OUT_PTR
                        + head_offset_o
                        + token_idx[:, None] * STRIDE_OT
                        + offs_d[None, :] * STRIDE_OD
                    )
                    prev = tl.load(
                        out_ptr,
                        mask=row_mask[:, None] & d_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    prev = prev * scale_prev[:, None]

                    v_tile = tl.load(
                        V_PTR + head_offset_v + offs_n[:, None] * STRIDE_VT + offs_d[None, :] * STRIDE_VD,
                        mask=(offs_n[:, None] < T) & d_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    v_tile = v_tile * scale_v
                    contrib = tl.dot(p, v_tile)
                    prev += contrib
                    tl.store(
                        out_ptr,
                        prev,
                        mask=row_mask[:, None] & d_mask[None, :],
                    )

        inv_l = tl.where(row_mask & (l_i > 0.0), 1.0 / l_i, 0.0)

        for start_d in range(0, D, BLOCK_D):
            offs_d = start_d + offs_d_init
            d_mask = offs_d < D
            out_ptr = (
                OUT_PTR
                + head_offset_o
                + token_idx[:, None] * STRIDE_OT
                + offs_d[None, :] * STRIDE_OD
            )
            out_block = tl.load(
                out_ptr,
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            out_block = out_block * inv_l[:, None]
            tl.store(
                out_ptr,
                out_block,
                mask=row_mask[:, None] & d_mask[None, :],
            )
else:
    _shortlist_attention_kernel = None  # type: ignore[assignment]


def _pack_active_rows(
    active_mask: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Pack active (head, token) pairs for sparse processing."""

    if active_mask.dim() != 2:
        raise ValueError("active_mask must have rank 2 after squeeze")

    active_positions = active_mask.nonzero(as_tuple=False)
    if active_positions.numel() == 0:
        return None

    head_idx = active_positions[:, 0]
    token_idx = active_positions[:, 1]

    total_heads = active_mask.size(0)
    counts = torch.bincount(head_idx, minlength=total_heads).to(torch.int32)
    row_offsets = torch.zeros(
        total_heads + 1, device=active_mask.device, dtype=torch.int32
    )
    torch.cumsum(counts, dim=0, out=row_offsets[1:])

    max_rows = int(counts.max().item()) if counts.numel() > 0 else 0
    return head_idx.to(torch.int64), token_idx.to(torch.int64), row_offsets, max_rows


def _fallback_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_active: torch.Tensor,
    token_idx: torch.Tensor,
    row_offsets: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    scale: float,
    dropout_real: float,
    training: bool,
    shortlist_candidates: Optional[torch.Tensor],
    shortlist_lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    """Reference Shortlist attention using PyTorch ops per head (fallback path)."""

    batch_heads, tokens, head_dim = q.shape

    result = torch.zeros_like(q)
    row_offsets_cpu = row_offsets.cpu()
    use_shortlist = shortlist_candidates is not None and shortlist_lengths is not None and shortlist_candidates.numel() > 0

    for head in range(batch_heads):
        start = int(row_offsets_cpu[head].item())
        end = int(row_offsets_cpu[head + 1].item())
        if start == end:
            continue

        rows_tokens = token_idx[start:end]
        q_sel = q_active[start:end]

        if use_shortlist:
            cand_rows = shortlist_candidates[start:end].to(device=q.device, dtype=torch.long)
            len_rows = shortlist_lengths[start:end].to(device=q.device, dtype=torch.long)
            max_len = cand_rows.size(1)

            k_head = k[head]
            v_head = v[head]
            k_sel = k_head[cand_rows.clamp(min=0, max=tokens - 1)]
            v_sel = v_head[cand_rows.clamp(min=0, max=tokens - 1)]

            mask_invalid = torch.arange(max_len, device=q.device).unsqueeze(0) >= len_rows.unsqueeze(1)
            k_sel = k_sel.masked_fill(mask_invalid.unsqueeze(-1), 0.0)
            v_sel = v_sel.masked_fill(mask_invalid.unsqueeze(-1), 0.0)

            scores = torch.einsum('rd,rld->rl', q_sel, k_sel) * scale
            if causal_mask is not None:
                causal = causal_mask.index_select(0, rows_tokens.to(torch.long))
                causal = torch.gather(causal, 1, cand_rows.clamp(min=0, max=causal.size(1) - 1))
                scores = scores + causal.to(scores.dtype)

            scores = scores.masked_fill(mask_invalid, float('-inf'))
            probs = torch.softmax(scores, dim=-1)
            if dropout_real > 0.0 and training:
                probs = F.dropout(probs, p=dropout_real, training=True)

            out_sel = torch.bmm(probs.unsqueeze(1), v_sel).squeeze(1)
            result[head].index_copy_(0, rows_tokens, out_sel)
        else:
            k_head = k[head].unsqueeze(0)
            v_head = v[head].unsqueeze(0)

            max_token = int(rows_tokens.max().item())
            k_slice = k_head[:, : max_token + 1, :]
            v_slice = v_head[:, : max_token + 1, :]

            attn_mask = None
            if causal_mask is not None:
                mask_slice = causal_mask.index_select(0, rows_tokens.to(torch.long))
                mask_slice = mask_slice[:, : max_token + 1]
                attn_mask = mask_slice.unsqueeze(0)
            else:
                key_positions = torch.arange(
                    max_token + 1, device=rows_tokens.device, dtype=rows_tokens.dtype
                )
                future_mask = key_positions.unsqueeze(0) > rows_tokens.unsqueeze(1)
                if future_mask.any():
                    attn_mask = torch.zeros(
                        (1, rows_tokens.size(0), max_token + 1),
                        device=q.device,
                        dtype=q_sel.dtype,
                    )
                    attn_mask = attn_mask.masked_fill(future_mask.unsqueeze(0), float("-inf"))

            out_sel = F.scaled_dot_product_attention(
                q_sel.unsqueeze(0),
                k_slice,
                v_slice,
                attn_mask=attn_mask,
                dropout_p=dropout_real,
                is_causal=False,
            ).squeeze(0)

            result[head].index_copy_(0, rows_tokens, out_sel)

    return result


def _should_try_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    active_mask: Optional[torch.Tensor],
    dropout_p: float,
    training: bool,
    causal_mask: Optional[torch.Tensor],
    shortlist_candidates: Optional[torch.Tensor],
    shortlist_lengths: Optional[torch.Tensor],
) -> bool:
    """Return ``True`` when the Shortlist Triton path can be attempted."""

    if not TRITON_AVAILABLE:
        return False
    if active_mask is None:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda and active_mask.is_cuda):
        return False
    if causal_mask is not None and not causal_mask.is_cuda:
        return False
    limit = TRITON_SEQ_LEN_LIMIT
    if limit and limit > 0 and q.size(1) > limit:
        return False
    if shortlist_candidates is not None and shortlist_lengths is None:
        return False
    return True


def _run_shortlist_attention_kernel(
    block_cfg: Tuple[int, int, int],
    *,
    q_active: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    token_indices: torch.Tensor,
    row_offsets: torch.Tensor,
    mask: torch.Tensor,
    cand: torch.Tensor,
    lengths: torch.Tensor,
    tokens: int,
    head_dim: int,
    scale: float,
    dropout_seed: int,
    dropout_offset: int,
    dropout_p: float,
    dropout_scale: float,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    max_cand: int,
    use_candidates: bool,
    has_mask: bool,
    has_dropout: bool,
    has_qscale: bool,
    has_kscale: bool,
    has_vscale: bool,
    heads: int,
    max_rows: int,
) -> None:
    q_active_stride0 = q_active.stride(0)
    q_active_stride1 = q_active.stride(1)
    k_stride0 = k.stride(0)
    k_stride1 = k.stride(1)
    k_stride2 = k.stride(2)
    v_stride0 = v.stride(0)
    v_stride1 = v.stride(1)
    v_stride2 = v.stride(2)
    out_stride0 = out.stride(0)
    out_stride1 = out.stride(1)
    out_stride2 = out.stride(2)
    token_stride0 = token_indices.stride(0)
    row_offsets_stride0 = row_offsets.stride(0)
    mask_stride0 = mask.stride(0) if has_mask else 0
    mask_stride1 = mask.stride(1) if has_mask else 0

    block_m, block_n, block_d = block_cfg
    grid = (
        triton.cdiv(max_rows, block_m),
        heads,
    )
    _shortlist_attention_kernel[grid](
        q_active,
        k,
        v,
        out,
        token_indices,
        row_offsets,
        mask,
        cand,
        lengths,
        q_active_stride0,
        q_active_stride1,
        k_stride0,
        k_stride1,
        k_stride2,
        v_stride0,
        v_stride1,
        v_stride2,
        out_stride0,
        out_stride1,
        out_stride2,
        token_stride0,
        row_offsets_stride0,
        mask_stride0,
        mask_stride1,
        tokens,
        head_dim,
        scale,
        dropout_seed,
        dropout_offset,
        dropout_p,
        dropout_scale,
        q_scale,
        k_scale,
        v_scale,
        max_cand,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        USE_CANDIDATES=int(use_candidates),
        HAS_MASK=has_mask,
        HAS_DROPOUT=has_dropout,
        HAS_QSCALE=has_qscale,
        HAS_KSCALE=has_kscale,
        HAS_VSCALE=has_vscale,
        num_warps=4,
        num_stages=2,
    )


def _select_block_config(
    device: torch.device,
    head_dim: int,
    benchmark_fn,
) -> Tuple[int, int, int]:
    default_cfg = _BLOCK_CONFIG_CACHE.get(
        _block_cache_key(device, head_dim), _BLOCK_DEFAULT_CONFIG
    )
    if device.type != "cuda":
        return default_cfg

    key = _block_cache_key(device, head_dim)

    if not _should_force_autotune() and key in _BLOCK_CONFIG_CACHE:
        return _BLOCK_CONFIG_CACHE[key]
    if _should_disable_autotune():
        return _BLOCK_CONFIG_CACHE.get(key, default_cfg)

    best_cfg: Optional[Tuple[int, int, int]] = None
    best_time: Optional[float] = None

    for cfg in _candidate_block_configs(head_dim):
        try:
            elapsed = benchmark_fn(cfg)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Block configuration {cfg} failed during autotune ({exc}); skipping.",
                RuntimeWarning,
            )
            continue
        if elapsed is None:
            continue
        if best_time is None or elapsed < best_time:
            best_time = elapsed
            best_cfg = cfg

    if best_cfg is None:
        best_cfg = default_cfg

    _BLOCK_CONFIG_CACHE[key] = best_cfg
    _save_block_cache()
    return best_cfg


def _launch_triton_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_active: torch.Tensor,
    token_indices: torch.Tensor,
    row_offsets: torch.Tensor,
    max_rows: int,
    causal_mask: Optional[torch.Tensor],
    scale: float,
    dropout_p: float,
    dropout_seed: int,
    dropout_offset: int,
    dropout_scale: float,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    shortlist_candidates: Optional[torch.Tensor],
    shortlist_lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    """Launch the Triton Shortlist kernel for the DMoAH sparse attention forward pass."""

    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton runtime is not available")
    if not q.is_cuda:
        raise RuntimeError("Triton path requires CUDA tensors")

    if max_rows <= 0:
        return torch.zeros_like(q)

    q_contig = q.contiguous()
    k_contig = k.contiguous()
    v_contig = v.contiguous()
    q_active_contig = q_active.contiguous()
    token_contig = token_indices.to(device=q.device, dtype=torch.int32).contiguous()
    row_offsets_contig = row_offsets.to(device=q.device, dtype=torch.int32).contiguous()

    heads = q_contig.size(0)
    tokens = q_contig.size(1)
    head_dim = q_contig.size(2)

    out = torch.zeros_like(q_contig)

    has_qscale = q_scale is not None
    has_kscale = k_scale is not None
    has_vscale = v_scale is not None
    if has_qscale:
        q_scale_contig = q_scale.to(device=q.device, dtype=torch.float32).contiguous()
    else:
        q_scale_contig = torch.ones(1, device=q.device, dtype=torch.float32)
    if has_kscale:
        k_scale_contig = k_scale.to(device=q.device, dtype=torch.float32).contiguous()
    else:
        k_scale_contig = torch.ones(1, device=q.device, dtype=torch.float32)
    if has_vscale:
        v_scale_contig = v_scale.to(device=q.device, dtype=torch.float32).contiguous()
    else:
        v_scale_contig = torch.ones(1, device=q.device, dtype=torch.float32)

    has_mask = causal_mask is not None
    has_dropout = dropout_p > 0.0
    if has_mask:
        mask = causal_mask.contiguous()
        if mask.dim() != 2 or mask.size(0) != tokens or mask.size(1) != tokens:
            raise NotImplementedError("Triton path expects a (T, T) causal mask")
    else:
        mask = torch.empty(1, device=q_contig.device, dtype=q_contig.dtype)

    dropout_seed = int(dropout_seed)
    dropout_offset = int(dropout_offset)
    dropout_p = float(dropout_p)
    dropout_scale = float(dropout_scale)

    if shortlist_candidates is not None and shortlist_lengths is not None and shortlist_candidates.numel() > 0:
        cand_contig = shortlist_candidates.to(device=q.device, dtype=torch.int32).contiguous()
        len_contig = shortlist_lengths.to(device=q.device, dtype=torch.int32).contiguous()
        if cand_contig.size(0) != token_contig.size(0):
            raise ValueError("Shortlist candidate rows must match the number of active tokens.")
        if len_contig.size(0) != token_contig.size(0):
            raise ValueError("Shortlist candidate lengths must match the number of active tokens.")
        max_cand = int(cand_contig.size(1))
        use_candidates = max_cand > 0
    else:
        cand_contig = torch.empty(1, device=q.device, dtype=torch.int32)
        len_contig = torch.empty(1, device=q.device, dtype=torch.int32)
        max_cand = 0
        use_candidates = False

    kernel_args = dict(
        q_active=q_active_contig,
        k=k_contig,
        v=v_contig,
        token_indices=token_contig,
        row_offsets=row_offsets_contig,
        mask=mask,
        cand=cand_contig,
        lengths=len_contig,
        tokens=tokens,
        head_dim=head_dim,
        scale=scale,
        dropout_seed=dropout_seed,
        dropout_offset=dropout_offset,
        dropout_p=dropout_p,
        dropout_scale=dropout_scale,
        q_scale=q_scale_contig,
        k_scale=k_scale_contig,
        v_scale=v_scale_contig,
        max_cand=max_cand,
        use_candidates=use_candidates,
        has_mask=has_mask,
        has_dropout=has_dropout,
        has_qscale=has_qscale,
        has_kscale=has_kscale,
        has_vscale=has_vscale,
        heads=heads,
        max_rows=max_rows,
    )

    device = q_contig.device

    if device.type == "cuda":

        def _benchmark(block_cfg: Tuple[int, int, int]) -> Optional[float]:
            tmp_out = torch.empty_like(out)
            torch.cuda.synchronize(device)
            _run_shortlist_attention_kernel(block_cfg, out=tmp_out, **kernel_args)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            _run_shortlist_attention_kernel(block_cfg, out=tmp_out, **kernel_args)
            torch.cuda.synchronize(device)
            return time.perf_counter() - start

    else:

        def _benchmark(block_cfg: Tuple[int, int, int]) -> Optional[float]:
            return None

    block_cfg = _select_block_config(device, head_dim, _benchmark)

    _run_shortlist_attention_kernel(block_cfg, out=out, **kernel_args)

    return out


def dmoah_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    active_mask: Optional[torch.Tensor] = None,
    causal_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    prepacked: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    ] = None,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    shortlist_candidates: Optional[torch.Tensor] = None,
    shortlist_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reference fallback for the DMoAH sparse attention kernel.

    Parameters
    ----------
    q, k, v:
        Projected tensors with shape ``(B * H, T, D)``.
    active_mask:
        Optional mask with shape ``(B * H, T, 1)``.  In the real kernel this
        will be used to skip inactive heads entirely.  Here we simply zero the
        corresponding rows after the dense computation.
    causal_mask:
        Optional additive mask with shape ``(T, T)`` (values should be 0 or
        ``-inf``).  When provided we expand it for SDPA.
    shortlist_candidates / shortlist_lengths:
        Optional per-row candidate lists for Shortlist mode. ``shortlist_candidates``
        should have shape ``(N_active, L_max)`` and ``shortlist_lengths`` shape
        ``(N_active,)`` describing how many entries of each row are valid.
    dropout_p:
        Attention dropout probability.  Passed through to the dense routine.
    training:
        Whether to apply dropout.  When ``False`` we force ``dropout_p=0``.
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q, k, v must be rank-3 tensors of shape (B*H, T, D)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have matching shapes")

    batch_heads, tokens, head_dim = q.shape
    dropout_p = float(dropout_p)
    if dropout_p < 0.0 or dropout_p >= 1.0:
        raise ValueError("dropout_p must satisfy 0.0 <= dropout_p < 1.0")
    training = bool(training)
    dropout_real = dropout_p if training else 0.0
    _record_backend("uninitialized")

    is_quantized = q.dtype == torch.int8 or k.dtype == torch.int8 or v.dtype == torch.int8
    if out_dtype is None:
        out_dtype = q.dtype if getattr(q.dtype, "is_floating_point", False) and q.dtype.is_floating_point else torch.float32

    if is_quantized:
        if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
            raise ValueError("Quantized path expects int8 q, k, v tensors")
        if q_scale is None or k_scale is None or v_scale is None:
            raise ValueError("Quantized path requires q_scale, k_scale, and v_scale tensors")
        q_scale = q_scale.to(device=q.device, dtype=torch.float32).contiguous().view(-1)
        k_scale = k_scale.to(device=k.device, dtype=torch.float32).contiguous().view(-1)
        v_scale = v_scale.to(device=v.device, dtype=torch.float32).contiguous().view(-1)
        if q_scale.numel() != batch_heads or k_scale.numel() != batch_heads or v_scale.numel() != batch_heads:
            raise ValueError("Scale tensors must have length equal to the number of heads in q/k/v")
    else:
        q_scale = None
        k_scale = None
        v_scale = None

    if causal_mask is not None:
        if causal_mask.dim() != 2 or causal_mask.size(0) != tokens or causal_mask.size(1) != tokens:
            raise ValueError("causal_mask must have shape (T, T)")
        causal_mask = causal_mask.to(device=q.device, dtype=out_dtype).contiguous()
    if active_mask is None:
        mask_bool_bool = torch.ones(batch_heads, tokens, device=q.device, dtype=torch.bool)
    else:
        if active_mask.dim() == 3 and active_mask.size(2) == 1:
            active_mask = active_mask.squeeze(-1)
        if active_mask.dim() != 2 or active_mask.size(0) != batch_heads or active_mask.size(1) != tokens:
            raise ValueError("active_mask must have shape (B*H, T)")

        active_mask = active_mask.to(device=q.device).contiguous()
        if active_mask.dtype == torch.bool:
            mask_bool_bool = active_mask
        else:
            mask_bool_bool = (active_mask > 0.0)
        if mask_bool_bool.dtype != torch.bool:
            mask_bool_bool = mask_bool_bool.to(torch.bool)
        mask_bool_bool = mask_bool_bool.contiguous()

    if prepacked is not None:
        packed = prepacked
    else:
        packed = _pack_active_rows(mask_bool_bool)
    if packed is None:
        if mask_bool_bool.all():
            head_idx = torch.arange(batch_heads, device=q.device, dtype=torch.long).repeat_interleave(tokens)
            token_idx = torch.arange(tokens, device=q.device, dtype=torch.long).repeat(batch_heads)
            row_offsets = torch.arange(0, (batch_heads + 1) * tokens, tokens, device=q.device, dtype=torch.int32)
            max_rows = tokens
            packed = (head_idx, token_idx, row_offsets, max_rows)
        else:
            _record_backend("empty", device=str(q.device), heads=batch_heads, quantized=bool(is_quantized))
            return torch.zeros(q.shape, device=q.device, dtype=out_dtype)

    head_idx, token_idx, row_offsets, max_rows = packed

    shortlist_runtime_candidates: Optional[torch.Tensor]
    shortlist_runtime_lengths: Optional[torch.Tensor]
    if shortlist_candidates is not None or shortlist_lengths is not None:
        if shortlist_candidates is None or shortlist_lengths is None:
            raise ValueError("shortlist_candidates and shortlist_lengths must be provided together.")
        if shortlist_candidates.dim() != 2:
            raise ValueError("shortlist_candidates must have shape (N_active, L_max)")
        if shortlist_lengths.dim() != 1:
            raise ValueError("shortlist_lengths must have shape (N_active,)")
        if shortlist_candidates.size(0) != shortlist_lengths.size(0):
            raise ValueError("shortlist_lengths must align with shortlist_candidate rows")
        shortlist_runtime_candidates = shortlist_candidates.to(device=q.device, dtype=torch.long).contiguous()
        shortlist_runtime_lengths = shortlist_lengths.to(device=q.device, dtype=torch.long).contiguous()
        if shortlist_runtime_candidates.size(0) != token_idx.numel():
            raise ValueError("Shortlist candidates must match the number of active rows")
    else:
        shortlist_runtime_candidates = None
        shortlist_runtime_lengths = None

    if CUDA_BACKEND is not None and q.is_cuda and not is_quantized:
        try:  # pragma: no cover - optional path
            out_cuda = CUDA_BACKEND.dmoah_sparse_attention(
                q,
                k,
                v,
                active_mask=mask_bool_bool,
                causal_mask=causal_mask,
                dropout_p=dropout_real,
                training=training,
                prepacked=packed,
                shortlist_candidates=shortlist_runtime_candidates,
                shortlist_lengths=shortlist_runtime_lengths,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Falling back from CUDA kernel: {exc}", RuntimeWarning)
        else:
            _record_backend(
                "cuda",
                device=str(q.device),
                heads=batch_heads,
                tokens=tokens,
                max_rows=int(max_rows),
            )
            return out_cuda
    q_dense = q.view(batch_heads, tokens, head_dim)
    k_dense = k.view(batch_heads, tokens, head_dim)
    v_dense = v.view(batch_heads, tokens, head_dim)
    if is_quantized:
        q_dense_float = q_dense.to(torch.float32) * q_scale.view(-1, 1, 1)
        k_dense_float = k_dense.to(torch.float32) * k_scale.view(-1, 1, 1)
        v_dense_float = v_dense.to(torch.float32) * v_scale.view(-1, 1, 1)
    else:
        q_dense_float = q_dense if q_dense.dtype.is_floating_point else q_dense.to(out_dtype)
        k_dense_float = k_dense if k_dense.dtype.is_floating_point else k_dense.to(out_dtype)
        v_dense_float = v_dense if v_dense.dtype.is_floating_point else v_dense.to(out_dtype)

    q_active = q_dense[head_idx, token_idx]
    q_active_float = q_dense_float[head_idx, token_idx]
    token_idx_int = token_idx.to(torch.int32)

    scale = 1.0 / (q.size(-1) ** 0.5)

    if _should_try_triton(
        q,
        k,
        v,
        active_mask=mask_bool_bool,
        dropout_p=dropout_real,
        training=training,
        causal_mask=causal_mask,
        shortlist_candidates=shortlist_runtime_candidates,
        shortlist_lengths=shortlist_runtime_lengths,
    ):
        has_dropout = dropout_real > 0.0
        dropout_seed = 0
        dropout_offset = 0
        dropout_scale = 1.0
        if has_dropout:
            dropout_seed = int(
                torch.randint(
                    0,
                    2**31 - 1,
                    (1,),
                    device=q.device,
                    dtype=torch.int64,
                ).item()
            )
            dropout_offset = int(
                torch.randint(
                    0,
                    2**31 - 1,
                    (1,),
                    device=q.device,
                    dtype=torch.int64,
                ).item()
            )
            dropout_scale = 1.0 / (1.0 - dropout_real)
        try:
            triton_out = _launch_triton_sparse_attention(
                q,
                k,
                v,
                q_active=q_active,
                token_indices=token_idx_int,
                row_offsets=row_offsets,
                max_rows=max_rows,
                causal_mask=causal_mask,
                scale=scale,
                dropout_p=dropout_real,
                dropout_seed=dropout_seed,
                dropout_offset=dropout_offset,
                dropout_scale=dropout_scale,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                shortlist_candidates=shortlist_runtime_candidates,
                shortlist_lengths=shortlist_runtime_lengths,
            )
        except (NotImplementedError, RuntimeError):
            triton_out = None
        else:
            _record_backend(
                "triton",
                device=str(q.device),
                heads=batch_heads,
                tokens=tokens,
                max_rows=int(max_rows),
                quantized=bool(is_quantized),
            )
            return triton_out.to(out_dtype)

    fallback = _fallback_sparse_attention(
        q_dense_float,
        k_dense_float,
        v_dense_float,
        q_active=q_active_float,
        token_idx=token_idx,
        row_offsets=row_offsets,
        causal_mask=causal_mask,
        scale=scale,
        dropout_real=dropout_real,
        training=training,
        shortlist_candidates=shortlist_runtime_candidates,
        shortlist_lengths=shortlist_runtime_lengths,
    )

    fallback = fallback.view(batch_heads, tokens, head_dim)
    _record_backend(
        "fallback_dense",
        device=str(q.device),
        heads=batch_heads,
        tokens=tokens,
        max_rows=int(max_rows),
        quantized=bool(is_quantized),
    )
    return fallback.reshape_as(q).to(out_dtype)


__all__ = [
    "dmoah_sparse_attention",
    "get_last_backend",
    "get_last_backend_info",
    "build_shortlist_candidates",
    "build_packed_shortlist_candidates",
]
