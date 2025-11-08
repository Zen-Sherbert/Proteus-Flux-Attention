#!/usr/bin/env python
"""
Synthetic checks for the chunked Shortlist pipeline.

The helpers here exercise the sentinel/needle recall behaviour, the effect of
adding an auxiliary Proto-style importance boost, and the ordering guarantees that
keep indices remain monotonic (mirroring the RoPE position story).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Sequence, Union, Optional

import torch

# Ensure the project `src` tree is importable when running standalone.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
SAMPLES_DIR = THIS_DIR / "samples"
DEFAULT_SAMPLE_FILE = SAMPLES_DIR / "shortlist_sample.txt"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from proteus_attention.tools.chunked_shortlist import (
    ChunkedShortlistConfig,
    ChunkedShortlistRunner,
    MAX_SHORTLIST_CHUNK_TOKENS,
    _chunk_iter,
    _finalise_indices,
    _select_top_tokens,
    _nucleus_select,
)


@dataclass
class ChunkLog:
    """Simple container for per-chunk diagnostics."""

    start: int
    end: int
    promoted: List[int]
    scores: List[float]


class _MockModel:
    """Mimic the small interface used by `_select_top_tokens`."""

    def __init__(self) -> None:
        self._last_token_importance: torch.Tensor | None = None
        self._last_token_mask: torch.Tensor | None = None


def _select_top_tokens_cpu(
    importance: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    global_offset: int,
    *,
    per_chunk_budget: int,
    nucleus_top_p: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if importance is None:
        return None
    scores = importance.squeeze(0).to(torch.float32)
    if mask is not None:
        scores = torch.where(
            mask.squeeze(0).to(torch.bool),
            scores,
            torch.full_like(scores, float("-inf")),
        )
    selection = _nucleus_select(scores, per_chunk_budget, nucleus_top_p)
    if selection is None:
        return None
    values, indices = selection
    finite_mask = torch.isfinite(values)
    if not torch.any(finite_mask):
        return None
    values = values[finite_mask]
    indices = indices[finite_mask] + int(global_offset)
    return values, indices


def _resolve_device(choice: Optional[str]) -> torch.device:
    if choice is None or str(choice).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def _fmt_ms(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:7.1f}"


def _fmt_rate(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value >= 1e6:
        return f"{value/1e6:.2f}M"
    if value >= 1e3:
        return f"{value/1e3:.2f}k"
    return f"{value:.0f}"


def _fmt_percent(value: Optional[float]) -> str:
    return "-" if value is None else f"{value*100.0:6.2f}%"


def _fmt_mem(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:7.1f}"


def _sentinel_positions(seq_len: int, count: int) -> List[int]:
    count = max(1, count)
    step = seq_len / float(count + 1)
    slots: List[int] = []
    for idx in range(count):
        pos = int(round(step * (idx + 1)))
        pos = max(0, min(seq_len - 1, pos))
        if slots and pos <= slots[-1]:
            pos = min(seq_len - 1, slots[-1] + 1)
        slots.append(pos)
    return slots


def _build_demo_sequence(
    seq_len: int,
    d_model: int,
    sentinels: Sequence[int],
    boost: float,
    width: int,
    seed: Optional[int],
) -> torch.Tensor:
    width = max(1, min(width, d_model))
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    base = torch.randn(1, seq_len, d_model, generator=generator, dtype=torch.float32)
    for idx in sentinels:
        if 0 <= idx < seq_len:
            base[0, idx, :width] += boost
            if width < d_model:
                base[0, idx, width:] -= boost * 0.15
    return base


def _run_pipeline(
    *,
    seq_len: int,
    d_model: int,
    chunk_len: int,
    per_chunk_budget: int,
    buffer_tokens: int,
    sentinel_idx: Union[int, Sequence[int]],
    base_boost: Union[float, torch.Tensor, dict[int, float]],
    proto_boost: Union[float, dict[int, float]] = 0.0,
    rng_seed: int = 123,
    adaptive_margin: Optional[float] = None,
    max_chunk_extra: int = 8,
    nucleus_top_p: float = 0.9,
) -> Tuple[torch.Tensor, List[ChunkLog]]:
    """Execute the chunking helpers with synthetic importance scores."""

    torch.manual_seed(rng_seed)
    model = _MockModel()
    chunk_scores: List[torch.Tensor] = []
    chunk_indices: List[torch.Tensor] = []
    logs: List[ChunkLog] = []

    if isinstance(sentinel_idx, int):
        sentinels = [sentinel_idx]
    else:
        sentinels = list(sentinel_idx)

    def _boost_for(idx: int, value: Union[float, dict[int, float]]) -> float:
        if isinstance(value, dict):
            return float(value.get(idx, 0.0))
        return float(value)

    for start, end in _chunk_iter(seq_len, chunk_len):
        current_len = end - start
        importance = torch.randn(1, current_len, dtype=torch.float32) * 0.05
        sentinel_present = False
        for idx in sentinels:
            if start <= idx < end:
                sentinel_present = True
                pos = idx - start
                importance[0, pos] += _boost_for(idx, base_boost)
                if proto_boost:
                    importance[0, pos] += _boost_for(idx, proto_boost)
        if not sentinel_present:
            # Create a couple of distractors so the sentinel has competition.
            importance[0, torch.randint(0, current_len, (2,))] += 0.25

        model._last_token_importance = importance
        model._last_token_mask = None
        budget_this_chunk = min(per_chunk_budget, current_len)
        if adaptive_margin is not None and budget_this_chunk < current_len:
            raw_scores = importance.squeeze(0).to(torch.float32)
            max_score = float(raw_scores.max().item()) if raw_scores.numel() > 0 else 0.0
            threshold = max_score - float(adaptive_margin)
            qualifying = int((raw_scores >= threshold).sum().item())
            if qualifying > budget_this_chunk:
                extra = min(max_chunk_extra, qualifying - budget_this_chunk)
                budget_this_chunk = min(current_len, budget_this_chunk + extra)
        promoted = _select_top_tokens(
            model,
            start,
            per_chunk_budget=budget_this_chunk,
            nucleus_top_p=nucleus_top_p,
        )
        if promoted is None:
            promoted = _select_top_tokens_cpu(
                model._last_token_importance,
                model._last_token_mask,
                start,
                per_chunk_budget=budget_this_chunk,
                nucleus_top_p=nucleus_top_p,
            )
        if promoted is None:
            continue
        scores_tensor, indices_tensor = promoted
        chunk_scores.append(scores_tensor)
        chunk_indices.append(indices_tensor)
        logs.append(
            ChunkLog(
                start=start,
                end=end,
                promoted=indices_tensor.tolist(),
                scores=[float(s) for s in scores_tensor],
            )
        )

    if not chunk_scores:
        return torch.empty(0, dtype=torch.long), logs

    all_scores = torch.cat(chunk_scores)
    all_indices = torch.cat(chunk_indices)
    keep_indices_tensor, _ = _finalise_indices(
        all_scores, all_indices, max_tokens=buffer_tokens
    )
    return keep_indices_tensor.to(torch.long), logs


def _format_logs(logs: Iterable[ChunkLog]) -> str:
    lines = []
    for item in logs:
        lines.append(
            f"[chunk {item.start:>5}:{item.end:<5}] promoted={item.promoted} scores={[round(s, 3) for s in item.scores]}"
        )
    return "\n".join(lines)


def _load_sample_tokens(path: Path, limit: Optional[int] = None) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to read sample file {path}: {exc}") from exc
    tokens = text.replace("\n", " ").split()
    if limit is not None and limit > 0:
        tokens = tokens[:limit]
    return tokens


def _find_phrase_positions(
    tokens: Sequence[str],
    phrase: Sequence[str],
    *,
    casefold: bool,
) -> List[int]:
    if not phrase:
        return []
    if casefold:
        lowered = [tok.casefold() for tok in tokens]
        needle = [tok.casefold() for tok in phrase]
    else:
        lowered = list(tokens)
        needle = list(phrase)
    k = len(needle)
    hits: List[int] = []
    for idx in range(0, len(lowered) - k + 1):
        if lowered[idx : idx + k] == needle:
            hits.append(idx)
    return hits


def _sample_text_demo(args: argparse.Namespace) -> None:
    tokens: List[str]
    source_label: str
    tokens = []
    source_label = ""
    if args.sample_file is not None:
        sample_path = args.sample_file
        if sample_path.exists():
            tokens = _load_sample_tokens(sample_path, limit=args.sample_limit)
            source_label = str(sample_path)
        else:
            print(f"[sample] Sample file {sample_path} not found; falling back to inline text.")
    if not tokens:
        sample_text = args.sample_text or ""
        tokens = sample_text.replace("\n", " ").split()
        if args.sample_limit is not None and args.sample_limit > 0:
            tokens = tokens[: args.sample_limit]
        source_label = source_label or "inline sample text"
    if not tokens:
        print("[sample] No tokens available for sample demo; skipping.")
        return
    print(
        f"[sample] Loaded {len(tokens):,} tokens from {source_label} "
        f"(limit={args.sample_limit or 'none'})"
    )
    phrase = (args.sample_needle.split() if args.sample_needle else [])
    sentinel_indices = _find_phrase_positions(
        tokens,
        phrase,
        casefold=args.sample_casefold,
    ) if phrase else []
    if not sentinel_indices:
        # Fall back to simple anchors if no explicit phrase hit.
        pivots = [
            max(0, len(tokens) // 4),
            max(0, len(tokens) // 2),
            max(0, (3 * len(tokens)) // 4),
        ]
        sentinel_indices = sorted({idx for idx in pivots if idx < len(tokens)})
        if args.sample_needle:
            print(
                f"[sample] Phrase '{args.sample_needle}' not found; "
                "falling back to heuristic sentinel positions."
            )
    if not sentinel_indices:
        print("[sample] Unable to determine sentinel indices; skipping sample demo.")
        return
    chunk_len = min(args.chunk_len, max(1, len(tokens)))
    per_chunk_budget = min(chunk_len, max(args.per_chunk_budget, len(sentinel_indices)))
    buffer_tokens = max(args.buffer_tokens, len(sentinel_indices) * 2)
    keep_indices, _ = _run_pipeline(
        seq_len=len(tokens),
        d_model=args.d_model,
        chunk_len=chunk_len,
        per_chunk_budget=per_chunk_budget,
        buffer_tokens=buffer_tokens,
        sentinel_idx=sentinel_indices,
        base_boost=5.0,
        proto_boost=1.0,
        nucleus_top_p=args.nucleus_top_p,
    )
    keep_indices = keep_indices.to(torch.long)
    retained_flags = {
        idx: bool((keep_indices == idx).any()) for idx in sentinel_indices
    }
    print(
        f"[sample] Retained {sum(retained_flags.values())}/{len(sentinel_indices)} sentinel anchors "
        f"(chunk_len={chunk_len}, buffer_tokens={buffer_tokens})."
    )
    for idx in sentinel_indices:
        start = max(0, idx - 5)
        end = min(len(tokens), idx + 6)
        snippet = " ".join(tokens[start:end])
        flag = "✓" if retained_flags[idx] else "✗"
        print(f"  [{flag}] idx={idx}: {snippet}")


def _needle_cluster_test(args: argparse.Namespace) -> None:
    chunk_origin = args.chunk_len * 2
    sentinels = [chunk_origin + offset for offset in range(0, 100, 10)]
    keep_indices, _ = _run_pipeline(
        seq_len=args.seq_len,
        d_model=args.d_model,
        chunk_len=args.chunk_len,
        per_chunk_budget=max(1, args.per_chunk_budget // 2),
        buffer_tokens=max(args.buffer_tokens, len(sentinels)),
        sentinel_idx=sentinels,
        base_boost=3.0,
        proto_boost=0.0,
        nucleus_top_p=args.nucleus_top_p,
    )
    retained = sum(bool((keep_indices == idx).any()) for idx in sentinels)
    recall = retained / len(sentinels)
    print(
        f"[cluster] needles kept {retained}/{len(sentinels)} "
        f"(recall={recall:.2f}) with per_chunk_budget={max(1, args.per_chunk_budget // 2)}"
    )

    adaptive_margin = args.cluster_adaptive_margin
    if adaptive_margin is not None and adaptive_margin >= 0.0:
        keep_indices_adapt, _ = _run_pipeline(
            seq_len=args.seq_len,
            d_model=args.d_model,
            chunk_len=args.chunk_len,
            per_chunk_budget=max(1, args.per_chunk_budget // 2),
            buffer_tokens=max(args.buffer_tokens, len(sentinels)),
            sentinel_idx=sentinels,
            base_boost=3.0,
            proto_boost=0.0,
            adaptive_margin=adaptive_margin,
            max_chunk_extra=args.cluster_max_extra,
            nucleus_top_p=args.nucleus_top_p,
        )
        retained_adapt = sum(bool((keep_indices_adapt == idx).any()) for idx in sentinels)
        recall_adapt = retained_adapt / len(sentinels)
        print(
            f"[cluster-adapt] needles kept {retained_adapt}/{len(sentinels)} "
            f"(recall={recall_adapt:.2f}) margin={adaptive_margin} max_extra={args.cluster_max_extra}"
        )


def _fading_signal_test(
    args: argparse.Namespace,
    boosts: Sequence[float],
    trials: int,
    sentinel_idx: int,
) -> None:
    print("[fading] sensitivity sweep:")
    for boost in boosts:
        hits = 0
        for seed in range(trials):
            keep_indices, _ = _run_pipeline(
                seq_len=args.seq_len,
                d_model=args.d_model,
                chunk_len=args.chunk_len,
                per_chunk_budget=args.per_chunk_budget,
                buffer_tokens=args.buffer_tokens,
                sentinel_idx=sentinel_idx,
                base_boost=boost,
                proto_boost=0.0,
                rng_seed=seed,
                nucleus_top_p=args.nucleus_top_p,
            )
            if bool((keep_indices == sentinel_idx).any()):
                hits += 1
        rate = hits / trials
        print(f"  boost={boost:.2f} -> recall {rate:.2%} ({hits}/{trials})")


def _jigsaw_puzzle_test(args: argparse.Namespace) -> None:
    early = args.chunk_len // 2
    late = args.seq_len - args.chunk_len // 3
    sentinels = [early, late]
    keep_indices, _ = _run_pipeline(
        seq_len=args.seq_len,
        d_model=args.d_model,
        chunk_len=args.chunk_len,
        per_chunk_budget=args.per_chunk_budget,
        buffer_tokens=max(args.buffer_tokens, len(sentinels)),
        sentinel_idx=sentinels,
        base_boost=4.0,
        proto_boost=0.5,
        nucleus_top_p=args.nucleus_top_p,
    )
    both = all(bool((keep_indices == idx).any()) for idx in sentinels)
    print(f"[jigsaw] both clues retained? {both} | keep_indices={keep_indices.tolist()}")


def _live_fire_test(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[live-fire] skipping (transformers unavailable: {exc})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.live_fire_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
    model.eval()

    text = args.live_fire_text
    max_len = min(args.seq_len, getattr(tokenizer, "model_max_length", args.seq_len))
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", None)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    attentions = outputs.attentions[-1][0]  # (heads, T, T)
    importance = attentions.mean(dim=0).mean(dim=0)  # (T,)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = importance.numel()
    chunk_len = min(args.chunk_len, seq_len)

    model_stub = _MockModel()
    chunk_scores: List[torch.Tensor] = []
    chunk_indices: List[torch.Tensor] = []
    for start, end in _chunk_iter(seq_len, chunk_len):
        current_len = end - start
        imp_slice = importance[start:end].unsqueeze(0).to(torch.float32)
        model_stub._last_token_importance = imp_slice
        model_stub._last_token_mask = None
        promoted = _select_top_tokens(
            model_stub,
            start,
            per_chunk_budget=args.per_chunk_budget,
            nucleus_top_p=args.nucleus_top_p,
        )
        if promoted is None:
            promoted = _select_top_tokens_cpu(
                model_stub._last_token_importance,
                model_stub._last_token_mask,
                start,
                per_chunk_budget=args.per_chunk_budget,
                nucleus_top_p=args.nucleus_top_p,
            )
        if promoted is None:
            continue
        scores_tensor, indices_tensor = promoted
        chunk_scores.append(scores_tensor)
        chunk_indices.append(indices_tensor)

    if not chunk_scores:
        print("[live-fire] no tokens promoted.")
        return

    keep_indices = _finalise_indices(
        torch.cat(chunk_scores),
        torch.cat(chunk_indices),
        max_tokens=min(args.buffer_tokens, seq_len),
    )
    retained_tokens = [tokens[idx] for idx in keep_indices.tolist()]
    print(f"[live-fire] retained {len(retained_tokens)} tokens:")
    print("  " + " ".join(retained_tokens))


def _runner_showcase(args: argparse.Namespace) -> None:
    if not args.runner_demo:
        return

    try:
        device = _resolve_device(args.demo_device)
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - invalid device
        print(f"[runner-demo] Unable to resolve device '{args.demo_device}': {exc}")
        return

    seq_len = int(args.demo_seq_len or args.seq_len)
    d_model = int(args.demo_d_model or args.d_model)
    chunk_len = min(int(args.chunk_len), seq_len)
    if chunk_len > MAX_SHORTLIST_CHUNK_TOKENS:
        print(
            f"[runner-demo] chunk_len {chunk_len:,} exceeds MAX_SHORTLIST_CHUNK_TOKENS "
            f"({MAX_SHORTLIST_CHUNK_TOKENS:,}); using the cap."
        )
        chunk_len = MAX_SHORTLIST_CHUNK_TOKENS

    buffer_tokens = max(
        args.buffer_tokens,
        int(seq_len * float(args.demo_buffer_ratio)),
    )
    per_chunk_budget = max(args.per_chunk_budget, args.demo_per_chunk_budget)
    sentinels = _sentinel_positions(seq_len, args.demo_sentinel_count)
    base_sequence = _build_demo_sequence(
        seq_len=seq_len,
        d_model=d_model,
        sentinels=sentinels,
        boost=args.demo_signal_boost,
        width=args.demo_signal_width,
        seed=args.demo_seed,
    )

    alphas = args.demo_alphas or [0.0, 0.35, 1.0]
    results: List[dict] = []
    ram_limit = int(args.demo_ram_limit_mb * 1024 * 1024) if args.demo_ram_limit_mb else None

    print(
        f"[runner-demo] seq_len={seq_len:,} chunk_len={chunk_len:,} buffer={buffer_tokens:,} "
        f"per_chunk_budget={per_chunk_budget:,} heads={args.demo_heads} device={device} "
        f"sentinels={sentinels}"
    )

    for alpha in alphas:
        cfg = ChunkedShortlistConfig(
            seq_len=seq_len,
            d_model=d_model,
            chunk_len=chunk_len,
            buffer_tokens=buffer_tokens,
            per_chunk_budget=per_chunk_budget,
            device=device,
            heads=args.demo_heads,
            nucleus_top_p=args.nucleus_top_p,
            chunk_sparse_ratio=args.demo_chunk_sparse_ratio,
            final_sparse_ratio=args.demo_final_sparse_ratio,
            seed=args.demo_seed,
            report_latency=args.demo_report_latency,
            progress=args.demo_progress,
            run_final_pass=not args.demo_skip_final,
            shortlist_alpha=float(alpha),
            storage=args.demo_storage,
            temp_dir=args.demo_temp_dir,
            ram_limit_bytes=ram_limit,
        )
        runner = ChunkedShortlistRunner(cfg)
        try:
            outcome = runner.run(sequence=base_sequence.clone())
        except RuntimeError as exc:  # pragma: no cover - device/runtime issues
            print(f"[runner-demo] alpha={alpha:.3f} failed: {exc}")
            continue

        metrics = outcome.metrics
        keep_indices = outcome.keep_indices.detach().cpu()
        hits = sum(int((keep_indices == idx).any().item()) for idx in sentinels)
        recall = hits / len(sentinels)
        entry = {
            "alpha": float(alpha),
            "metrics": metrics,
            "sentinel_hits": hits,
            "sentinel_recall": recall,
            "backend": outcome.backend_info,
            "storage_reason": metrics.storage_reason,
        }
        results.append(entry)

    if not results:
        print("[runner-demo] No successful runs recorded.")
        return

    header = (
        f"{'alpha':>7} {'retain':>10} {'ratio':>8} "
        f"{'chunk':>9} {'final':>9} {'total':>9} {'tok/s':>9} {'VRAM':>9} {'sentinel':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in results:
        metrics = row["metrics"]
        sentinel_pct = f"{row['sentinel_recall']*100.0:6.2f}%"
        print(
            f"{row['alpha']:7.3f} "
            f"{metrics.retained_tokens:10d} "
            f"{metrics.retention_ratio:8.4f} "
            f"{_fmt_ms(metrics.chunk_time_ms):>9} "
            f"{_fmt_ms(metrics.final_time_ms):>9} "
            f"{_fmt_ms(metrics.total_time_ms):>9} "
            f"{_fmt_rate(metrics.total_tokens_per_s):>9} "
            f"{_fmt_mem(metrics.peak_memory_mb):>9} "
            f"{sentinel_pct:>10}"
        )
    print(f"\n[runner-demo] storage={results[-1]['metrics'].storage_mode}")
    if results[-1]["storage_reason"]:
        print(f"[runner-demo] storage reason: {results[-1]['storage_reason']}")
    backend = results[-1].get("backend")
    if backend:
        print(f"[runner-demo] backend: {backend}")

    if args.demo_report is not None:
        payload = {
            "config": {
                "seq_len": seq_len,
                "d_model": d_model,
                "chunk_len": chunk_len,
                "buffer_tokens": buffer_tokens,
                "per_chunk_budget": per_chunk_budget,
                "device": str(device),
                "sentinels": sentinels,
            },
            "results": [
                {
                    "alpha": row["alpha"],
                    "sentinel_hits": row["sentinel_hits"],
                    "sentinel_recall": row["sentinel_recall"],
                    "metrics": asdict(row["metrics"]),
                    "backend": row["backend"],
                    "storage_reason": row["storage_reason"],
                }
                for row in results
            ],
        }
        args.demo_report.parent.mkdir(parents=True, exist_ok=True)
        args.demo_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[runner-demo] Saved report to {args.demo_report}")


def run_checks(args: argparse.Namespace) -> None:
    seq_len = args.seq_len
    chunk_len = args.chunk_len
    sentinel_idx = max(0, min(args.sentinel_index, max(seq_len - 1, 0)))
    if sentinel_idx != args.sentinel_index and args.verbose:
        print(
            f"[warn] adjusted sentinel index from {args.sentinel_index} to {sentinel_idx} to fit sequence length."
        )
    per_chunk_budget = args.per_chunk_budget
    buffer_tokens = args.buffer_tokens

    print("=== Needle recall (high router score) ===")
    keep_indices, logs = _run_pipeline(
        seq_len=seq_len,
        d_model=args.d_model,
        chunk_len=chunk_len,
        per_chunk_budget=per_chunk_budget,
        buffer_tokens=buffer_tokens,
        sentinel_idx=sentinel_idx,
        base_boost=5.0,
        proto_boost=0.0,
        nucleus_top_p=args.nucleus_top_p,
    )
    if args.verbose:
        print(_format_logs(logs))
    retained = bool((keep_indices == sentinel_idx).any())
    print(f"retained sentinel? {retained} | keep_indices={keep_indices.tolist()}")

    print("\n=== Proto teleportation hypothesis ===")
    keep_no_proto, logs_no_proto = _run_pipeline(
        seq_len=seq_len,
        d_model=args.d_model,
        chunk_len=chunk_len,
        per_chunk_budget=per_chunk_budget,
        buffer_tokens=buffer_tokens,
        sentinel_idx=sentinel_idx,
        base_boost=-0.2,
        proto_boost=0.0,
        nucleus_top_p=args.nucleus_top_p,
    )
    keep_with_proto, logs_with_proto = _run_pipeline(
        seq_len=seq_len,
        d_model=args.d_model,
        chunk_len=chunk_len,
        per_chunk_budget=per_chunk_budget,
        buffer_tokens=buffer_tokens,
        sentinel_idx=sentinel_idx,
        base_boost=-0.2,
        proto_boost=1.0,
        nucleus_top_p=args.nucleus_top_p,
    )
    if args.verbose:
        print("-- without Proto boost --")
        print(_format_logs(logs_no_proto))
        print("-- with Proto boost --")
        print(_format_logs(logs_with_proto))
    no_proto_retained = bool((keep_no_proto == sentinel_idx).any())
    proto_retained = bool((keep_with_proto == sentinel_idx).any())
    print(
        f"without Proto retained? {no_proto_retained} | with Proto retained? {proto_retained}"
    )

    print("\n=== Ordering sanity (RoPE alignment) ===")
    monotonic = bool(torch.all(keep_indices[:-1] <= keep_indices[1:])) if keep_indices.numel() > 1 else True
    print(f"keep_indices monotonic? {monotonic}")

    print("\n=== Needle cluster stress test ===")
    _needle_cluster_test(args)

    print("\n=== Fading signal sweep ===")
    _fading_signal_test(
        args,
        boosts=args.fading_boosts,
        trials=args.fading_trials,
        sentinel_idx=sentinel_idx,
    )

    print("\n=== Jigsaw puzzle synthesis ===")
    _jigsaw_puzzle_test(args)

    if args.enable_live_fire:
        print("\n=== Live fire integration ===")
        _live_fire_test(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic validation checks for the chunked Shortlist demo."
    )
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--buffer-tokens", type=int, default=8)
    parser.add_argument("--per-chunk-budget", type=int, default=2)
    parser.add_argument("--sentinel-index", type=int, default=5231)
    parser.add_argument(
        "--nucleus-top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) filter threshold when selecting chunk tokens.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip the synthetic unit tests and only run the showcase demos.",
    )
    parser.add_argument(
        "--fading-boosts",
        type=float,
        nargs="+",
        default=[5.0, 2.0, 1.0, 0.5, 0.25, 0.1],
        help="Boost values to sweep in the fading-signal sensitivity test.",
    )
    parser.add_argument(
        "--fading-trials",
        type=int,
        default=20,
        help="Number of random seeds per boost level for the fading-signal test.",
    )
    parser.add_argument(
        "--enable-live-fire",
        action="store_true",
        help="Run the optional live-fire integration test using a pretrained transformer.",
    )
    parser.add_argument(
        "--live-fire-model",
        default="distilbert-base-uncased",
        help="Model name for the live-fire integration test.",
    )
    parser.add_argument(
        "--live-fire-text",
        default=(
            "The Titan vault code is 1234. "
            "Later, security confirmed the Titan vault remains sealed. "
            "Additional context discusses unrelated topics and background information."
        ),
        help="Input text used by the live-fire integration test.",
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help=f"Path to a text file for the sample data demo (default: {DEFAULT_SAMPLE_FILE}).",
    )
    parser.add_argument(
        "--sample-text",
        type=str,
        default=(
            "The Titan vault code is 1234. Later, security confirmed the Titan vault remains sealed. "
            "Additional context discusses unrelated topics and background information."
        ),
        help="Inline text used for the sample demo when no --sample-file is supplied.",
    )
    parser.add_argument(
        "--sample-needle",
        type=str,
        default="Titan vault code 9876",
        help="Phrase to track within the sample text (defaults to 'Titan vault code 9876').",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional limit on the number of tokens loaded from the sample text.",
    )
    parser.add_argument(
        "--sample-casefold",
        action="store_true",
        help="Enable case-insensitive matching for the sample needle phrase.",
    )
    parser.add_argument(
        "--cluster-adaptive-margin",
        type=float,
        default=0.1,
        help="Adaptive margin for the cluster test (set negative to disable).",
    )
    parser.add_argument(
        "--cluster-max-extra",
        type=int,
        default=8,
        help="Maximum additional tokens a bursty chunk may promote in the cluster test.",
    )
    parser.add_argument(
        "--runner-demo",
        action="store_true",
        help="Run a full ChunkedShortlistRunner demo using synthetic embeddings.",
    )
    parser.add_argument("--demo-device", default="auto", help="Device used for the runner demo (default: auto).")
    parser.add_argument("--demo-seq-len", type=int, default=262_144, help="Sequence length for the runner demo.")
    parser.add_argument("--demo-d-model", type=int, default=None, help="Override d_model for the runner demo.")
    parser.add_argument(
        "--demo-heads",
        type=int,
        default=8,
        help="Attention heads used by the runner demo.",
    )
    parser.add_argument(
        "--demo-buffer-ratio",
        type=float,
        default=0.02,
        help="Buffer ratio (fraction of seq_len) retained during the runner demo.",
    )
    parser.add_argument(
        "--demo-per-chunk-budget",
        type=int,
        default=4_096,
        help="Per-chunk shortlist budget for the runner demo.",
    )
    parser.add_argument(
        "--demo-sentinel-count",
        type=int,
        default=4,
        help="Number of synthetic sentinel positions injected into the runner demo.",
    )
    parser.add_argument(
        "--demo-signal-boost",
        type=float,
        default=6.0,
        help="Magnitude added to sentinel token embeddings in the runner demo.",
    )
    parser.add_argument(
        "--demo-signal-width",
        type=int,
        default=16,
        help="Embedding width influenced by the sentinel boost.",
    )
    parser.add_argument(
        "--demo-seed",
        type=int,
        default=123,
        help="Random seed for the runner demo synthetic embeddings.",
    )
    parser.add_argument(
        "--demo-chunk-sparse-ratio",
        type=float,
        default=0.05,
        help="Chunk-stage sparsity ratio for the runner demo.",
    )
    parser.add_argument(
        "--demo-final-sparse-ratio",
        type=float,
        default=0.5,
        help="Final-pass sparsity ratio for the runner demo.",
    )
    parser.add_argument(
        "--demo-alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.35, 1.0],
        help="Shortlist alpha settings swept during the runner demo.",
    )
    parser.add_argument(
        "--demo-progress",
        action="store_true",
        help="Enable tqdm progress bars during the runner demo.",
    )
    parser.add_argument(
        "--demo-report-latency",
        action="store_true",
        help="Capture CUDA event timings during the runner demo.",
    )
    parser.add_argument(
        "--demo-skip-final",
        action="store_true",
        help="Skip the final attention pass in the runner demo.",
    )
    parser.add_argument(
        "--demo-storage",
        choices=["auto", "cpu", "disk"],
        default="auto",
        help="Staging location for the runner demo sequence.",
    )
    parser.add_argument(
        "--demo-temp-dir",
        type=Path,
        default=None,
        help="Scratch directory for disk-backed runner demo staging.",
    )
    parser.add_argument(
        "--demo-ram-limit-mb",
        type=int,
        default=None,
        help="RAM limit (MB) before spilling runner demo data to disk.",
    )
    parser.add_argument(
        "--demo-report",
        type=Path,
        default=None,
        help="Optional JSON file summarising runner demo metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_synthetic:
        keep_params = args.chunk_len * args.buffer_tokens / max(args.seq_len, 1)
        print(
            f"config: seq_len={args.seq_len:,} chunk_len={args.chunk_len:,} "
            f"buffer_tokens={args.buffer_tokens:,} per_chunk_budget={args.per_chunk_budget:,} "
            f"(keep ratio {keep_params:.3f})"
        )
        run_checks(args)
    else:
        print("[info] Skipping synthetic shortlist unit tests (--skip-synthetic).")
    if args.sample_file is not None or (args.sample_text and args.sample_text.strip()):
        print("\n=== Sample text demo ===")
        _sample_text_demo(args)
    if args.enable_live_fire:
        print("\n=== Live fire integration ===")
        _live_fire_test(args)
    if args.runner_demo:
        print("\n=== Chunked Shortlist runner showcase ===")
        _runner_showcase(args)


if __name__ == "__main__":
    main()
