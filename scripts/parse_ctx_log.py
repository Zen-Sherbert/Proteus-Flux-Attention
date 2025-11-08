#!/usr/bin/env python
"""Parse training comparison logs into CSV (and optional plot).

The script expects logs produced by ``aspa_train.py`` where each section is
headed by lines like ``=== Scenario ===`` and training summaries include the
standard "Training finished" + "summary" lines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCENARIO_RE = re.compile(r"^===\s+(?P<label>.+?)\s+===\s*$")
FINISH_RE = re.compile(r"^Training finished in (?P<elapsed>[0-9.]+)s \((?P<steps>\d+) steps\).$")
SUMMARY_RE = re.compile(
    r"^(?P<model>ASPA|Standard Attention) summary: "
    r"steps=(?P<steps>\d+), avg_loss=(?P<loss>[0-9.]+)"  # noqa: E501
)
AUTO_RE = re.compile(r"^\[AutoLog\]\s+(?P<payload>\{.+\})\s*$")


@dataclass
class ModelRecord:
    model: str
    steps: int
    avg_loss: float
    elapsed_sec: Optional[float]

    @property
    def steps_per_sec(self) -> Optional[float]:
        if self.elapsed_sec is None or self.elapsed_sec <= 0:
            return None
        return self.steps / self.elapsed_sec


@dataclass
class AutoLogRecord:
    scenario: Optional[str]
    payload: Dict[str, object]


def parse_log(lines: Iterable[str]) -> Tuple[Dict[str, Dict[str, ModelRecord]], List[AutoLogRecord]]:
    scenarios: Dict[str, Dict[str, ModelRecord]] = {}
    current_label: Optional[str] = None
    pending_finish: Optional[dict] = None
    auto_logs: List[AutoLogRecord] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        match = SCENARIO_RE.match(line)
        if match:
            current_label = match.group("label")
            scenarios.setdefault(current_label, {})
            pending_finish = None
            continue

        finish = FINISH_RE.match(line)
        if finish:
            pending_finish = {
                "elapsed": float(finish.group("elapsed")),
                "steps": int(finish.group("steps")),
            }
            continue

        summary = SUMMARY_RE.match(line)
        if summary and current_label is not None:
            model_name = summary.group("model")
            key = "aspa" if "ASPA" in model_name else "standard"
            steps = int(summary.group("steps"))
            loss = float(summary.group("loss"))
            elapsed = pending_finish["elapsed"] if pending_finish else None
            record = ModelRecord(model=model_name, steps=steps, avg_loss=loss, elapsed_sec=elapsed)
            scenarios.setdefault(current_label, {})[key] = record
            pending_finish = None
            continue

        auto = AUTO_RE.match(line)
        if auto:
            try:
                payload = json.loads(auto.group("payload"))
            except json.JSONDecodeError:
                continue
            entry = AutoLogRecord(scenario=current_label, payload=payload)
            auto_logs.append(entry)
            continue

    return scenarios, auto_logs


def write_csv(path: Path, scenarios: Dict[str, Dict[str, ModelRecord]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["scenario", "model", "steps", "avg_loss", "elapsed_sec", "steps_per_sec"])
        for label, records in scenarios.items():
            for key, record in records.items():
                rate = record.steps_per_sec
                writer.writerow([
                    label,
                    key,
                    record.steps,
                    f"{record.avg_loss:.6f}",
                    f"{record.elapsed_sec:.3f}" if record.elapsed_sec is not None else "",
                    f"{rate:.3f}" if rate is not None else "",
                ])


def write_auto_csv(path: Path, records: List[AutoLogRecord]) -> None:
    if not records:
        return
    fieldnames = [
        "scenario",
        "model",
        "variant",
        "label",
        "seq_len",
        "batch_size",
        "mode_setting",
        "mode_selected",
        "backend",
        "linear_backend",
        "latency_ms",
        "throughput_seq_s",
        "tokens_per_s",
        "token_fraction",
        "active_k",
        "quantized",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        for entry in records:
            payload = entry.payload
            row: List[object] = []
            for field in fieldnames:
                if field == "scenario":
                    value = entry.scenario
                else:
                    value = payload.get(field)
                row.append("" if value is None else value)
            writer.writerow(row)


def maybe_plot(path: Path, scenarios: Dict[str, Dict[str, ModelRecord]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Warning: matplotlib unavailable ({exc}); skipping plot.", file=sys.stderr)
        return

    labels = list(scenarios.keys())
    if not labels:
        print("No scenarios to plot.", file=sys.stderr)
        return

    aspa_vals: List[float] = []
    s_vals: List[float] = []
    for label in labels:
        d_elapsed = scenarios[label].get("aspa")
        s_elapsed = scenarios[label].get("standard")
        aspa_vals.append(d_elapsed.elapsed_sec if d_elapsed and d_elapsed.elapsed_sec is not None else math.nan)
        s_vals.append(s_elapsed.elapsed_sec if s_elapsed and s_elapsed.elapsed_sec is not None else math.nan)

    width = 0.35
    xs = range(len(labels))

    plt.figure(figsize=(max(6, len(labels) * 2.5), 4))
    plt.bar([x - width / 2 for x in xs], aspa_vals, width, label="ASPA")
    plt.bar([x + width / 2 for x in xs], s_vals, width, label="Standard")
    plt.ylabel("Elapsed seconds (20k steps)")
    plt.title("ASPA vs Standard Training Time")
    plt.xticks(list(xs), labels, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Parse ASPA vs dense comparison logs into CSV/plot.")
    parser.add_argument("log", type=Path, help="Path to the text log file.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Where to write the CSV summary.")
    parser.add_argument("--plot-out", type=Path, default=None, help="Optional PNG path for a runtime bar chart.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON dump of parsed data.")
    parser.add_argument("--auto-csv-out", type=Path, default=None, help="Optional CSV path for parsed auto-mode logs.")
    parser.add_argument("--auto-json-out", type=Path, default=None, help="Optional JSON dump for auto-mode logs.")
    args = parser.parse_args(argv)

    if not args.log.exists():
        parser.error(f"Log file {args.log} does not exist")

    scenarios, auto_logs = parse_log(args.log.read_text(encoding="utf-8").splitlines())
    if not scenarios:
        print("No scenarios found in log.", file=sys.stderr)
        # Still allow auto logs to be exported even if scenarios absent.
        if not auto_logs:
            return 1

    if args.csv_out:
        write_csv(args.csv_out, scenarios)
        print(f"CSV written to {args.csv_out}")

    if args.json_out:
        payload: Dict[str, object] = {
            label: {
                key: {
                    "model": record.model,
                    "steps": record.steps,
                    "avg_loss": record.avg_loss,
                    "elapsed_sec": record.elapsed_sec,
                    "steps_per_sec": record.steps_per_sec,
                }
                for key, record in records.items()
            }
            for label, records in scenarios.items()
        }
        if auto_logs:
            payload["auto_logs"] = [
                {
                    "scenario": entry.scenario,
                    **entry.payload,
                }
                for entry in auto_logs
            ]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"JSON written to {args.json_out}")

    if args.plot_out:
        maybe_plot(args.plot_out, scenarios)
        print(f"Plot written to {args.plot_out}")

    if args.auto_csv_out and auto_logs:
        write_auto_csv(args.auto_csv_out, auto_logs)
        print(f"Auto-mode CSV written to {args.auto_csv_out}")

    if args.auto_json_out and auto_logs:
        auto_payload = [
            {
                "scenario": entry.scenario,
                **entry.payload,
            }
            for entry in auto_logs
        ]
        args.auto_json_out.parent.mkdir(parents=True, exist_ok=True)
        args.auto_json_out.write_text(json.dumps(auto_payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Auto-mode JSON written to {args.auto_json_out}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
