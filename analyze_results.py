#!/usr/bin/env python3
"""Compatibility analyzer for experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    representatives: dict[str, Path] = {}
    for run in sorted(args.results.glob("run_*")):
        for condition in ("chatbot", "agent"):
            directory = run / condition
            summary = directory / "summary.csv"
            if not summary.exists():
                continue
            representatives.setdefault(condition, directory)
            for row in read(summary):
                latency = float(row["latency_s"])
                average = float(row["avg_process_cpu_percent"])
                values = {
                    "latency_s": latency,
                    "avg_process_cpu_percent": average,
                    "peak_process_cpu_percent": float(
                        row["peak_process_cpu_percent"]
                    ),
                    "cpu_time_s": average * latency / 100,
                    "llm_calls": float(row["llm_calls"]),
                    "tool_calls": float(row["tool_calls"]),
                }
                for name, value in values.items():
                    metrics[condition][name].append(value)
    if not metrics:
        raise SystemExit("No run_* results found")
    aggregate = {}
    for condition, condition_metrics in metrics.items():
        aggregate[condition] = {}
        for name, values in condition_metrics.items():
            aggregate[condition][name] = {
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0,
                "n": len(values),
            }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "aggregate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "metric", "mean", "sd", "n"])
        for condition, condition_metrics in aggregate.items():
            for name, value in condition_metrics.items():
                writer.writerow(
                    [condition, name, value["mean"], value["sd"], value["n"]]
                )
    colors = {
        "context_build": "#8dd3c7",
        "llm_call": "#80b1d3",
        "agent_parse": "#fdb462",
        "tool_call": "#fb8072",
        "finalize": "#bebada",
    }
    for condition, directory in representatives.items():
        request_id = read(directory / "summary.csv")[0]["request_id"]
        samples = [
            row for row in read(directory / "cpu_samples.csv")
            if row["request_id"] == request_id
        ]
        events = [
            row for row in read(directory / "events.csv")
            if row["request_id"] == request_id and row["event_type"] != "request"
        ]
        figure, axis = plt.subplots(figsize=(11, 4.5))
        axis.plot(
            [float(row["elapsed_s"]) for row in samples],
            [float(row["process_cpu_percent"]) for row in samples],
            color="black",
            linewidth=1.3,
        )
        seen: set[str] = set()
        for event in events:
            name = event["event_type"]
            axis.axvspan(
                float(event["start_s"]),
                float(event["end_s"]),
                color=colors.get(name, "#cccccc"),
                alpha=0.3,
                label=name if name not in seen else None,
            )
            seen.add(name)
        axis.set(
            title=f"{condition} CPU timeline",
            xlabel="Elapsed request time (s)",
            ylabel="Client process-tree CPU (%)",
        )
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
        axis.legend(ncol=3, fontsize=8)
        figure.tight_layout()
        figure.savefig(args.output / f"{condition}_timeline.png", dpi=180)
        plt.close(figure)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
