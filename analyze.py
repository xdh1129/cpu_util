#!/usr/bin/env python3
"""Aggregate repeated runs and plot representative CPU timelines."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "context_build": "#8dd3c7",
    "llm_call": "#80b1d3",
    "agent_parse": "#fdb462",
    "tool_call": "#fb8072",
    "finalize": "#bebada",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot(directory: Path, request_id: str, output: Path) -> None:
    samples = [
        row for row in rows(directory / "cpu_samples.csv")
        if row["request_id"] == request_id
    ]
    events = [
        row for row in rows(directory / "events.csv")
        if row["request_id"] == request_id and row["event_type"] != "request"
    ]
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(
        [float(row["elapsed_s"]) for row in samples],
        [float(row["process_cpu_percent"]) for row in samples],
        color="#222222",
        linewidth=1.4,
    )
    seen: set[str] = set()
    for event in events:
        name = event["event_type"]
        axis.axvspan(
            float(event["start_s"]),
            float(event["end_s"]),
            color=COLORS.get(name, "#cccccc"),
            alpha=0.3,
            label=name if name not in seen else None,
        )
        seen.add(name)
    axis.set(
        title=f"{directory.name}: {request_id}",
        xlabel="Elapsed request time (s)",
        ylabel="Client process-tree CPU (%)",
    )
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--run-prefix", default="run_")
    parser.add_argument("--output", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    data: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    directories: dict[str, list[Path]] = defaultdict(list)
    for run in sorted(args.results_root.glob(f"{args.run_prefix}*")):
        for condition in ("chatbot", "agent"):
            directory = run / condition
            if not (directory / "summary.csv").exists():
                continue
            directories[condition].append(directory)
            for row in rows(directory / "summary.csv"):
                for metric in (
                    "latency_s",
                    "avg_process_cpu_percent",
                    "peak_process_cpu_percent",
                    "cpu_time_s",
                    "llm_calls",
                    "tool_calls",
                ):
                    data[condition][metric].append(float(row[metric]))
    if not data:
        raise SystemExit("No completed runs found")

    aggregate = {}
    for condition, metrics in data.items():
        aggregate[condition] = {}
        for metric, values in metrics.items():
            aggregate[condition][metric] = {
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
        for condition, metrics in aggregate.items():
            for metric, stats in metrics.items():
                writer.writerow(
                    [condition, metric, stats["mean"], stats["sd"], stats["n"]]
                )
    for condition, condition_dirs in directories.items():
        directory = condition_dirs[0]
        request_id = rows(directory / "summary.csv")[0]["request_id"]
        plot(directory, request_id, args.output / f"{condition}_timeline.png")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
