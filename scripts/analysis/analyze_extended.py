#!/usr/bin/env python3
"""Aggregate CPU, token, KV-cache, and benchmark-specific summary metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


IDENTITY = {"request_id", "condition"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for summary in sorted(args.results.glob("run_*/*/summary.csv")):
        with summary.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                condition = row["condition"]
                for key, raw in row.items():
                    if key in IDENTITY or raw == "":
                        continue
                    if raw in ("True", "False"):
                        values[condition][key].append(float(raw == "True"))
                        continue
                    try:
                        values[condition][key].append(float(raw))
                    except ValueError:
                        pass
    if not values:
        raise SystemExit("No run_* summaries found")
    aggregate = {
        condition: {
            metric: {
                "mean": statistics.fmean(rows),
                "median": statistics.median(rows),
                "sd": statistics.stdev(rows) if len(rows) > 1 else 0.0,
                "n": len(rows),
            }
            for metric, rows in metrics.items()
        }
        for condition, metrics in values.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "aggregate_extended.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "aggregate_extended.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "metric", "mean", "median", "sd", "n"])
        for condition, metrics in aggregate.items():
            for metric, stats in metrics.items():
                writer.writerow(
                    [
                        condition,
                        metric,
                        stats["mean"],
                        stats["median"],
                        stats["sd"],
                        stats["n"],
                    ]
                )

    conditions = list(aggregate)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    token_metrics = ("prompt_tokens", "completion_tokens")
    bottom = [0.0] * len(conditions)
    for metric in token_metrics:
        heights = [aggregate[c].get(metric, {}).get("mean", 0.0) for c in conditions]
        axes[0].bar(conditions, heights, bottom=bottom, label=metric)
        bottom = [a + b for a, b in zip(bottom, heights)]
    axes[0].set(title="Mean token length per request", ylabel="Tokens")
    axes[0].legend()
    kv = [
        aggregate[c]
        .get("peak_kv_cache_usage_fraction", {})
        .get("mean", 0.0)
        * 100
        for c in conditions
    ]
    axes[1].bar(conditions, kv)
    axes[1].set(
        title="Mean per-request peak KV-cache usage",
        ylabel="Allocated KV blocks (%)",
    )
    figure.tight_layout()
    figure.savefig(args.output / "tokens_kv_summary.png", dpi=200)
    figure.savefig(args.output / "tokens_kv_summary.pdf")
    plt.close(figure)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
