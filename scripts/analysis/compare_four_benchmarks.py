#!/usr/bin/env python3
"""Create one compact comparison table and figure for all four benchmarks."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOTS = {
    "MATH": Path("results_math_metrics_10ms"),
    "HotpotQA": Path("results_hotpotqa_metrics_10ms"),
    "HumanEval": Path("results_humaneval_10ms"),
    "WebShop": Path("results_webshop_10ms"),
}
METRICS = ["total_tokens", "peak_kv_cache_usage_fraction", "avg_process_cpu_percent", "latency_s"]
KV_CACHE_BYTES = 25_769_803_776
LOGICAL_CPUS = 64


def main() -> None:
    output = Path("results_comparison_10ms")
    output.mkdir(exist_ok=True)
    rows = []
    for benchmark, root in ROOTS.items():
        aggregate = json.loads((root / "analysis/aggregate_extended.json").read_text())
        for condition in ("chatbot", "agent"):
            row = {"benchmark": benchmark, "condition": condition}
            row.update({metric: aggregate[condition][metric]["mean"] for metric in METRICS})
            row["mean_peak_kv_mib"] = row["peak_kv_cache_usage_fraction"] * KV_CACHE_BYTES / 2**20
            row["mean_peak_kv_gib"] = row["peak_kv_cache_usage_fraction"] * KV_CACHE_BYTES / 2**30
            row["mean_client_cpu_total_percent"] = row["avg_process_cpu_percent"] / LOGICAL_CPUS
            rows.append(row)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_metrics = ["total_tokens", "peak_kv_cache_usage_fraction", "mean_client_cpu_total_percent", "latency_s"]
    titles = ["Mean total tokens", "Mean request peak KV cache", "Mean client CPU (64 logical CPUs)", "Mean latency"]
    labels = ["tokens", "KV cache (MiB)", "Total CPU capacity (%)", "seconds"]
    for axis, metric, title, label in zip(axes.flat, plot_metrics, titles, labels):
        factor = KV_CACHE_BYTES / 2**20 if metric == "peak_kv_cache_usage_fraction" else 1
        x = range(len(ROOTS)); width = 0.36
        for offset, condition in ((-width / 2, "chatbot"), (width / 2, "agent")):
            values = [next(r[metric] for r in rows if r["benchmark"] == b and r["condition"] == condition) * factor for b in ROOTS]
            axis.bar([i + offset for i in x], values, width, label=condition)
        axis.set_xticks(list(x), ROOTS); axis.set_title(title); axis.set_ylabel(label)
    axes[0, 0].legend(); figure.tight_layout()
    figure.savefig(output / "four_benchmarks.png", dpi=200)
    figure.savefig(output / "four_benchmarks.pdf")


if __name__ == "__main__":
    main()
