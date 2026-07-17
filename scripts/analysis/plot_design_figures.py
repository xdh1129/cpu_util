#!/usr/bin/env python3
"""Render the timeline figures specified in the experiment design."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


RESULTS = Path("results/run_001")
OUTPUT = Path("results/analysis")
PHASE_COLORS = {
    "context_build": "#59a14f",
    "llm_call": "#4e79a7",
    "agent_parse": "#f28e2b",
    "tool_call": "#e15759",
    "finalize": "#b07aa1",
}
PHASE_LABELS = {
    "context_build": "Context build",
    "llm_call": "LLM call / API wait",
    "agent_parse": "Agent reasoning / parsing",
    "tool_call": "Tool execution",
    "finalize": "Final answer formatting",
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def request_data(condition: str) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    directory = RESULTS / condition
    request_id = read(directory / "summary.csv")[0]["request_id"]
    samples = [
        row
        for row in read(directory / "cpu_samples.csv")
        if row["request_id"] == request_id
    ]
    events = [
        row
        for row in read(directory / "events.csv")
        if row["request_id"] == request_id
    ]
    return request_id, samples, events


def draw_timeline(axis: plt.Axes, condition: str) -> None:
    request_id, samples, events = request_data(condition)
    x = [float(row["elapsed_s"]) for row in samples]
    y = [float(row["process_cpu_percent"]) for row in samples]
    request = next(row for row in events if row["event_type"] == "request")
    start = float(request["start_s"])
    end = float(request["end_s"])

    for event in events:
        phase = event["event_type"]
        if phase == "request":
            continue
        axis.axvspan(
            float(event["start_s"]),
            float(event["end_s"]),
            color=PHASE_COLORS.get(phase, "#cccccc"),
            alpha=0.24,
            linewidth=0,
        )
    axis.plot(x, y, color="#202020", linewidth=1.6, zorder=3)
    axis.axvline(start, color="#2ca02c", linestyle="--", linewidth=1.4)
    axis.axvline(end, color="#d62728", linestyle="--", linewidth=1.4)
    axis.set_title(f"{condition.capitalize()} — {request_id}", loc="left")
    axis.set_xlabel("Elapsed request time (seconds)")
    axis.set_ylabel("Client process-tree CPU utilization (%)")
    axis.set_xlim(left=0, right=max(x + [end]) * 1.02)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.2)


def legend_handles() -> list[object]:
    handles: list[object] = [
        Line2D([0], [0], color="#202020", linewidth=1.6, label="CPU utilization"),
        Line2D([0], [0], color="#2ca02c", linestyle="--", label="Request start"),
        Line2D([0], [0], color="#d62728", linestyle="--", label="Request end"),
    ]
    for phase, color in PHASE_COLORS.items():
        handles.append(Patch(facecolor=color, alpha=0.24, label=PHASE_LABELS[phase]))
    return handles


def save_single(condition: str) -> None:
    figure, axis = plt.subplots(figsize=(12, 5.2))
    draw_timeline(axis, condition)
    axis.legend(handles=legend_handles(), loc="upper right", ncol=2, fontsize=8)
    figure.suptitle("Client-side CPU utilization during one complete request", fontsize=14)
    figure.tight_layout()
    figure.savefig(OUTPUT / f"{condition}_single_request_timeline.png", dpi=220)
    figure.savefig(OUTPUT / f"{condition}_single_request_timeline.pdf")
    plt.close(figure)


def save_comparison() -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharey=False)
    draw_timeline(axes[0], "chatbot")
    draw_timeline(axes[1], "agent")
    axes[0].set_xlabel("")
    figure.suptitle("Chatbot versus tool-augmented agent CPU timelines", fontsize=15)
    figure.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    figure.savefig(OUTPUT / "chatbot_agent_timeline_comparison.png", dpi=220)
    figure.savefig(OUTPUT / "chatbot_agent_timeline_comparison.pdf")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_single("chatbot")
    save_single("agent")
    save_comparison()


if __name__ == "__main__":
    main()
