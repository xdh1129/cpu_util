#!/usr/bin/env python3
"""Run existing MATH/HotpotQA conditions with token and KV-cache metrics."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import statistics
import threading
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openai import OpenAI

import experiment as math_benchmark
import hotpot_experiment as hotpot_benchmark


class RecordingCompletions:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.responses: list[Any] = []

    def create(self, **kwargs: Any) -> Any:
        response = self.inner.create(**kwargs)
        self.responses.append(response)
        return response

    def reset(self) -> None:
        self.responses.clear()

    def usage(self) -> tuple[int, int]:
        prompt = sum(
            int(response.usage.prompt_tokens)
            for response in self.responses
            if response.usage is not None
        )
        completion = sum(
            int(response.usage.completion_tokens)
            for response in self.responses
            if response.usage is not None
        )
        return prompt, completion


class RecordingChat:
    def __init__(self, inner: Any) -> None:
        self.completions = RecordingCompletions(inner.completions)


class RecordingClient:
    def __init__(self, inner: OpenAI) -> None:
        self.chat = RecordingChat(inner.chat)


class KvCacheSampler(threading.Thread):
    METRICS = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")

    def __init__(
        self, request_id: str, origin: float, interval: float, metrics_url: str
    ) -> None:
        super().__init__(daemon=True)
        self.request_id = request_id
        self.origin = origin
        self.interval = interval
        self.metrics_url = metrics_url
        self.stopping = threading.Event()
        self.samples: list[dict[str, Any]] = []

    @classmethod
    def parse(cls, payload: str) -> float | None:
        for line in payload.splitlines():
            if line.startswith("#"):
                continue
            if any(
                line.startswith(metric + "{") or line.startswith(metric + " ")
                for metric in cls.METRICS
            ):
                return float(line.rsplit(maxsplit=1)[-1])
        return None

    def run(self) -> None:
        deadline = time.perf_counter()
        while not self.stopping.is_set():
            try:
                with urllib.request.urlopen(self.metrics_url, timeout=1) as response:
                    value = self.parse(response.read().decode("utf-8"))
                if value is not None:
                    self.samples.append(
                        {
                            "request_id": self.request_id,
                            "timestamp_s": time.time(),
                            "elapsed_s": time.perf_counter() - self.origin,
                            "kv_cache_usage_fraction": value,
                        }
                    )
            except (OSError, ValueError):
                pass
            deadline += self.interval
            delay = deadline - time.perf_counter()
            if delay <= 0:
                deadline = time.perf_counter()
                delay = 0
            self.stopping.wait(delay)

    def stop(self) -> None:
        self.stopping.set()
        self.join()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("math", "hotpotqa"), required=True)
    parser.add_argument("--condition", choices=("chatbot", "agent"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--kv-sample-interval", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    module = math_benchmark if args.benchmark == "math" else hotpot_benchmark
    runner = module.chatbot if args.condition == "chatbot" else module.agent
    items = [
        json.loads(line)
        for line in args.workload.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    client = RecordingClient(
        OpenAI(base_url=args.base_url, api_key="local-vllm")
    )
    metrics_url = args.base_url.removesuffix("/v1") + "/metrics"
    output = args.output_root / args.run_id / args.condition
    output.mkdir(parents=True, exist_ok=True)

    def invoke(item: dict[str, Any], events: Any) -> tuple[str, int, int]:
        argument = item["question"] if args.benchmark == "math" else item
        return runner(client, args.model, argument, events)

    for item in items[: args.warmup]:
        invoke(item, math_benchmark.EventLog("warmup", time.perf_counter()))
        client.chat.completions.reset()

    cpu_samples: list[dict[str, Any]] = []
    kv_samples: list[dict[str, Any]] = []
    events_out: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    with (output / "responses.jsonl").open("w", encoding="utf-8") as responses:
        for index, item in enumerate(items, start=1):
            request_id = f"{args.condition}_{index:03d}"
            origin = time.perf_counter()
            events = math_benchmark.EventLog(request_id, origin)
            cpu = math_benchmark.CpuSampler(
                request_id, origin, args.sample_interval
            )
            kv = KvCacheSampler(
                request_id, origin, args.kv_sample_interval, metrics_url
            )
            client.chat.completions.reset()
            cpu.start()
            kv.start()
            with events.phase("request"):
                answer, llm_calls, tool_calls = invoke(item, events)
            cpu.stop()
            kv.stop()
            latency = time.perf_counter() - origin
            prompt_tokens, completion_tokens = client.chat.completions.usage()
            cpu_values = [
                row["process_cpu_percent"] for row in cpu.samples
            ]
            kv_values = [
                row["kv_cache_usage_fraction"] for row in kv.samples
            ]
            cpu_samples.extend(cpu.samples)
            kv_samples.extend(kv.samples)
            events_out.extend(asdict(event) for event in events.events)
            summaries.append(
                {
                    "request_id": request_id,
                    "condition": args.condition,
                    "latency_s": latency,
                    "avg_process_cpu_percent": (
                        statistics.fmean(cpu_values) if cpu_values else 0.0
                    ),
                    "peak_process_cpu_percent": max(cpu_values, default=0.0),
                    "llm_calls": llm_calls,
                    "tool_calls": tool_calls,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "avg_kv_cache_usage_fraction": (
                        statistics.fmean(kv_values) if kv_values else 0.0
                    ),
                    "peak_kv_cache_usage_fraction": max(kv_values, default=0.0),
                }
            )
            responses.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "benchmark_id": item["id"],
                        "question": item["question"],
                        "answer": answer,
                    }
                )
                + "\n"
            )

    write_csv(
        output / "cpu_samples.csv",
        cpu_samples,
        [
            "request_id",
            "timestamp_s",
            "elapsed_s",
            "process_cpu_percent",
            "system_cpu_percent",
        ],
    )
    write_csv(
        output / "kv_cache_samples.csv",
        kv_samples,
        [
            "request_id",
            "timestamp_s",
            "elapsed_s",
            "kv_cache_usage_fraction",
        ],
    )
    write_csv(
        output / "events.csv",
        events_out,
        ["request_id", "event_type", "start_s", "end_s"],
    )
    write_csv(
        output / "summary.csv",
        summaries,
        [
            "request_id",
            "condition",
            "latency_s",
            "avg_process_cpu_percent",
            "peak_process_cpu_percent",
            "llm_calls",
            "tool_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "avg_kv_cache_usage_fraction",
            "peak_kv_cache_usage_fraction",
        ],
    )
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
