#!/usr/bin/env python3
"""Measure chatbot versus test-execution/reflection agent on HumanEval."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import re
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from experiment import CpuSampler, EventLog, completion
from instrumented_runner import KvCacheSampler, RecordingClient


def extract_code(text: str, prompt: str, entry_point: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = fenced.group(1).strip() if fenced else text.strip()
    if f"def {entry_point}" not in code:
        code = prompt + code
    return code



def run_tests(code: str, test: str, entry_point: str) -> str:
    program = f"{code}\n\n{test}\n\ncheck({entry_point})\nprint('PASS')\n"
    with tempfile.TemporaryDirectory(prefix="humaneval-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        command = [
            "docker", "run", "--rm", "--network", "none",
            "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "32", "--memory", "512m", "--cpus", "1",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
            "--mount", f"type=bind,src={path},dst=/candidate.py,readonly",
            "swebench/sweb.eval.x86_64.pylint-dev_1776_pylint-8898:latest",
            "python", "-I", "-S", "/candidate.py",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "FAIL: timed out"
    if result.returncode == 0:
        return "PASS"
    detail = (result.stderr or result.stdout).strip()[-1200:]
    return f"FAIL: {detail}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def chatbot(
    client: RecordingClient, model: str, item: dict[str, Any], events: EventLog
) -> tuple[str, int, int, str]:
    messages = [
        {
            "role": "system",
            "content": "Complete the Python function. Return only executable Python code.",
        },
        {"role": "user", "content": item["prompt"]},
    ]
    with events.phase("llm_call"):
        response = completion(client, model=model, messages=messages)
    answer = extract_code(
        response.choices[0].message.content or "",
        item["prompt"],
        item["entry_point"],
    )
    return answer, 1, 0, "not_executed"


def agent(
    client: RecordingClient, model: str, item: dict[str, Any], events: EventLog
) -> tuple[str, int, int, str]:
    messages = [
        {
            "role": "system",
            "content": "Complete the Python function. Return only executable Python code.",
        },
        {"role": "user", "content": item["prompt"]},
    ]
    with events.phase("llm_call"):
        first = completion(client, model=model, messages=messages)
    candidate = extract_code(
        first.choices[0].message.content or "",
        item["prompt"],
        item["entry_point"],
    )
    with events.phase("tool_call"):
        result = run_tests(
            candidate, item["test"], item["entry_point"]
        )
    messages.extend(
        [
            {"role": "assistant", "content": candidate},
            {
                "role": "user",
                "content": (
                    f"Sandboxed official tests returned: {result}. "
                    "Return the corrected complete code only. If tests passed, "
                    "return the same code."
                ),
            },
        ]
    )
    with events.phase("llm_call"):
        final = completion(client, model=model, messages=messages)
    answer = extract_code(
        final.choices[0].message.content or "",
        item["prompt"],
        item["entry_point"],
    )
    with events.phase("tool_call"):
        final_result = run_tests(answer, item["test"], item["entry_point"])
    return answer, 2, 2, final_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("chatbot", "agent"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--workload",
        type=Path,
        default=Path("workloads/humaneval_subset_5.jsonl"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results_humaneval_10ms")
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--kv-sample-interval", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    items = [
        json.loads(line)
        for line in args.workload.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    client = RecordingClient(OpenAI(base_url=args.base_url, api_key="local-vllm"))
    runner = chatbot if args.condition == "chatbot" else agent
    metrics_url = args.base_url.removesuffix("/v1") + "/metrics"
    output = args.output_root / args.run_id / args.condition
    output.mkdir(parents=True, exist_ok=True)
    for item in items[: args.warmup]:
        runner(client, args.model, item, EventLog("warmup", time.perf_counter()))
        client.chat.completions.reset()

    cpu_rows: list[dict[str, Any]] = []
    kv_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    with (output / "responses.jsonl").open("w", encoding="utf-8") as responses:
        for index, item in enumerate(items, start=1):
            request_id = f"{args.condition}_{index:03d}"
            origin = time.perf_counter()
            events = EventLog(request_id, origin)
            cpu = CpuSampler(request_id, origin, args.sample_interval)
            kv = KvCacheSampler(
                request_id, origin, args.kv_sample_interval, metrics_url
            )
            client.chat.completions.reset()
            cpu.start()
            kv.start()
            with events.phase("request"):
                answer, llm_calls, tool_calls, test_result = runner(
                    client, args.model, item, events
                )
            cpu.stop()
            kv.stop()
            latency = time.perf_counter() - origin
            # Evaluate one-shot code after measurement so Docker does not affect CPU/latency.
            if test_result == "not_executed":
                test_result = run_tests(answer, item["test"], item["entry_point"])
            prompt_tokens, completion_tokens = client.chat.completions.usage()
            cpu_values = [row["process_cpu_percent"] for row in cpu.samples]
            kv_values = [row["kv_cache_usage_fraction"] for row in kv.samples]
            cpu_rows.extend(cpu.samples)
            kv_rows.extend(kv.samples)
            event_rows.extend(asdict(event) for event in events.events)
            summaries.append(
                {
                    "request_id": request_id,
                    "condition": args.condition,
                    "latency_s": latency,
                    "avg_process_cpu_percent": statistics.fmean(cpu_values),
                    "peak_process_cpu_percent": max(cpu_values),
                    "llm_calls": llm_calls,
                    "tool_calls": tool_calls,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "avg_kv_cache_usage_fraction": (
                        statistics.fmean(kv_values) if kv_values else 0.0
                    ),
                    "peak_kv_cache_usage_fraction": max(kv_values, default=0.0),
                    "passed": test_result == "PASS",
                }
            )
            responses.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "benchmark_id": item["id"],
                        "answer": answer,
                        "test_result": test_result,
                    }
                )
                + "\n"
            )
    write_csv(output / "cpu_samples.csv", cpu_rows, list(cpu_rows[0]))
    write_csv(
        output / "kv_cache_samples.csv",
        kv_rows,
        [
            "request_id",
            "timestamp_s",
            "elapsed_s",
            "kv_cache_usage_fraction",
        ],
    )
    write_csv(output / "events.csv", event_rows, list(event_rows[0]))
    write_csv(output / "summary.csv", summaries, list(summaries[0]))
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
