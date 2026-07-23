#!/usr/bin/env python3
"""Measure single-turn baseline versus interactive agent on WebShop."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from experiment import CpuSampler, EventLog, completion
from instrumented_runner import KvCacheSampler, RecordingClient


ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "webshop_action",
        "description": "Take one action in the WebShop text environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Exactly search[keywords] or click[visible text]",
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


class WebShopWorker:
    def __init__(self, root: Path) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(root / ".cache/webshop"),
                "JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64",
            }
        )
        self.process = subprocess.Popen(
            [
                str(root / ".venv-webshop/bin/python"),
                str(root / "scripts/tools/webshop_env_worker.py"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=environment,
        )
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"WebShop worker failed to start: {ready}")

    def _read(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("WebShop worker exited")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def command(self, **payload: Any) -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return self._read()

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.command(command="close")
            finally:
                self.process.wait(timeout=5)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def baseline(
    client: RecordingClient,
    model: str,
    state: dict[str, Any],
    worker: WebShopWorker,
    events: EventLog,
) -> tuple[str, int, int, float, int]:
    prompt = (
        "Choose exactly one next WebShop action. Output only search[keywords] "
        "or click[visible text].\n\n" + state["observation"]
    )
    with events.phase("llm_call"):
        response = completion(client, model=model, messages=[{"role": "user", "content": prompt}])
    text = response.choices[0].message.content or ""
    match = re.search(r"(?:search|click)\[[^\]]+\]", text, re.IGNORECASE)
    action = match.group(0) if match else "search[product]"
    with events.phase("tool_call"):
        outcome = worker.command(command="step", action=action)
    return action, 1, 0, float(outcome["reward"]), 1


def agent(
    client: RecordingClient,
    model: str,
    state: dict[str, Any],
    worker: WebShopWorker,
    events: EventLog,
    max_steps: int,
) -> tuple[str, int, int, float, int]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Navigate WebShop. At every turn call webshop_action exactly "
                "once using search[keywords] or click[visible text]."
            ),
        },
        {"role": "user", "content": state["observation"]},
    ]
    reward = 0.0
    actions: list[str] = []
    for _ in range(max_steps):
        with events.phase("llm_call"):
            response = completion(
                client,
                model=model,
                messages=messages,
                tools=[ACTION_TOOL],
                tool_choice="required",
            )
        assistant = response.choices[0].message
        calls = assistant.tool_calls or []
        if len(calls) == 1:
            action = json.loads(calls[0].function.arguments)["action"]
            tool_call_id = calls[0].id
            messages.append(assistant.model_dump(exclude_none=True))
        else:
            text = assistant.content or ""
            match = re.search(r"(?:search|click)\[[^\]]+\]", text, re.IGNORECASE)
            action = match.group(0) if match else "search[product]"
            tool_call_id = None
            messages.append({"role": "assistant", "content": text})
        actions.append(action)
        with events.phase("tool_call"):
            outcome = worker.command(command="step", action=action)
        reward = float(outcome["reward"])
        if tool_call_id is not None:
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": outcome["observation"]})
        else:
            messages.append({"role": "user", "content": outcome["observation"]})
        if outcome["done"]:
            break
    return " -> ".join(actions), len(actions), len(actions), reward, len(actions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("chatbot", "agent"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results_webshop_10ms")
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--kv-sample-interval", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    client = RecordingClient(OpenAI(base_url=args.base_url, api_key="local-vllm"))
    metrics_url = args.base_url.removesuffix("/v1") + "/metrics"
    output = args.output_root / args.run_id / args.condition
    output.mkdir(parents=True, exist_ok=True)
    worker = WebShopWorker(root)
    runner = baseline if args.condition == "chatbot" else agent

    def invoke(session: int, events: EventLog) -> tuple[str, int, int, float, int]:
        state = worker.command(command="reset", session=session)
        if args.condition == "chatbot":
            return runner(client, args.model, state, worker, events)
        return runner(client, args.model, state, worker, events, args.max_steps)

    try:
        for session in range(args.warmup):
            invoke(session, EventLog("warmup", time.perf_counter()))
            client.chat.completions.reset()
        cpu_rows: list[dict[str, Any]] = []
        kv_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        with (output / "responses.jsonl").open("w", encoding="utf-8") as responses:
            for session in range(args.limit):
                request_id = f"{args.condition}_{session + 1:03d}"
                origin = time.perf_counter()
                events = EventLog(request_id, origin)
                cpu = CpuSampler(request_id, origin, args.sample_interval)
                kv = KvCacheSampler(request_id, origin, args.kv_sample_interval, metrics_url)
                client.chat.completions.reset()
                cpu.start()
                kv.start()
                with events.phase("request"):
                    answer, llm_calls, tool_calls, reward, steps = invoke(session, events)
                cpu.stop()
                kv.stop()
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
                        "latency_s": time.perf_counter() - origin,
                        "avg_process_cpu_percent": statistics.fmean(cpu_values),
                        "peak_process_cpu_percent": max(cpu_values),
                        "llm_calls": llm_calls,
                        "tool_calls": tool_calls,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "avg_kv_cache_usage_fraction": statistics.fmean(kv_values) if kv_values else 0.0,
                        "peak_kv_cache_usage_fraction": max(kv_values, default=0.0),
                        "reward": reward,
                        "steps": steps,
                    }
                )
                responses.write(json.dumps({"request_id": request_id, "session": session, "actions": answer, "reward": reward}) + "\n")
        write_csv(output / "cpu_samples.csv", cpu_rows, list(cpu_rows[0]))
        write_csv(output / "kv_cache_samples.csv", kv_rows, ["request_id", "timestamp_s", "elapsed_s", "kv_cache_usage_fraction"])
        write_csv(output / "events.csv", event_rows, list(event_rows[0]))
        write_csv(output / "summary.csv", summaries, list(summaries[0]))
        (output / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")
    finally:
        worker.close()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
