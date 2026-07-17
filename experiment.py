#!/usr/bin/env python3
"""Measure chatbot and calculator-agent client-side CPU timelines."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import multiprocessing as mp
import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import psutil
from openai import OpenAI


@dataclass
class Event:
    request_id: str
    event_type: str
    start_s: float
    end_s: float


class EventLog:
    def __init__(self, request_id: str, origin: float) -> None:
        self.request_id = request_id
        self.origin = origin
        self.events: list[Event] = []

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter() - self.origin
        try:
            yield
        finally:
            self.events.append(
                Event(
                    self.request_id,
                    name,
                    start,
                    time.perf_counter() - self.origin,
                )
            )


class CpuSampler(threading.Thread):
    def __init__(self, request_id: str, origin: float, interval: float) -> None:
        super().__init__(daemon=True)
        self.request_id = request_id
        self.origin = origin
        self.interval = interval
        self.stopping = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.root = psutil.Process()

    def run(self) -> None:
        known: dict[int, psutil.Process] = {}
        psutil.cpu_percent(None)
        while not self.stopping.is_set():
            processes = [self.root]
            try:
                processes.extend(self.root.children(recursive=True))
            except psutil.Error:
                pass
            for process in processes:
                if process.pid not in known:
                    try:
                        process.cpu_percent(None)
                        known[process.pid] = process
                    except psutil.Error:
                        pass
            process_cpu = 0.0
            for pid, process in list(known.items()):
                try:
                    process_cpu += process.cpu_percent(None)
                except psutil.Error:
                    known.pop(pid, None)
            self.samples.append(
                {
                    "request_id": self.request_id,
                    "timestamp_s": time.time(),
                    "elapsed_s": time.perf_counter() - self.origin,
                    "process_cpu_percent": process_cpu,
                    "system_cpu_percent": psutil.cpu_percent(None),
                }
            )
            self.stopping.wait(self.interval)

    def stop(self) -> None:
        self.stopping.set()
        self.join()


FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "gcd": math.gcd,
    "sqrt": math.sqrt,
}
NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Call,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    if not all(isinstance(node, NODES) for node in ast.walk(tree)):
        raise ValueError("Unsupported calculator expression")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in FUNCTIONS:
            raise ValueError(f"Unknown calculator name: {node.id}")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS
        ):
            raise ValueError("Unsupported calculator function")
    value = eval(
        compile(tree, "<calculator>", "eval"),
        {"__builtins__": {}},
        FUNCTIONS,
    )
    return str(value)


def calculator_process(expression: str, connection: Any) -> None:
    try:
        connection.send({"ok": True, "value": calculate(expression)})
    except Exception as exc:
        connection.send({"ok": False, "error": str(exc)})
    finally:
        connection.close()


def call_calculator(expression: str) -> str:
    parent, child = mp.Pipe(duplex=False)
    process = mp.Process(target=calculator_process, args=(expression, child))
    process.start()
    child.close()
    if not parent.poll(10):
        process.terminate()
        process.join()
        raise TimeoutError("Calculator timed out")
    result = parent.recv()
    process.join()
    if not result["ok"]:
        raise ValueError(result["error"])
    return result["value"]


TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate one arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}


def completion(client: OpenAI, **kwargs: Any) -> Any:
    return client.chat.completions.create(
        temperature=0,
        seed=0,
        max_tokens=512,
        **kwargs,
    )


def chatbot(
    client: OpenAI, model: str, question: str, events: EventLog
) -> tuple[str, int, int]:
    with events.phase("context_build"):
        messages = [
            {
                "role": "system",
                "content": "Solve the math problem and give a concise final answer.",
            },
            {"role": "user", "content": question},
        ]
    with events.phase("llm_call"):
        response = completion(client, model=model, messages=messages)
    with events.phase("finalize"):
        answer = response.choices[0].message.content or ""
    return answer, 1, 0


def agent(
    client: OpenAI, model: str, question: str, events: EventLog
) -> tuple[str, int, int]:
    with events.phase("context_build"):
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Solve the problem. Use the calculator tool for arithmetic, "
                    "then give a concise final answer."
                ),
            },
            {"role": "user", "content": question},
        ]
    with events.phase("llm_call"):
        first = completion(
            client,
            model=model,
            messages=messages,
            tools=[TOOL],
            tool_choice="required",
        )
    with events.phase("agent_parse"):
        assistant = first.choices[0].message
        calls = assistant.tool_calls or []
        if not calls:
            raise RuntimeError("Agent did not call the calculator")
        messages.append(assistant.model_dump(exclude_none=True))
    for call in calls:
        with events.phase("agent_parse"):
            expression = json.loads(call.function.arguments)["expression"]
        with events.phase("tool_call"):
            result = call_calculator(expression)
        messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": result}
        )
    with events.phase("llm_call"):
        final = completion(client, model=model, messages=messages, tools=[TOOL])
    with events.phase("finalize"):
        answer = final.choices[0].message.content or ""
    return answer, 2, len(calls)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("chatbot", "agent"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workload", type=Path, default=Path("workloads/math.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in args.workload.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        items = items[: args.limit]
    client = OpenAI(base_url=args.base_url, api_key="local-vllm")
    runner = chatbot if args.condition == "chatbot" else agent
    output = args.output_root / args.run_id / args.condition
    output.mkdir(parents=True, exist_ok=True)

    for item in items[: args.warmup]:
        runner(
            client,
            args.model,
            item["question"],
            EventLog("warmup", time.perf_counter()),
        )

    samples: list[dict[str, Any]] = []
    events_out: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    with (output / "responses.jsonl").open("w", encoding="utf-8") as responses:
        for index, item in enumerate(items, start=1):
            request_id = f"{args.condition}_{index:03d}"
            origin = time.perf_counter()
            events = EventLog(request_id, origin)
            sampler = CpuSampler(request_id, origin, args.sample_interval)
            sampler.start()
            with events.phase("request"):
                answer, llm_calls, tool_calls = runner(
                    client, args.model, item["question"], events
                )
            sampler.stop()
            latency = time.perf_counter() - origin
            cpu_values = [row["process_cpu_percent"] for row in sampler.samples]
            samples.extend(sampler.samples)
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
        samples,
        [
            "request_id",
            "timestamp_s",
            "elapsed_s",
            "process_cpu_percent",
            "system_cpu_percent",
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
        ],
    )
    (output / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
