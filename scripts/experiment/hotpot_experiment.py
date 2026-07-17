#!/usr/bin/env python3
"""Measure chatbot versus document-search-agent client CPU on HotpotQA."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from experiment import CpuSampler, EventLog, completion, write_csv


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "document_search",
        "description": "Search the local document collection for relevant paragraphs.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def search_documents(query: str, context: list[dict[str, Any]], top_k: int = 3) -> str:
    query_terms = tokenize(query)
    ranked = []
    for document in context:
        text = document["title"] + " " + " ".join(document["sentences"])
        terms = tokenize(text)
        score = sum(3 if term in tokenize(document["title"]) else 1 for term in query_terms & terms)
        ranked.append((score, document["title"], document["sentences"]))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return "\n\n".join(
        f"[{title}] {''.join(sentences)}" for _, title, sentences in ranked[:top_k]
    )


def search_process(query: str, context: list[dict[str, Any]], connection: Any) -> None:
    try:
        connection.send({"ok": True, "value": search_documents(query, context)})
    except Exception as exc:
        connection.send({"ok": False, "error": str(exc)})
    finally:
        connection.close()


def call_search(query: str, context: list[dict[str, Any]]) -> str:
    parent, child = mp.Pipe(duplex=False)
    process = mp.Process(target=search_process, args=(query, context, child))
    process.start()
    child.close()
    if not parent.poll(10):
        process.terminate()
        process.join()
        raise TimeoutError("Document search timed out")
    result = parent.recv()
    process.join()
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return result["value"]


def format_context(context: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{doc['title']}] {''.join(doc['sentences'])}" for doc in context
    )


def chatbot(client: OpenAI, model: str, item: dict[str, Any], events: EventLog) -> tuple[str, int, int]:
    with events.phase("context_build"):
        messages = [
            {"role": "system", "content": "Answer using only the supplied documents. Give a concise answer."},
            {"role": "user", "content": f"Question: {item['question']}\n\nDocuments:\n{format_context(item['context'])}"},
        ]
    with events.phase("llm_call"):
        response = completion(client, model=model, messages=messages)
    with events.phase("finalize"):
        answer = response.choices[0].message.content or ""
    return answer, 1, 0


def agent(client: OpenAI, model: str, item: dict[str, Any], events: EventLog) -> tuple[str, int, int]:
    with events.phase("context_build"):
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "First call document_search exactly once with a concise query containing the key entities. Then answer using only its results."},
            {"role": "user", "content": item["question"]},
        ]
    with events.phase("llm_call"):
        first = completion(client, model=model, messages=messages, tools=[SEARCH_TOOL], tool_choice="required")
    with events.phase("agent_parse"):
        assistant = first.choices[0].message
        calls = assistant.tool_calls or []
        if len(calls) != 1:
            raise RuntimeError(f"Expected one document_search call, got {len(calls)}")
        query = json.loads(calls[0].function.arguments)["query"]
        messages.append(assistant.model_dump(exclude_none=True))
    with events.phase("tool_call"):
        result = call_search(query, item["context"])
    messages.append({"role": "tool", "tool_call_id": calls[0].id, "content": result})
    with events.phase("llm_call"):
        final = completion(client, model=model, messages=messages, tools=[SEARCH_TOOL])
    with events.phase("finalize"):
        answer = final.choices[0].message.content or ""
    return answer, 2, 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("chatbot", "agent"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workload", type=Path, default=Path("workloads/hotpotqa_distractor_dev_subset_5.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("results_hotpotqa_10ms"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    items = [json.loads(line) for line in args.workload.read_text(encoding="utf-8").splitlines() if line.strip()][:args.limit]
    client = OpenAI(base_url=args.base_url, api_key="local-vllm")
    runner = chatbot if args.condition == "chatbot" else agent
    output = args.output_root / args.run_id / args.condition
    output.mkdir(parents=True, exist_ok=True)
    for item in items[:args.warmup]:
        runner(client, args.model, item, EventLog("warmup", time.perf_counter()))
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
                answer, llm_calls, tool_calls = runner(client, args.model, item, events)
            sampler.stop()
            latency = time.perf_counter() - origin
            cpu_values = [row["process_cpu_percent"] for row in sampler.samples]
            samples.extend(sampler.samples)
            events_out.extend(asdict(event) for event in events.events)
            summaries.append({"request_id": request_id, "condition": args.condition, "latency_s": latency, "avg_process_cpu_percent": statistics.fmean(cpu_values) if cpu_values else 0.0, "peak_process_cpu_percent": max(cpu_values, default=0.0), "llm_calls": llm_calls, "tool_calls": tool_calls})
            responses.write(json.dumps({"request_id": request_id, "benchmark_id": item["id"], "question": item["question"], "reference_answer": item["reference_answer"], "answer": answer}) + "\n")
    write_csv(output / "cpu_samples.csv", samples, ["request_id", "timestamp_s", "elapsed_s", "process_cpu_percent", "system_cpu_percent"])
    write_csv(output / "events.csv", events_out, ["request_id", "event_type", "start_s", "end_s"])
    write_csv(output / "summary.csv", summaries, ["request_id", "condition", "latency_s", "avg_process_cpu_percent", "peak_process_cpu_percent", "llm_calls", "tool_calls"])
    (output / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
