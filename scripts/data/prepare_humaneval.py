#!/usr/bin/env python3
"""Prepare a pinned five-task subset of the official OpenAI HumanEval set."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path


REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
URL = (
    "https://raw.githubusercontent.com/openai/human-eval/"
    f"{REVISION}/data/HumanEval.jsonl.gz"
)
TASK_IDS = tuple(f"HumanEval/{index}" for index in range(5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workloads/humaneval_subset_5.jsonl"),
    )
    args = parser.parse_args()
    try:
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read()
    except OSError:
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "clone", "--filter=blob:none", "https://github.com/openai/human-eval.git", directory], check=True)
            subprocess.run(["git", "-C", directory, "checkout", "--detach", REVISION], check=True)
            payload = (Path(directory) / "data/HumanEval.jsonl.gz").read_bytes()
    rows = [json.loads(line) for line in gzip.decompress(payload).decode("utf-8").splitlines()]
    selected = []
    for row in rows:
        if row["task_id"] not in TASK_IDS:
            continue
        selected.append(
            {
                "id": row["task_id"],
                "prompt": row["prompt"],
                "entry_point": row["entry_point"],
                "canonical_solution": row["canonical_solution"],
                "test": row["test"],
                "dataset_repo": "openai/human-eval",
                "dataset_revision": REVISION,
                "dataset_row_index": int(row["task_id"].split("/")[-1]),
            }
        )
    if len(selected) != len(TASK_IDS):
        raise RuntimeError("Pinned HumanEval tasks were not all found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(f"Wrote {len(selected)} HumanEval tasks to {args.output}")


if __name__ == "__main__":
    main()
