#!/usr/bin/env python3
"""Create a pinned five-question HotpotQA distractor-dev workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


REPO_ID = "hotpotqa/hotpot_qa"
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
FILENAME = "distractor/validation-00000-of-00001.parquet"
ROW_INDICES = (0, 1, 2, 3, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workloads/hotpotqa_distractor_dev_subset_5.jsonl"),
    )
    args = parser.parse_args()
    source = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=FILENAME,
        revision=REVISION,
    )
    table = pq.read_table(source)
    records = []
    for index in ROW_INDICES:
        row = table.slice(index, 1).to_pylist()[0]
        context = [
            {"title": title, "sentences": sentences}
            for title, sentences in zip(
                row["context"]["title"], row["context"]["sentences"]
            )
        ]
        records.append(
            {
                "id": row["id"],
                "question": row["question"],
                "reference_answer": row["answer"],
                "question_type": row["type"],
                "level": row["level"],
                "supporting_facts": row["supporting_facts"],
                "context": context,
                "dataset_repo": REPO_ID,
                "dataset_revision": REVISION,
                "dataset_config": "distractor",
                "dataset_split": "validation",
                "dataset_row_index": index,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} HotpotQA questions to {args.output}")


if __name__ == "__main__":
    main()
