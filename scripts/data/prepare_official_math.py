#!/usr/bin/env python3
"""Create a pinned five-problem workload from the MATH test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


REPO_ID = "EleutherAI/hendrycks_math"
REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
SPLIT = "test"
SELECTION = (
    ("algebra", 0),
    ("counting_and_probability", 0),
    ("geometry", 0),
    ("number_theory", 0),
    ("prealgebra", 0),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workloads/math_official_test_subset_5.jsonl"),
    )
    args = parser.parse_args()
    records = []
    for subject, row_index in SELECTION:
        filename = f"{subject}/{SPLIT}-00000-of-00001.parquet"
        path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            revision=REVISION,
        )
        row = pq.read_table(path).slice(row_index, 1).to_pylist()[0]
        records.append(
            {
                "id": f"math_test_{subject}_{row_index:04d}",
                "question": row["problem"],
                "reference_solution": row["solution"],
                "level": row["level"],
                "type": row["type"],
                "dataset_repo": REPO_ID,
                "dataset_revision": REVISION,
                "dataset_split": SPLIT,
                "dataset_subject": subject,
                "dataset_row_index": row_index,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} pinned MATH test problems to {args.output}")


if __name__ == "__main__":
    main()
