#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

output_root="${OUTPUT_ROOT:-results_math_official_10ms}"
workload="${WORKLOAD:-workloads/math_official_test_subset_5.jsonl}"
repeats="${REPEATS:-3}"

curl --fail --silent --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null
for repeat in $(seq 1 "$repeats"); do
  run_id="run_$(printf '%03d' "$repeat")"
  .venv/bin/python scripts/capture_environment.py "$output_root/$run_id/environment.json"
  for condition in chatbot agent; do
    .venv/bin/python experiment.py \
      --condition "$condition" \
      --run-id "$run_id" \
      --output-root "$output_root" \
      --workload "$workload" \
      --limit 5 \
      --sample-interval 0.01
  done
done

.venv/bin/python analyze_results.py \
  --results "$output_root" \
  --output "$output_root/analysis"
