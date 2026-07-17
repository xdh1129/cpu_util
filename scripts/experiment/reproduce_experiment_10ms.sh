#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

output_root="${OUTPUT_ROOT:-results_10ms}"
for repeat in 1 2 3; do
  run_id="run_$(printf '%03d' "$repeat")"
  .venv/bin/python scripts/utils/capture_environment.py "$output_root/$run_id/environment.json"
  for condition in chatbot agent; do
    .venv/bin/python scripts/experiment/experiment.py \
      --condition "$condition" \
      --run-id "$run_id" \
      --output-root "$output_root" \
      --workload workloads/math_experiment.jsonl \
      --limit 5 \
      --sample-interval 0.01
  done
done

.venv/bin/python scripts/analysis/analyze_results.py \
  --results "$output_root" \
  --output "$output_root/analysis"
