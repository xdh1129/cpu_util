#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
repeats="${REPEATS:-3}"
kv_cache_bytes="${KV_CACHE_MEMORY_BYTES:-25769803776}"

curl --fail --silent --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null
for repeat in $(seq 1 "$repeats"); do
  run_id="run_$(printf '%03d' "$repeat")"
  for root in \
    results_math_metrics_10ms \
    results_hotpotqa_metrics_10ms \
    results_humaneval_10ms \
    results_webshop_10ms
  do
    printf '{"kv_cache_bytes": %s, "kv_cache_gib": 24, "gpu_kv_capacity_tokens": 114185}\n' \
      "$kv_cache_bytes" > "$root/server_kv_cache_config.json"
    .venv/bin/python scripts/utils/capture_environment.py \
      "$root/$run_id/environment.json"
  done

  for condition in chatbot agent; do
    .venv/bin/python scripts/experiment/instrumented_runner.py \
      --benchmark math --condition "$condition" --run-id "$run_id" \
      --workload workloads/math_official_test_subset_5.jsonl \
      --output-root results_math_metrics_10ms
    .venv/bin/python scripts/experiment/instrumented_runner.py \
      --benchmark hotpotqa --condition "$condition" --run-id "$run_id" \
      --workload workloads/hotpotqa_distractor_dev_subset_5.jsonl \
      --output-root results_hotpotqa_metrics_10ms
    .venv/bin/python scripts/experiment/humaneval_experiment.py \
      --condition "$condition" --run-id "$run_id" \
      --output-root results_humaneval_10ms
    .venv/bin/python scripts/experiment/webshop_experiment.py \
      --condition "$condition" --run-id "$run_id" \
      --output-root results_webshop_10ms
  done
done

for root in \
  results_math_metrics_10ms \
  results_hotpotqa_metrics_10ms \
  results_humaneval_10ms \
  results_webshop_10ms
do
  .venv/bin/python scripts/analysis/analyze_extended.py \
    --results "$root" --output "$root/analysis"
done
