#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
PYTHON="${VENV_DIR:-${ROOT_DIR}/.venv}/bin/python"
REPEATS="${REPEATS:-3}"
LIMIT="${LIMIT:-5}"

curl --fail --silent --max-time 5 \
  http://127.0.0.1:8000/v1/models >/dev/null
for repeat in $(seq 1 "${REPEATS}"); do
  run_id="run_$(printf '%03d' "${repeat}")"
  "${PYTHON}" scripts/utils/capture_environment.py \
    "results/${run_id}/environment.json"
  "${PYTHON}" scripts/experiment/experiment.py \
    --condition chatbot --run-id "${run_id}" --limit "${LIMIT}"
  "${PYTHON}" scripts/experiment/experiment.py \
    --condition agent --run-id "${run_id}" --limit "${LIMIT}"
done
"${PYTHON}" scripts/analysis/analyze_results.py
