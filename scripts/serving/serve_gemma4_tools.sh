#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
CUDA_HOME="${CUDA_HOME:-${VENV_DIR}/lib/python3.12/site-packages/nvidia/cu13}"
export CUDA_HOME
export PATH="${VENV_DIR}/bin:${CUDA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "${VENV_DIR}/bin/vllm" serve \
  "${MODEL:-google/gemma-4-26B-A4B-it}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8000}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-25769803776}" \
  --served-model-name "${SERVED_MODEL_NAME:-gemma4}" \
  --moe-backend triton \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice
