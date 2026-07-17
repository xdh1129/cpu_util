#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON="${VENV_DIR}/bin/python"
VLLM="${VENV_DIR}/bin/vllm"
CUDA_HOME="${CUDA_HOME:-${VENV_DIR}/lib/python3.12/site-packages/nvidia/cu13}"

if [[ ! -x "${PYTHON}" || ! -x "${VLLM}" ]]; then
  echo "Missing virtual environment. Create it and install requirements first." >&2
  exit 1
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA compiler not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi

export CUDA_HOME
export PATH="${VENV_DIR}/bin:${CUDA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# FlashInfer 0.6.13 JIT headers conflict with its bundled CUDA 13.2 compiler
# on this Blackwell host. These supported vLLM fallbacks avoid that JIT path.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

exec "${VLLM}" serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --moe-backend triton

