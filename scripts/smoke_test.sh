#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${SERVED_MODEL_NAME:-gemma4}"

curl --fail --silent --show-error \
  --max-time 120 \
  "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "model": "${MODEL}",
  "messages": [
    {"role": "system", "content": "Answer concisely."},
    {"role": "user", "content": "What is 17 multiplied by 23?"}
  ],
  "temperature": 0,
  "max_tokens": 64,
  "seed": 0
}
JSON
echo

