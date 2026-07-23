# HumanEval benchmark

This run uses tasks `HumanEval/0` through `HumanEval/4` from the official OpenAI
HumanEval repository pinned at commit `6d43fb980f9fee3c892a914eda09951f772ad10d`.
Generate the fixed workload with `.venv/bin/python scripts/data/prepare_humaneval.py`.

The chatbot makes one code-generation call; its official tests run only after
CPU, KV-cache, and latency measurement ends. The agent is a bounded reflection
agent: generate, execute official tests, feed the result back, regenerate, and
test once more. It therefore has two LLM calls and two test-tool calls.

Candidate code runs in Docker with no network, a read-only root filesystem,
dropped capabilities, one CPU, 512 MiB RAM, a PID limit, and temporary writable
mounts. Docker is daemon-owned, so candidate-code CPU is outside the measured
client process tree; Docker CLI orchestration is included. This deterministic
five-task subset is a smoke experiment, not the full official score.
