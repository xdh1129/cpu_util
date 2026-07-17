# CPU Utilization Timeline Experiment

This repository compares client-side CPU utilization for a single-turn
chatbot and a tool-augmented agent while both use the same local Gemma 4
backend served by vLLM.

The model server's CPU usage is not the primary measurement target. The
experiment measures the request runner, agent orchestration, context
construction, response parsing, and local tool subprocesses.

## Current serving configuration

- Model: `google/gemma-4-26B-A4B-it`
- Served name: `gemma4`
- vLLM: `0.25.1`
- Precision: BF16
- Context limit: 8,192 tokens
- GPU: one GPU, selected with `CUDA_VISIBLE_DEVICES`
- MoE backend: Triton
- Sampler: native vLLM fallback
- API: `http://127.0.0.1:8000/v1`

The Triton/native fallbacks are required on the current Blackwell and CUDA 13
host because FlashInfer 0.6.13's optional JIT extensions use CUDA headers that
are incompatible with the bundled CUDA 13.2 compiler.

## Environment setup

Gemma is gated. Accept its Hugging Face license and authenticate before the
first model download.

```bash
sudo apt-get install -y python3.12-venv python3.12-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
hf auth login
```

## Start and validate the server

Run the server in one terminal:

```bash
scripts/serving/serve_gemma4.sh
```

Select a different physical GPU if needed:

```bash
CUDA_VISIBLE_DEVICES=1 scripts/serving/serve_gemma4.sh
```

Validate it from another terminal:

```bash
scripts/serving/smoke_test.sh
```

Capture machine and software metadata with each experimental run:

```bash
source .venv/bin/activate
python scripts/utils/capture_environment.py results/run_name/environment.json
```

Do not collect measured trials during initial model loading, kernel
compilation, or warm-up. Those activities create one-time CPU bursts that are
not representative of steady-state request execution.

The detailed research design is in
[`cpu_timeline_experiment_design.md`](cpu_timeline_experiment_design.md).
