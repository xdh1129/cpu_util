# Script layout

| Directory | Purpose |
|---|---|
| `analysis/` | Aggregate CSV/JSON results and render timeline figures |
| `data/` | Download pinned benchmark revisions and prepare fixed workloads |
| `experiment/` | Run chatbot/agent measurements and complete reproductions |
| `serving/` | Start the Gemma 4 vLLM server and run API smoke tests |
| `setup/` | Create benchmark-specific environments and pinned dependencies |
| `tools/` | Tool subprocesses and environment bridges used by agents |
| `utils/` | Capture machine and software environment metadata |

Scripts that write experiment outputs change to the repository root before
resolving workloads and result directories, so they can be launched from any
working directory.

Current primary reproduction commands:

```bash
scripts/serving/serve_gemma4_tools.sh
scripts/experiment/reproduce_hotpotqa_10ms.sh
.venv/bin/python scripts/analysis/plot_hotpotqa_10ms.py
```

All four token/KV runs:

```bash
scripts/setup/setup_webshop.sh
scripts/experiment/reproduce_four_benchmarks_10ms.sh
.venv/bin/python scripts/analysis/compare_four_benchmarks.py
```
