# Four-benchmark token and KV-cache experiment (10 ms)

## Method

Every condition used the same local `google/gemma-4-26B-A4B-it` vLLM server on
one GPU. The KV cache was explicitly fixed at 25,769,803,776 bytes (24 GiB),
verified by the saved startup log; vLLM reported capacity for 114,185 tokens.
Each benchmark measured five fixed tasks and three repeats (15
requests per condition). Client-process-tree CPU and vLLM metrics used a
configured 10 ms interval. Tokens sum API-reported prompt and completion usage
across all LLM calls in each request.

KV usage is the mean per-request peak `vllm:kv_cache_usage_perc`, expressed as
percent of allocated KV blocks. This server-wide gauge has no request ID.
Sequential isolated requests align it with the active request, and the fixed
pool allows a direct conversion to allocated KV memory.

## Results

| Benchmark | Condition | Mean tokens | Mean peak KV | Mean peak KV MiB | Mean client CPU<br>(total 64-CPU capacity) | Mean latency |
|---|---|---:|---:|---:|---:|---:|
| MATH | chatbot | 488.7 | 0.433% | 106.34 | 0.911% | 2.650 s |
| MATH | agent | 687.2 | 0.414% | 101.63 | 1.320% | 2.160 s |
| HotpotQA | chatbot | 1,334.8 | 0.952% | 234.01 | 0.917% | 0.263 s |
| HotpotQA | agent | 698.0 | 0.502% | 123.26 | 1.630% | 0.871 s |
| HumanEval | chatbot | 317.0 | 0.283% | 69.50 | 0.917% | 1.203 s |
| HumanEval | reflection agent | 834.4 | 0.459% | 112.88 | 0.959% | 2.676 s |
| WebShop | chatbot | 105.0 | 0.099% | 24.25 | 1.561% | 0.172 s |
| WebShop | iterative agent | 9,354.4 | 1.068% | 262.55 | 1.163% | 1.889 s |

HumanEval pass@1 on this five-task subset was 100% for both conditions in all
three deterministic repeats. WebShop mean reward was 0 for the chatbot and
0.181 for the agent; the agent averaged 5.67 steps.

Raw process CPU follows psutil's one-core convention. Reported and plotted CPU is normalized by the 64 logical CPUs available to the experiment, so 100% means all 64 CPUs are fully occupied. Observed median CPU intervals were 9.997--9.999 ms across the four benchmarks (p95 10.085--10.304 ms). Polling adds small client overhead. HumanEval
candidate CPU is excluded as documented in `HUMANEVAL.md`; WebShop uses the
official small setting documented in `WEBSHOP.md`. Raw data and individual
figures are under `results_*_10ms`; combined artifacts are in
`results_comparison_10ms`.
