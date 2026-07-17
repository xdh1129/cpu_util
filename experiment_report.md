# CPU Utilization Timeline Experiment Report

## Setup

- Backend: `google/gemma-4-26B-A4B-it`
- Server: vLLM 0.25.1, BF16, one RTX PRO 6000 Blackwell GPU
- Context limit: 8,192 tokens
- Client concurrency: 1
- CPU sampling interval: 0.1 seconds
- Decoding: temperature 0, seed 0, maximum 512 output tokens
- Workload: five fixed arithmetic questions
- Repeats: three
- Measured requests: 15 per condition
- Warm-up: one excluded request per condition and repeat

The chatbot makes one LLM request and does not invoke a tool. The agent makes
one tool-selection LLM request, executes a restricted calculator in a child
process, and makes a second LLM request for its final response.

## Aggregate results

| Metric | Chatbot mean (SD) | Agent mean (SD) |
|---|---:|---:|
| Latency (s) | 1.939 (0.619) | 1.808 (1.268) |
| Average client CPU (%) | 10.367 (1.643) | 30.693 (8.452) |
| Peak client CPU (%) | 19.287 (2.733) | 112.080 (5.584) |
| Estimated client CPU time (s) | 0.208 (0.088) | 0.473 (0.186) |
| LLM calls | 1.0 (0.0) | 2.0 (0.0) |
| Tool calls | 0.0 (0.0) | 1.0 (0.0) |

Relative to the chatbot, the agent used approximately 2.96 times the average
client CPU, 5.81 times the peak CPU, and 2.28 times the estimated client CPU
time. Mean latency was not higher in this small sample because response length
varied substantially between conditions; latency is therefore not a clean
proxy for local CPU overhead.

## Interpretation

The measurements support the main hypothesis: tool-augmented agent execution
is substantially more CPU-bursty on the client. The agent timelines contain a
distinct child-process spike during calculator execution and additional
request construction, parsing, and finalization around the second LLM call.
The chatbot spends most of each request waiting for the model server and shows
lower, smoother client CPU utilization.

These results do not measure model-serving CPU. The vLLM server is a separate
process and the measured process tree is rooted at the client runner.

## Artifacts

- Aggregate data: `results/analysis/aggregate.csv` and `aggregate.json`
- Chatbot timeline: `results/analysis/chatbot_timeline.png`
- Agent timeline: `results/analysis/agent_timeline.png`
- Raw samples and events: `results/run_001` through `results/run_003`
- Environment metadata: `results/run_*/environment.json`

One agent response reached the 512-token output limit while verbally
re-deriving a GCD, despite receiving the correct calculator result. This does
not invalidate CPU instrumentation, but it demonstrates why answer quality and
output length should be reported separately in a larger benchmark study.
