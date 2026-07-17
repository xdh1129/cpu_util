# Gemma 4 client CPU experiment: 10 ms configured sampling

## Run definition

This reruns the original experiment unchanged except for the configured CPU
sampling interval, reduced from 100 ms to 10 ms. It uses Gemma 4 26B A4B IT
through vLLM with the Triton MoE backend, three repeats, five fixed questions,
and both chatbot and tool-augmented-agent conditions (15 measured requests per
condition). The client process tree and host-wide CPU percentages were sampled
with `psutil`.

Reproduce after starting the server with `scripts/serve_gemma4_tools.sh`:

```bash
bash scripts/reproduce_experiment_10ms.sh
.venv/bin/python plot_10ms_design_figures.py
```

## Aggregate results

| Condition | Requests | Mean latency (s) | Mean client CPU (%) | Mean estimated CPU time (s) | Mean LLM calls | Mean tool calls |
|---|---:|---:|---:|---:|---:|---:|
| Chatbot | 15 | 1.9245 | 42.6742 | 0.8259 | 1.0 | 0.0 |
| Agent | 15 | 1.7949 | 87.4857 | 1.4377 | 2.0 | 1.0 |

## Sampling-cadence validation

All six condition/run configuration files record `sample_interval: 0.01`.
Across 3,121 samples and 3,091 within-request intervals, the actual spacing was:

- median: 19.201 ms
- mean: 17.894 ms
- 95th percentile: 19.552 ms
- range: 14.920–21.896 ms

Therefore, this is a **10 ms requested-interval experiment**, but the current
whole-process-tree `psutil` sampler did not achieve a true 10 ms measurement
cadence. Its sampling work is performed before the 10 ms wait, so overhead adds
to the requested interval.

## Peak-value caveat

One raw agent sample in run 001 reports 23,419.8% process-tree CPU, which makes
the mean of per-request peaks unsuitable for interpretation. Multi-core process
CPU can legitimately exceed 100%, but this magnitude is a short-interval
`psutil`/process-tree aggregation artifact. The raw value is retained for audit;
latency, call counts, timelines, and cadence statistics remain available without
silently filtering the observation.

Machine-readable aggregate output is in `results_10ms/analysis/aggregate.csv`
and `aggregate.json`; raw samples and events are under each run directory.
