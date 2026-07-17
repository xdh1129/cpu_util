# Official MATH client CPU experiment

This run uses five pinned problems from the official Hendrycks MATH test set.
It repeats each problem three times in chatbot and calculator-agent conditions,
for 15 measured requests per condition. It is a CPU-timeline experiment, not a
full-dataset MATH accuracy score.

## Manipulation check

All 15 chatbot requests made one LLM call and zero tool calls. All 15 agent
requests made two LLM calls and exactly one calculator call. For problems with
no useful numeric subcalculation, the strict agent protocol permits a harmless
`0` calculator expression so tool-phase overhead remains controlled.

## Aggregate results

| Condition | Requests | Mean latency (s) | Mean client CPU (%) | Mean estimated CPU time (s) | LLM calls | Tool calls |
|---|---:|---:|---:|---:|---:|---:|
| Chatbot | 15 | 2.6851 | 41.7392 | 1.1296 | 1.0 | 0.0 |
| Agent | 15 | 2.2930 | 65.1916 | 1.3648 | 2.0 | 1.0 |

## Sampling cadence

The configured interval was 10 ms. Across 4,206 samples, actual within-request
spacing had a 19.163 ms median, 17.756 ms mean, and 19.576 ms 95th percentile
(range 15.114–26.357 ms). As in the earlier run, whole-process-tree sampling
overhead prevents the current `psutil` loop from sustaining a true 10 ms
cadence, so this must be described as a requested-10-ms experiment.

Raw samples, event boundaries, responses, configurations, and environment
records are retained under `results_math_official_10ms/`.
