# HotpotQA client CPU experiment

This run uses five pinned questions from the official HotpotQA distractor
validation split. Each question is repeated three times in chatbot and local
document-search-agent conditions, giving 15 measured requests per condition.

The chatbot receives all ten candidate paragraphs in one prompt. The agent
receives only the question, calls deterministic local lexical search once, gets
the top three paragraphs, and then answers. This is a one-shot retrieval-tool
agent rather than ReAct, Reflection, LATS, or full-wiki retrieval.

## Manipulation check

All 15 chatbot requests made one LLM call and zero tool calls. All 15 agent
requests made two LLM calls and exactly one `document_search` call.

## Aggregate results

| Condition | Requests | Mean latency (s) | Mean client CPU (%) | Mean estimated CPU time (s) | LLM calls | Tool calls |
|---|---:|---:|---:|---:|---:|---:|
| Chatbot | 15 | 0.2618 | 34.8653 | 0.0940 | 1.0 | 0.0 |
| Agent | 15 | 0.8622 | 108.7167 | 0.8971 | 2.0 | 1.0 |

## Measurement caveats

The configured sampling interval was 10 ms. Across 1,044 samples, actual
within-request spacing had a 15.464 ms median, 16.172 ms mean, and 19.488 ms
95th percentile (range 15.169–21.154 ms). This remains a requested-10-ms run,
not a true 10 ms effective cadence.

One agent sample in run 002 reports 20,210.2% process-tree CPU. This is a
short-interval `psutil` process-tree aggregation artifact, so mean peak CPU is
not suitable for interpretation. The raw observation is retained rather than
silently filtered.

This five-question workload is for CPU timeline comparison. It is not large
enough to report an official HotpotQA benchmark accuracy score.
