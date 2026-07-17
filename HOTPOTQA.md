# HotpotQA workload

This experiment uses five pinned examples from the official HotpotQA
`distractor` validation split. Each example contains ten candidate paragraphs.

- Dataset: `hotpotqa/hotpot_qa`
- Revision: `1908d6afbbead072334abe2965f91bd2709910ab`
- Split/config: `distractor/validation`
- Selection: rows 0 through 4
- License: CC BY-SA 4.0

The chatbot receives all ten paragraphs in one prompt. The agent receives the
question, makes exactly one `document_search` tool call, receives the top three
paragraphs from a deterministic local lexical search, and then answers. The
search runs in a Client child process, so its CPU is included in Client CPU.

This is a one-shot retrieval-tool agent, not iterative ReAct, Reflection, LATS,
or full-wiki retrieval. Five examples are enough for the CPU experiment but not
for reporting an official HotpotQA accuracy score.

```bash
.venv/bin/python scripts/data/prepare_hotpotqa.py
bash scripts/experiment/reproduce_hotpotqa_10ms.sh
.venv/bin/python scripts/analysis/plot_hotpotqa_10ms.py
```
