# CPU Utilization Timeline Experiment Design

## Research Question

This experiment compares a single-turn chatbot workload with a tool-augmented agent workload under the same black-box model backend. Since the LLM server implementation is unavailable, the experiment focuses on client-side CPU cost: the request runner, agent runtime, prompt/context construction, response parsing, and local tool execution.

Main question:

> Does a tool-augmented agent workload create higher or more bursty client-side CPU utilization than a single-turn chatbot workload?

Secondary question:

> Which phases of an agent request contribute most to CPU utilization: LLM waiting, agent reasoning/parsing, tool execution, or final answer formatting?

## Experimental Conditions

| Condition | Description | Paper Analogy |
|---|---|---|
| Chatbot baseline | One prompt produces one model response. No iterative loop and no tool call. | ShareGPT / conventional chatbot workload |
| Agent with tool use | Agent performs iterative reasoning and invokes local tools before producing a final answer. | ReAct-style agent on HotpotQA, MATH, or HumanEval |

The backend model should be held constant and treated as a black box. This means CPU measurements should not claim to represent model-serving CPU utilization.

## Recommended Workload

Start with MATH plus a Python calculator tool.

Reasons:

- The tool runs locally, so tool CPU usage is visible.
- It avoids dependency on external APIs.
- It makes tool-related CPU bursts easier to interpret.
- It maps cleanly to the paper's MATH benchmark, where agents can use calculation tools.

If there is time for a second workload, add HotpotQA with a local document-search or mock Wikipedia tool. This helps distinguish CPU-heavy local tools from I/O-heavy search tools.

## Metrics

Primary metrics:

- Per-request CPU utilization timeline
- Average CPU utilization per request
- Peak CPU utilization per request
- CPU time per request
- End-to-end request latency

Optional metrics:

- User CPU time
- System CPU time
- Tool subprocess CPU time
- Number of LLM calls
- Number of tool calls
- Total tool execution time

## Timeline Figure

The main figure should show one full request execution.

X-axis:

- Elapsed request time in seconds

Y-axis:

- CPU utilization percentage

Plot elements:

- CPU utilization line over time
- Vertical line for request start
- Vertical line for request end
- Semi-transparent phase spans:
  - LLM call / API wait
  - Agent reasoning and parsing
  - Tool execution
  - Final answer formatting

Recommended figures:

1. Chatbot single-request CPU timeline
2. Agent plus tool single-request CPU timeline
3. Optional overlay or stacked subplot comparison

## Data Logs

CPU samples:

```csv
request_id,timestamp_s,elapsed_s,process_cpu_percent,system_cpu_percent
chatbot_001,172.40,0.00,4.1,18.2
chatbot_001,172.50,0.10,2.9,17.7
```

Execution events:

```csv
request_id,event_type,start_s,end_s
agent_001,agent_parse,0.00,0.08
agent_001,llm_call,0.08,3.92
agent_001,tool_call,4.01,4.38
agent_001,llm_call,4.48,8.10
agent_001,finalize,8.10,8.24
```

Suggested event types:

- `request`
- `llm_call`
- `agent_parse`
- `context_build`
- `tool_call`
- `finalize`

## Controls

Keep these fixed:

- Same backend model
- Same machine
- Same number of benchmark questions, e.g. 50
- Same concurrency, starting with 1
- Same temperature
- Same max output tokens
- Same sampling interval for CPU monitoring
- Same warm-up policy

Each condition should run at least three times and report mean plus variation.

## Expected Interpretation

The chatbot baseline is expected to spend most of its time waiting for the model response, with relatively low and smooth local CPU utilization. Short CPU bumps may appear during request construction, response streaming, and output parsing.

The agent workload is expected to show a more bursty timeline. CPU utilization may rise during context construction, action parsing, tool invocation, and final response formatting. With a Python calculator or code-execution tool, the tool phase should create visible CPU spikes.

The key result is not only whether the agent has higher average CPU utilization, but whether CPU usage becomes phase-dependent and bursty because of the repeated reasoning-tool loop.

## Reporting Wording

Use this framing:

> Because the LLM server is treated as a black box, this experiment does not measure model-serving CPU utilization. Instead, it measures the additional client-side CPU overhead introduced by agentic control flow and local tool execution, compared with a conventional single-turn chatbot workload.

This avoids overclaiming while still directly addressing the cost of agentic workflows.

