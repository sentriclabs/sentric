# Feature 13: Custom ReAct Agent Harness (from scratch)

**Priority**: P0
**Effort**: L
**Status**: planned

## Summary
Build a minimal ReAct-style agent from scratch, fully instrumented with Sentric. Run on the same SWE-bench tasks as mini-swe-agent for comparison. This proves the SDK works in a non-framework context and generates comparative trajectory data.

## Requirements
- ReAct loop: system prompt → think → act (tool call) → observe (tool result) → repeat
- Tools: `bash` (run shell commands), `read_file`, `write_file`, `search` (grep)
- Uses the same Modal-hosted Qwen model
- Every step logged via `TrajectoryCollector.add_message()` or `add_step()`
- Run on same 5-10 SWE-bench Lite tasks
- All trajectories saved in Sentric format
- Max 25 turns per task to prevent runaway cost

## Design Notes
**Architecture**: Single Python file, ~150-200 lines. No framework dependencies. Direct HTTP calls to the Modal vLLM endpoint via `httpx`.

**Prompt design**: ReAct-style with structured output format:
```
Thought: what I'm thinking
Action: tool_name
Action Input: {"arg": "value"}
```
Parse with regex, not JSON (more robust to model output quirks).

**SWE-bench integration**: Clone the task repo, apply the base commit, run the agent, collect the patch, run tests.

## Files to Modify
- `examples/custom_harness.py` — new file

## Tests Needed
- N/A (proof of concept)

## Notes
