# Feature 12: SWE-bench PoC — Modal Deployment + mini-swe-agent

**Priority**: P0
**Effort**: XL
**Status**: planned

## Summary
Deploy Qwen2.5-Coder-7B-Instruct on Modal via vLLM, run mini-swe-agent on 5-10 SWE-bench Lite tasks with Sentric instrumentation. This is the proof-of-concept demonstrating the collect→analyze workflow. $10 Modal budget.

## Requirements
- Modal function serving Qwen2.5-Coder-7B-Instruct via vLLM with OpenAI-compatible API
- mini-swe-agent configured to use the Modal endpoint
- Sentric `@trace` wrapping the LLM calls (either via monkey-patching mini-swe-agent or via a LiteLLM callback)
- Run on 5-10 selected SWE-bench Lite tasks
- All trajectories saved locally in Sentric format
- Stay within $10 Modal budget

## Design Notes
**Modal setup**: Use `modal.App` with a GPU function (A10G). Serve vLLM as an OpenAI-compatible endpoint. Use `modal deploy` for persistent deployment during the run.

**mini-swe-agent integration**: Configure via its config to point to the Modal endpoint. LiteLLM should handle this with `api_base` override.

**Task selection**: Pick 5-10 tasks that are well-known and have clear pass/fail (e.g., django, sympy, scikit-learn tasks from SWE-bench Lite).

**Budget management**: A10G is ~$0.50/hr. 5-10 tasks at ~10 min each = ~1-2 hours = ~$1. Well within budget.

## Files to Modify
- `examples/modal_vllm.py` — new file, Modal deployment
- `examples/run_swebench.py` — new file, mini-swe-agent runner with Sentric

## Tests Needed
- N/A (proof of concept, not library code)

## Notes
Need to install: modal, mini-swe-agent, litellm
