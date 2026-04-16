# Feature 07: Automatic Cost Tracking

**Priority**: P1
**Effort**: M
**Status**: planned

## Summary
Automatically estimate and track cost per episode based on model pricing and token counts. Enterprises care deeply about cost — this makes trajectories immediately actionable for cost analysis without external tools.

## Requirements
- Built-in pricing table for common models (GPT-4o, Claude Sonnet, etc.)
- `collector.add_cost(amount)` for manual cost tracking
- Auto-calculate cost from token counts when model pricing is known
- Cost included in episode output as `total_cost_usd`
- Support custom pricing via `model={"name": ..., "pricing": {"input": 0.003, "output": 0.015}}`
- Separate input/output token tracking (currently only total)

## Design Notes
**Performance**: Pricing lookup is a dict lookup — O(1). No external calls. Pricing table is a module-level constant, loaded once.

**Token split**: Need to modify parsers to return `(messages, input_tokens, output_tokens)` instead of `(messages, total_tokens)`. This is a breaking change to the parser interface — handle with a migration path.

## Files to Modify
- `sentric/pricing.py` — new file, pricing table
- `sentric/collector.py` — add cost tracking, split input/output tokens
- `sentric/parsers.py` — return split token counts
- `sentric/trace.py` — pass split counts to collector

## Tests Needed
- `test_auto_cost_calculation()` — verify cost calculated from tokens + pricing
- `test_manual_cost()` — verify add_cost()
- `test_custom_pricing()` — verify custom pricing in model dict
- `test_unknown_model_no_cost()` — graceful handling of unknown models

## Notes
- Eng review: CLEAN BREAK in v0.2. No backward compat shim for custom normalizers. Near-zero users = near-zero breakage. Document in changelog.
