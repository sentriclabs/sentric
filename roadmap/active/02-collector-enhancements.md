# Feature 02: Collector Enhancements — add_step, to_dict, reset fix, env capture

**Priority**: P0
**Effort**: M
**Status**: planned

## Summary
Four targeted improvements to `TrajectoryCollector` that make it production-ready:
1. `add_step()` — log a full tool-call round-trip (assistant + tool) in one call
2. `to_dict()` — get the episode as a dict without writing to disk
3. Fix `reset()` — preserve metadata when not explicitly overridden
4. Environment capture — auto-record python version, key package versions, git hash

## Requirements
- `add_step(content, tool_name, tool_args, tool_result, tool_call_id=None)` adds both the assistant message with tool_call and the tool response in one call. Auto-generates tool_call_id if not provided.
- `to_dict()` returns the same dict that `save_episode()` would write, without disk I/O
- `reset(metadata=None)` preserves existing metadata when `metadata` is not passed (currently resets to `{}`)
- `capture_env()` stores python version, platform, and installed package versions for sentric + optional deps (openai, anthropic) in metadata under `_env` key
- All methods must be zero-allocation where possible (reuse existing structures)

## Design Notes
**Performance**: `add_step()` should be a thin wrapper over two `add_message()` calls — no extra copies. `to_dict()` should build the dict lazily using the same logic as `save_episode()` but skip file I/O. `capture_env()` runs once at init time, not per-message.

**reset() fix**: Change signature to `reset(task_id=None, metadata=_SENTINEL)` using a sentinel to distinguish "not passed" from "passed as None".

## Files to Modify
- `sentric/collector.py` — all four changes

## Tests Needed
- `test_add_step()` — verify both messages added with correct structure
- `test_to_dict()` — verify output matches save_episode JSON structure
- `test_reset_preserves_metadata()` — verify metadata preserved when not passed
- `test_reset_clears_metadata_when_passed()` — verify explicit None clears it
- `test_capture_env()` — verify env data captured in metadata

## Notes
- Eng review: capture_env() git hash via subprocess with graceful fallback to None on FileNotFoundError
- Eng review: use `__slots__` with all attributes declared upfront (including future OTel, executor as None)
- Eng review: reset() sentinel approach confirmed
