# Feature 03: Async Trace Decorator

**Priority**: P0
**Effort**: M
**Status**: planned

## Summary
Add `@atrace(collector)` decorator for async LLM functions. Modern production LLM usage is heavily async (httpx, aiohttp, async OpenAI/Anthropic clients). Without this, the SDK is unusable in most production codebases.

## Requirements
- `@atrace(collector)` works identically to `@trace(collector)` but wraps async functions
- Same input message extraction logic
- Same auto-detection and parsing of OpenAI/Anthropic responses
- Same deduplication behavior
- Same normalizer support
- Must not block the event loop (no sync I/O in the hot path)
- Import: `from sentric import TrajectoryCollector, trace, atrace`

## Design Notes
**Performance**: The decorator itself should add minimal overhead. The message extraction and parsing logic is already synchronous and CPU-bound (no I/O), so it can run inline in the async context without concern. The key is that `await fn(*args, **kwargs)` replaces `fn(*args, **kwargs)`.

**Implementation**: Factor out the shared pre/post logic from `trace()` into helper functions, then both `trace()` and `atrace()` call the same helpers. Avoids code duplication.

## Files to Modify
- `sentric/trace.py` — add `atrace()`, refactor shared logic
- `sentric/__init__.py` — export `atrace`

## Tests Needed
- `test_atrace_text_response()` — basic async OpenAI text
- `test_atrace_tool_calls()` — async with tool calls
- `test_atrace_anthropic()` — async Anthropic response
- `test_atrace_custom_normalizer()` — async with custom normalizer
- `test_atrace_deduplication()` — async dedup behavior matches sync

## Notes
