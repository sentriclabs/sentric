# Feature 11: Performance Optimizations

**Priority**: P1
**Effort**: M
**Status**: planned

## Summary
Make trajectory collection near-zero overhead. Developers won't instrument their agents if it slows them down. Target: <100μs per add_message(), <1ms per save_episode() for typical trajectories.

## Requirements
- Pre-allocate message list with estimated capacity
- Use `__slots__` on TrajectoryCollector to avoid dict overhead
- Use `orjson` (optional dep) for JSON serialization when available, fall back to stdlib `json`
- Lazy timestamp formatting (compute only on save, not on every message)
- Buffer messages in a flat list, not nested dicts — build the nested structure only at save time
- `save_episode()` should write directly to file with buffered I/O, not build string then write
- Add `save_episode_async()` for non-blocking disk writes (background thread)
- Benchmark suite to track performance across versions

## Design Notes
**orjson**: 3-10x faster than stdlib json for serialization. Make it optional: `sentric[fast]` extra. Detect at import time, use if available.

**save_episode_async()**: Use `concurrent.futures.ThreadPoolExecutor` with a single worker thread. Fire-and-forget for non-critical writes. Return a `Future` for callers who want to wait.

**Benchmark suite**: Use `timeit` in a dedicated `benchmarks/` directory. Track: messages/sec, save latency, memory usage per 1000 messages.

## Files to Modify
- `sentric/collector.py` — slots, pre-allocation, lazy timestamps
- `sentric/_json.py` — new file, orjson/json abstraction
- `benchmarks/` — new directory
- `pyproject.toml` — add `[fast]` extra

## Tests Needed
- `test_orjson_fallback()` — works without orjson
- `test_save_async()` — async save completes correctly
- Benchmark suite (not unit tests)

## Notes
- Eng review: save_episode_async() uses MODULE-LEVEL singleton ThreadPoolExecutor(max_workers=1) + atexit.register(executor.shutdown)
- Eng review: returned Future must propagate exceptions; log warning via logging.getLogger('sentric') if Future is not checked
- Eng review: __slots__ declared upfront with all known attributes including _executor, _otel_tracer
