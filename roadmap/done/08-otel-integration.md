# Feature 08: OpenTelemetry Integration

**Priority**: P1
**Effort**: L
**Status**: planned

## Summary
Optional OpenTelemetry span emission alongside JSON logging. Companies with existing observability stacks (Datadog, Honeycomb, Grafana) can pipe trajectory data into their existing dashboards without changing their monitoring setup.

## Requirements
- Optional dependency: `sentric[otel]` installs `opentelemetry-api`
- When OTel is available, each `add_message()` creates a span event
- Each `save_episode()` creates a parent span for the full episode
- Span attributes include: role, token count, tool name (if tool call), model
- Zero overhead when OTel is not installed (lazy import, no-op fallback)
- Trace context propagation for distributed agent systems

## Design Notes
**Performance**: Lazy import of `opentelemetry` — if not installed, all OTel code paths are no-ops with zero overhead. Use `functools.lru_cache` to cache the import check.

**Architecture**: Add a `_maybe_emit_span()` method to collector that's a no-op by default. When OTel is detected, replace with real span emission.

## Files to Modify
- `sentric/otel.py` — new file, OTel integration
- `sentric/collector.py` — optional span emission hooks
- `pyproject.toml` — add `[otel]` extra

## Tests Needed
- `test_otel_disabled()` — verify no errors when OTel not installed
- `test_otel_spans()` — verify spans emitted when OTel available (mock)
- `test_otel_attributes()` — verify span attributes correct

## Notes
