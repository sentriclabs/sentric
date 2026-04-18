"""Optional OpenTelemetry integration for trajectory collection.

Zero overhead when OTel is not installed — all functions are no-ops
that return immediately. Uses functools.lru_cache to cache the import check.
"""

import functools


@functools.lru_cache(maxsize=1)
def _get_tracer():
    """Lazily import and return an OTel tracer, or None if unavailable."""
    try:
        from opentelemetry import trace
        return trace.get_tracer("sentric")
    except ImportError:
        return None


def start_episode_span(collector):
    """Start a parent span for an episode. Returns the span or None."""
    tracer = _get_tracer()
    if tracer is None:
        return None

    span = tracer.start_span(
        name=f"sentric.episode.{collector.task_id}",
        attributes={
            "sentric.episode_id": collector.episode_id,
            "sentric.task_id": collector.task_id,
            "sentric.domain": collector.domain,
            "sentric.model.name": collector.model.get("name", ""),
            "sentric.model.provider": collector.model.get("provider", ""),
        },
    )
    return span


def end_episode_span(span, collector):
    """End the episode span with final attributes."""
    if span is None:
        return

    total_tokens = collector._input_tokens + collector._output_tokens
    attrs = {
        "sentric.message_count": len(collector.messages),
        "sentric.total_tokens": total_tokens,
        "sentric.input_tokens": collector._input_tokens,
        "sentric.output_tokens": collector._output_tokens,
    }

    cost = collector._calculate_cost()
    if cost is not None:
        attrs["sentric.total_cost_usd"] = cost

    for key, value in attrs.items():
        span.set_attribute(key, value)

    span.end()


def emit_message_event(span, role, content=None, tool_calls=None, tool_call_id=None):
    """Add a span event for a message. No-op if span is None."""
    if span is None:
        return

    attributes = {"sentric.message.role": role}

    if content is not None:
        # Truncate content to avoid huge span events
        attributes["sentric.message.content_length"] = len(content)
        attributes["sentric.message.content_preview"] = content[:200]

    if tool_calls:
        attributes["sentric.message.tool_count"] = len(tool_calls)
        tool_names = [tc.get("name", "") for tc in tool_calls]
        attributes["sentric.message.tool_names"] = ",".join(tool_names)

    if tool_call_id:
        attributes["sentric.message.tool_call_id"] = tool_call_id

    span.add_event(f"message.{role}", attributes=attributes)
