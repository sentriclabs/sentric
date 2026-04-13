"""The @trace decorator for automatic trajectory collection.

Usage:

    from sentric import TrajectoryCollector, trace

    collector = TrajectoryCollector(
        task_id="my-task",
        domain="code",
        model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
    )

    @trace(collector)
    def call_llm(messages, **kwargs):
        return openai.chat.completions.create(messages=messages, model="gpt-4o", **kwargs)

For custom LLM wrappers, provide a normalizer:

    def my_normalizer(response) -> tuple[list[dict], int]:
        return [{"role": "assistant", "content": response.text}], response.token_count

    @trace(collector, normalizer=my_normalizer)
    def call_custom_llm(messages):
        return custom_client.generate(messages)
"""

import json
import functools
from sentric.parsers import detect_and_parse


def _extract_input_messages(args, kwargs) -> list[dict]:
    """Try to find the input messages from the function arguments."""
    messages = kwargs.get("messages")
    if messages is not None:
        return list(messages)

    if args:
        first = args[0]
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict) and "role" in first[0]:
            return list(first)

    return []


def _normalize_tool_calls(tool_calls):
    """Normalize tool_calls from OpenAI's format to our schema format.

    OpenAI format: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
    Our format:    {"id": "...", "name": "...", "arguments": "..."}
    """
    if not tool_calls:
        return None

    normalized = []
    for tc in tool_calls:
        if isinstance(tc, dict) and "function" in tc:
            normalized.append({
                "id": tc.get("id", ""),
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            })
        elif isinstance(tc, dict) and "name" in tc:
            normalized.append(tc)
        else:
            try:
                normalized.append({
                    "id": getattr(tc, "id", "") or tc.get("id", ""),
                    "name": getattr(tc, "function", tc).name if hasattr(tc, "function") else tc["name"],
                    "arguments": getattr(tc, "function", tc).arguments if hasattr(tc, "function") else tc["arguments"],
                })
            except (AttributeError, KeyError, TypeError):
                normalized.append({"id": str(tc), "name": "unknown", "arguments": "{}"})
    return normalized


def trace(collector, normalizer=None):
    """Decorator that logs LLM calls to a TrajectoryCollector.

    Args:
        collector: A TrajectoryCollector instance to log messages to.
        normalizer: Optional function that takes a response and returns
            (messages: list[dict], token_count: int). If provided, this
            is used instead of auto-detection.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            input_messages = _extract_input_messages(args, kwargs)
            already_logged = len(collector.messages)

            new_messages = input_messages[already_logged:]
            for msg in new_messages:
                collector.add_message(
                    role=msg["role"],
                    content=msg.get("content"),
                    tool_calls=_normalize_tool_calls(msg.get("tool_calls")),
                    tool_call_id=msg.get("tool_call_id"),
                )

            response = fn(*args, **kwargs)

            messages, tokens = detect_and_parse(response, normalizer=normalizer)
            for msg in messages:
                collector.add_message(
                    role=msg["role"],
                    content=msg.get("content"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                )

            if tokens > 0:
                collector.add_tokens(tokens)

            return response

        return wrapper

    return decorator
