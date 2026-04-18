"""The @trace and @atrace decorators for automatic trajectory collection.

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

For async functions:

    @atrace(collector)
    async def call_llm(messages, **kwargs):
        return await openai.chat.completions.create(messages=messages, model="gpt-4o", **kwargs)

For custom LLM wrappers, provide a normalizer:

    def my_normalizer(response) -> tuple[list[dict], int, int]:
        return [{"role": "assistant", "content": response.text}], input_tokens, output_tokens

    @trace(collector, normalizer=my_normalizer)
    def call_custom_llm(messages):
        return custom_client.generate(messages)
"""

import functools
from sentric.parsers import detect_and_parse
from sentric.streams import TracedStream, TracedAsyncStream


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
            except (AttributeError, KeyError, TypeError) as e:
                raise ValueError(f"Unrecognized tool_call format: {tc!r}") from e
    return normalized


def _pre_call(collector, args, kwargs):
    """Shared pre-call logic: extract and log new input messages."""
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


def _post_call(collector, response, normalizer):
    """Shared post-call logic: parse response and log output messages."""
    messages, input_tokens, output_tokens = detect_and_parse(response, normalizer=normalizer)
    for msg in messages:
        collector.add_message(
            role=msg["role"],
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls"),
            tool_call_id=msg.get("tool_call_id"),
        )

    if input_tokens > 0 or output_tokens > 0:
        collector.add_tokens(input_tokens=input_tokens, output_tokens=output_tokens)


def _detect_stream_type(response) -> str | None:
    """Detect if a response is a streaming response and return the provider type."""
    type_name = type(response).__name__
    module = getattr(type(response), "__module__", "") or ""

    if "openai" in module and "Stream" in type_name:
        return "openai"

    if "anthropic" in module and "Stream" in type_name:
        return "anthropic"

    return None


def trace(collector, normalizer=None):
    """Decorator that logs LLM calls to a TrajectoryCollector.

    Supports both regular and streaming responses. Streaming responses
    are wrapped in a TracedStream that logs content when exhausted.

    Args:
        collector: A TrajectoryCollector instance to log messages to.
        normalizer: Optional function that takes a response and returns
            (messages: list[dict], input_tokens: int, output_tokens: int).
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _pre_call(collector, args, kwargs)
            response = fn(*args, **kwargs)

            # Check for streaming response
            stream_type = _detect_stream_type(response)
            if stream_type and normalizer is None:
                return TracedStream(response, collector, stream_type=stream_type)

            _post_call(collector, response, normalizer)
            return response

        return wrapper

    return decorator


def atrace(collector, normalizer=None):
    """Async decorator that logs LLM calls to a TrajectoryCollector.

    Supports both regular and streaming responses. Async streaming responses
    are wrapped in a TracedAsyncStream.

    Args:
        collector: A TrajectoryCollector instance to log messages to.
        normalizer: Optional function that takes a response and returns
            (messages: list[dict], input_tokens: int, output_tokens: int).
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            _pre_call(collector, args, kwargs)
            response = await fn(*args, **kwargs)

            # Check for streaming response (sync or async)
            stream_type = _detect_stream_type(response)
            if stream_type and normalizer is None:
                if hasattr(response, "__aiter__"):
                    return TracedAsyncStream(response, collector, stream_type=stream_type)
                return TracedStream(response, collector, stream_type=stream_type)

            _post_call(collector, response, normalizer)
            return response

        return wrapper

    return decorator
