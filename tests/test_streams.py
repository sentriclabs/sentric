"""Tests for streaming response support."""

import tempfile
import pytest
from sentric import TrajectoryCollector, trace, atrace
from sentric.streams import TracedStream, TracedAsyncStream
from tests.mock_responses import (
    make_openai_stream,
    make_openai_stream_chunks,
    make_anthropic_stream,
)


def test_streaming_openai():
    """@trace wraps OpenAI streaming response and logs accumulated content."""
    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
    )

    @trace(collector)
    def call_llm(messages):
        return make_openai_stream(
            content="The fix is simple.",
            input_tokens=50,
            output_tokens=100,
        )

    stream = call_llm(messages=[{"role": "user", "content": "Fix it."}])

    # Stream should be a TracedStream
    assert isinstance(stream, TracedStream)

    # Consume the stream
    chunks = list(stream)
    assert len(chunks) > 0

    # After consumption, message should be logged
    assert len(collector.messages) == 2  # user + assistant
    assert collector.messages[1]["role"] == "assistant"
    assert collector.messages[1]["content"] == "The fix is simple."
    assert collector._input_tokens == 50
    assert collector._output_tokens == 100


def test_streaming_anthropic():
    """@trace wraps Anthropic streaming response and logs accumulated content."""
    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "claude-sonnet-4-20250514", "version": "base", "provider": "anthropic"},
    )

    @trace(collector)
    def call_llm(messages):
        return make_anthropic_stream(
            text="Here is the fix.",
            input_tokens=80,
            output_tokens=120,
        )

    stream = call_llm(messages=[{"role": "user", "content": "Fix it."}])
    assert isinstance(stream, TracedStream)

    chunks = list(stream)
    assert len(chunks) > 0

    assert len(collector.messages) == 2
    assert collector.messages[1]["role"] == "assistant"
    assert collector.messages[1]["content"] == "Here is the fix."
    assert collector._input_tokens == 80
    assert collector._output_tokens == 120


@pytest.mark.asyncio
async def test_streaming_async():
    """@atrace wraps async streaming response and logs accumulated content."""
    import sys
    import types as _types

    mod_name = "openai.lib.streaming._async"
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _types.ModuleType(mod_name)

    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
    )

    chunks = make_openai_stream_chunks(
        content="Async streaming works.",
        input_tokens=30,
        output_tokens=60,
    )

    class AsyncStream:
        __module__ = mod_name

        def __init__(self, chunks):
            self._chunks = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration

    @atrace(collector)
    async def call_llm(messages):
        return AsyncStream(chunks)

    stream = await call_llm(messages=[{"role": "user", "content": "Test async."}])
    assert isinstance(stream, TracedAsyncStream)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) > 0
    assert len(collector.messages) == 2
    assert collector.messages[1]["role"] == "assistant"
    assert collector.messages[1]["content"] == "Async streaming works."
    assert collector._input_tokens == 30
    assert collector._output_tokens == 60


def test_streaming_passthrough():
    """TracedStream preserves original chunk data for the caller."""
    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
    )

    @trace(collector)
    def call_llm(messages):
        return make_openai_stream(content="Hello world.")

    stream = call_llm(messages=[{"role": "user", "content": "Hi"}])

    # Should be wrapped
    assert isinstance(stream, TracedStream)

    # Consume and verify we get chunks with expected attributes
    received = list(stream)
    assert len(received) > 0
    # Content chunks should have choices attribute
    assert hasattr(received[0], "choices")


def test_streaming_token_count():
    """Token usage from streaming final chunk is tracked."""
    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
    )

    @trace(collector)
    def call_llm(messages):
        return make_openai_stream(
            content="Token tracking test.",
            input_tokens=200,
            output_tokens=300,
        )

    stream = call_llm(messages=[{"role": "user", "content": "Count tokens."}])
    list(stream)  # consume

    assert collector._input_tokens == 200
    assert collector._output_tokens == 300
    assert collector._total_tokens == 500


def test_streaming_partial_consumption():
    """Partial stream consumption logs partial content on close."""
    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
    )

    @trace(collector)
    def call_llm(messages):
        return make_openai_stream(
            content="one two three four five six seven eight nine ten",
            input_tokens=10,
            output_tokens=20,
        )

    stream = call_llm(messages=[{"role": "user", "content": "Count."}])

    # Consume only 3 chunks
    for i, _ in enumerate(stream):
        if i >= 2:
            break

    # Close the stream, which should log partial content
    stream.close()

    assert len(collector.messages) == 2
    assert collector.messages[1]["role"] == "assistant"
    # Should have partial content (first 3 words)
    content = collector.messages[1]["content"]
    assert content is not None
    assert len(content) > 0
    # Should NOT have the full content
    assert content != "one two three four five six seven eight nine ten"


def test_non_streaming_still_works():
    """@trace still handles non-streaming responses normally."""
    from tests.mock_responses import make_openai_response

    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
    )

    @trace(collector)
    def call_llm(messages):
        return make_openai_response(content="Not streaming.", total_tokens=100)

    response = call_llm(messages=[{"role": "user", "content": "Hi"}])

    # Should NOT be wrapped in TracedStream
    assert not isinstance(response, TracedStream)
    assert len(collector.messages) == 2
    assert collector.messages[1]["content"] == "Not streaming."
