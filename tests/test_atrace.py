"""Tests for the @atrace async decorator.

Covers async OpenAI text, async tool calls, async Anthropic, async custom
normalizer, and async deduplication. No real API calls — all responses are mocked.
"""

import json
import tempfile
import pytest
from sentric import TrajectoryCollector, atrace
from tests.mock_responses import make_openai_response, make_anthropic_response


@pytest.mark.asyncio
async def test_atrace_text_response():
    """@atrace auto-parses a plain text OpenAI response."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
            output_dir=tmpdir,
        )

        @atrace(collector)
        async def call_llm(messages):
            return make_openai_response(content="The fix is to use \\Z instead of $.", total_tokens=200)

        await call_llm(messages=[
            {"role": "system", "content": "You are a software engineer."},
            {"role": "user", "content": "Fix the regex bug."},
        ])

        assert len(collector.messages) == 3
        assert collector.messages[0]["role"] == "system"
        assert collector.messages[1]["role"] == "user"
        assert collector.messages[2]["role"] == "assistant"
        assert "\\Z" in collector.messages[2]["content"]
        assert collector._total_tokens == 200


@pytest.mark.asyncio
async def test_atrace_tool_calls():
    """@atrace auto-parses OpenAI tool calls with correct schema fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
            output_dir=tmpdir,
        )

        @atrace(collector)
        async def call_llm(messages):
            return make_openai_response(
                content="Let me search for that.",
                tool_calls=[{
                    "id": "call_abc",
                    "name": "bash",
                    "arguments": '{"command": "grep -r validator"}',
                }],
                total_tokens=300,
            )

        await call_llm(messages=[{"role": "user", "content": "Find the validator."}])

        assistant_msg = collector.messages[-1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["tool_calls"][0]["name"] == "bash"
        assert assistant_msg["tool_calls"][0]["id"] == "call_abc"
        assert collector._total_tokens == 300


@pytest.mark.asyncio
async def test_atrace_anthropic():
    """@atrace auto-parses Anthropic tool use blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "claude-sonnet-4-20250514", "version": "base", "provider": "anthropic"},
            output_dir=tmpdir,
        )

        @atrace(collector)
        async def call_llm(messages):
            return make_anthropic_response(
                text="Let me check the file.",
                tool_uses=[{
                    "id": "toolu_123",
                    "name": "read_file",
                    "input": {"path": "validators.py"},
                }],
                input_tokens=80,
                output_tokens=120,
            )

        await call_llm(messages=[{"role": "user", "content": "Look at validators.py"}])

        assert len(collector.messages) == 2
        assistant_msg = collector.messages[-1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Let me check the file."
        assert assistant_msg["tool_calls"][0]["name"] == "read_file"
        assert assistant_msg["tool_calls"][0]["id"] == "toolu_123"
        assert json.loads(assistant_msg["tool_calls"][0]["arguments"]) == {"path": "validators.py"}
        assert collector._total_tokens == 200


@pytest.mark.asyncio
async def test_atrace_custom_normalizer():
    """@atrace uses a custom normalizer when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "custom-model", "version": "v1", "provider": "local"},
            output_dir=tmpdir,
        )

        def my_normalizer(response):
            return [{"role": "assistant", "content": response["text"]}], response["tokens"]

        @atrace(collector, normalizer=my_normalizer)
        async def call_llm(messages):
            return {"text": "Custom async response.", "tokens": 42}

        await call_llm(messages=[{"role": "user", "content": "Hello"}])

        assert collector.messages[-1]["content"] == "Custom async response."
        assert collector._total_tokens == 42


@pytest.mark.asyncio
async def test_atrace_deduplication():
    """@atrace doesn't re-log messages already in the collector."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "gpt-4o", "version": "base", "provider": "openai"},
            output_dir=tmpdir,
        )

        collector.add_message(role="system", content="You are helpful.")

        @atrace(collector)
        async def call_llm(messages):
            return make_openai_response(content="Sure!")

        await call_llm(messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Help me."},
        ])

        # system (manual) + user (from atrace) + assistant (from atrace) = 3, not 4
        assert len(collector.messages) == 3
        assert collector.messages[0]["role"] == "system"
        assert collector.messages[1]["role"] == "user"
        assert collector.messages[2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_atrace_returns_response():
    """@atrace returns the original response object."""
    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "gpt-4o", "version": "base", "provider": "openai"},
    )

    @atrace(collector)
    async def call_llm(messages):
        return make_openai_response(content="Hello!")

    response = await call_llm(messages=[{"role": "user", "content": "Hi"}])
    assert response.choices[0].message.content == "Hello!"
