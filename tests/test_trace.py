"""Tests for the @trace decorator.

Covers OpenAI text, OpenAI tool calls, Anthropic text, Anthropic tool use,
Anthropic mixed blocks, custom normalizers, string fallback, input message
capture, deduplication across turns, and full episode save.

No real API calls — all responses are mocked.
"""

import json
import tempfile
from sentric import TrajectoryCollector, trace
from tests.mock_responses import make_openai_response, make_anthropic_response


# --- OpenAI ---


def test_openai_text_response():
    """@trace auto-parses a plain text OpenAI response."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
            output_dir=tmpdir,
        )

        @trace(collector)
        def call_llm(messages):
            return make_openai_response(content="The fix is to use \\Z instead of $.", total_tokens=200)

        call_llm(messages=[
            {"role": "system", "content": "You are a software engineer."},
            {"role": "user", "content": "Fix the regex bug."},
        ])

        assert len(collector.messages) == 3
        assert collector.messages[0]["role"] == "system"
        assert collector.messages[1]["role"] == "user"
        assert collector.messages[2]["role"] == "assistant"
        assert "\\Z" in collector.messages[2]["content"]
        assert collector._total_tokens == 200


def test_openai_tool_calls():
    """@trace auto-parses OpenAI tool calls with correct schema fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
            output_dir=tmpdir,
        )

        @trace(collector)
        def call_llm(messages):
            return make_openai_response(
                content="Let me search for that.",
                tool_calls=[{
                    "id": "call_abc",
                    "name": "bash",
                    "arguments": '{"command": "grep -r validator"}',
                }],
                total_tokens=300,
            )

        call_llm(messages=[{"role": "user", "content": "Find the validator."}])

        assistant_msg = collector.messages[-1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["tool_calls"][0]["name"] == "bash"
        assert assistant_msg["tool_calls"][0]["id"] == "call_abc"
        assert assistant_msg["tool_calls"][0]["arguments"] == '{"command": "grep -r validator"}'
        assert collector._total_tokens == 300


def test_openai_multiple_tool_calls():
    """@trace handles parallel tool calls in a single OpenAI response."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
            output_dir=tmpdir,
        )

        @trace(collector)
        def call_llm(messages):
            return make_openai_response(
                content=None,
                tool_calls=[
                    {"id": "call_1", "name": "get_time", "arguments": '{"tz": "UTC"}'},
                    {"id": "call_2", "name": "calculator", "arguments": '{"expr": "2+2"}'},
                    {"id": "call_3", "name": "get_fact", "arguments": '{"topic": "math"}'},
                ],
                total_tokens=400,
            )

        call_llm(messages=[{"role": "user", "content": "Do three things."}])

        assistant_msg = collector.messages[-1]
        assert len(assistant_msg["tool_calls"]) == 3
        assert assistant_msg["content"] is None
        names = [tc["name"] for tc in assistant_msg["tool_calls"]]
        assert names == ["get_time", "calculator", "get_fact"]


# --- Anthropic ---


def test_anthropic_text_response():
    """@trace auto-parses a plain text Anthropic response."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "claude-sonnet-4-20250514", "version": "base", "provider": "anthropic"},
            output_dir=tmpdir,
        )

        @trace(collector)
        def call_llm(messages):
            return make_anthropic_response(text="Here's the fix.", input_tokens=80, output_tokens=120)

        call_llm(messages=[{"role": "user", "content": "Fix this."}])

        assert len(collector.messages) == 2
        assert collector.messages[1]["role"] == "assistant"
        assert collector.messages[1]["content"] == "Here's the fix."
        assert collector._total_tokens == 200


def test_anthropic_tool_use():
    """@trace auto-parses Anthropic tool use blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "claude-sonnet-4-20250514", "version": "base", "provider": "anthropic"},
            output_dir=tmpdir,
        )

        @trace(collector)
        def call_llm(messages):
            return make_anthropic_response(
                text="Let me check the file.",
                tool_uses=[{
                    "id": "toolu_123",
                    "name": "read_file",
                    "input": {"path": "validators.py"},
                }],
            )

        call_llm(messages=[{"role": "user", "content": "Look at validators.py"}])

        assistant_msg = collector.messages[-1]
        assert assistant_msg["tool_calls"][0]["name"] == "read_file"
        assert assistant_msg["tool_calls"][0]["id"] == "toolu_123"
        # Anthropic input is a dict; we JSON-encode it for our schema
        assert json.loads(assistant_msg["tool_calls"][0]["arguments"]) == {"path": "validators.py"}


def test_anthropic_mixed_blocks():
    """@trace handles Anthropic responses with text and multiple tool_use blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "claude-sonnet-4-20250514", "version": "base", "provider": "anthropic"},
            output_dir=tmpdir,
        )

        @trace(collector)
        def call_llm(messages):
            return make_anthropic_response(
                text="I'll search in two places.",
                tool_uses=[
                    {"id": "toolu_1", "name": "bash", "input": {"command": "grep -r foo src/"}},
                    {"id": "toolu_2", "name": "bash", "input": {"command": "grep -r foo lib/"}},
                ],
                input_tokens=100,
                output_tokens=200,
            )

        call_llm(messages=[{"role": "user", "content": "Find foo"}])

        assistant_msg = collector.messages[-1]
        assert assistant_msg["content"] == "I'll search in two places."
        assert len(assistant_msg["tool_calls"]) == 2
        assert assistant_msg["tool_calls"][0]["name"] == "bash"
        assert assistant_msg["tool_calls"][1]["name"] == "bash"
        assert collector._total_tokens == 300


def test_anthropic_tool_only_no_text():
    """@trace handles Anthropic responses with only tool_use blocks and no text."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "claude-sonnet-4-20250514", "version": "base", "provider": "anthropic"},
            output_dir=tmpdir,
        )

        @trace(collector)
        def call_llm(messages):
            return make_anthropic_response(
                text=None,
                tool_uses=[{"id": "toolu_1", "name": "calculator", "input": {"expr": "1+1"}}],
            )

        call_llm(messages=[{"role": "user", "content": "Calculate"}])

        assistant_msg = collector.messages[-1]
        assert assistant_msg["content"] is None
        assert len(assistant_msg["tool_calls"]) == 1


# --- Custom normalizer ---


def test_custom_normalizer():
    """@trace uses a custom normalizer when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "custom-model", "version": "v1", "provider": "local"},
            output_dir=tmpdir,
        )

        def my_normalizer(response):
            return [{"role": "assistant", "content": response["text"]}], response["tokens"]

        @trace(collector, normalizer=my_normalizer)
        def call_llm(messages):
            return {"text": "Custom response here.", "tokens": 42}

        call_llm(messages=[{"role": "user", "content": "Hello"}])

        assert collector.messages[-1]["content"] == "Custom response here."
        assert collector._total_tokens == 42


def test_custom_normalizer_without_token_count():
    """Normalizer that returns only messages (no token tuple) still works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "custom", "version": "v1", "provider": "local"},
            output_dir=tmpdir,
        )

        def my_normalizer(response):
            return [{"role": "assistant", "content": response}]

        @trace(collector, normalizer=my_normalizer)
        def call_llm(messages):
            return "just a string"

        call_llm(messages=[{"role": "user", "content": "Hello"}])

        assert collector.messages[-1]["content"] == "just a string"
        assert collector._total_tokens == 0


# --- String fallback ---


def test_fallback_unknown_type():
    """@trace stringifies unknown response types as a fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "custom", "version": "v1", "provider": "local"},
            output_dir=tmpdir,
        )

        @trace(collector)
        def call_llm(messages):
            return "Just a plain string response"

        call_llm(messages=[{"role": "user", "content": "Hello"}])

        assert collector.messages[-1]["content"] == "Just a plain string response"
        assert collector._total_tokens == 0


def test_fallback_dict_response():
    """@trace stringifies a dict response when no normalizer is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "custom", "version": "v1", "provider": "local"},
            output_dir=tmpdir,
        )

        @trace(collector)
        def call_llm(messages):
            return {"result": "some value", "score": 0.95}

        call_llm(messages=[{"role": "user", "content": "Hello"}])

        content = collector.messages[-1]["content"]
        assert "some value" in content
        assert "0.95" in content


# --- Input message handling ---


def test_input_messages_as_positional_arg():
    """@trace captures input messages passed as the first positional arg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "gpt-4o", "version": "base", "provider": "openai"},
            output_dir=tmpdir,
        )

        @trace(collector)
        def call_llm(messages):
            return make_openai_response(content="Done.")

        # Positional, not keyword
        call_llm([{"role": "user", "content": "Fix it"}])

        assert collector.messages[0]["role"] == "user"
        assert collector.messages[0]["content"] == "Fix it"
        assert collector.messages[1]["role"] == "assistant"


def test_no_duplicate_messages_across_turns():
    """@trace doesn't re-log messages already in the collector on subsequent turns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "gpt-4o", "version": "base", "provider": "openai"},
            output_dir=tmpdir,
        )

        collector.add_message(role="system", content="You are helpful.")

        @trace(collector)
        def call_llm(messages):
            return make_openai_response(content="Sure!")

        call_llm(messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Help me."},
        ])

        # system (manual) + user (from trace) + assistant (from trace) = 3, not 4
        assert len(collector.messages) == 3
        assert collector.messages[0]["role"] == "system"
        assert collector.messages[1]["role"] == "user"
        assert collector.messages[2]["role"] == "assistant"


def test_multi_turn_with_tool_calls_in_history():
    """@trace handles multi-turn conversations where history includes OpenAI-format tool_calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "gpt-4o", "version": "base", "provider": "openai"},
            output_dir=tmpdir,
        )

        @trace(collector)
        def call_llm(messages):
            return make_openai_response(content="The answer is 42.")

        # Simulate turn 1 already happened: system + user + assistant(tool_call) + tool
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Calculate something."},
            {
                "role": "assistant",
                "content": None,
                # OpenAI dict format with nested "function" key
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expr": "6*7"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]

        # Manually log turn 1
        collector.add_message(role="system", content="You are helpful.")
        collector.add_message(role="user", content="Calculate something.")
        collector.add_message(
            role="assistant", content=None,
            tool_calls=[{"id": "call_1", "name": "calculator", "arguments": '{"expr": "6*7"}'}],
        )
        collector.add_message(role="tool", content="42", tool_call_id="call_1")

        # Turn 2: pass full history including OpenAI-format tool_calls
        call_llm(messages=messages)

        # Should have 4 (manual) + 1 (assistant from turn 2) = 5
        assert len(collector.messages) == 5
        assert collector.messages[4]["role"] == "assistant"
        assert collector.messages[4]["content"] == "The answer is 42."


# --- Full episode save ---


def test_full_episode_save():
    """End-to-end: @trace through save_episode produces valid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="django__django-11099", domain="code",
            model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
            output_dir=tmpdir,
            metadata={"repo": "django/django"},
        )

        @trace(collector)
        def call_llm(messages):
            return make_openai_response(content="Fixed.", total_tokens=500)

        collector.add_message(role="system", content="You are a software engineer.")
        call_llm(messages=[
            {"role": "system", "content": "You are a software engineer."},
            {"role": "user", "content": "Fix the bug."},
        ])

        path = collector.save_episode()
        data = json.loads(path.read_text())

        assert data["task_id"] == "django__django-11099"
        assert data["domain"] == "code"
        assert data["total_tokens"] == 500
        assert len(data["messages"]) == 3
        assert data["reward"] is None
        assert data["metadata"]["repo"] == "django/django"

        # Verify JSON is parseable and all required top-level keys exist
        required_keys = [
            "episode_id", "task_id", "domain", "model", "messages",
            "reward", "success", "verifier", "verified_at",
            "created_at", "duration_ms", "total_tokens", "metadata",
        ]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"
