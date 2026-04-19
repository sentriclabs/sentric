"""Tests for format importers (Feature 09)."""

import json
import logging
import tempfile
from pathlib import Path

from sentric.importers import from_langsmith, from_openai_messages, from_wandb, import_directory


# --- OpenAI Messages ---

def test_openai_messages_basic():
    """Import basic OpenAI message log."""
    data = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    episode = from_openai_messages(data)

    assert len(episode["messages"]) == 3
    assert episode["messages"][0]["role"] == "system"
    assert episode["messages"][1]["role"] == "user"
    assert episode["messages"][2]["content"] == "Hi there!"
    assert episode["model"]["name"] == "gpt-4o"
    assert episode["model"]["provider"] == "openai"
    assert episode["total_tokens"] == 150
    assert episode["input_tokens"] == 100
    assert episode["output_tokens"] == 50


def test_openai_messages_with_tool_calls():
    """Import OpenAI messages with tool calls."""
    data = {
        "messages": [
            {"role": "user", "content": "List files"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
                    }
                ],
            },
            {"role": "tool", "content": "file1.py", "tool_call_id": "call_1"},
        ],
        "model": "gpt-4o",
    }
    episode = from_openai_messages(data)

    assert len(episode["messages"]) == 3
    assert episode["messages"][1]["tool_calls"][0]["name"] == "bash"
    assert episode["messages"][1]["tool_calls"][0]["id"] == "call_1"
    assert episode["messages"][2]["tool_call_id"] == "call_1"


def test_openai_messages_unmapped_fields(caplog):
    """Warns about unmapped fields."""
    data = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o",
        "custom_field": "value",
        "another": 123,
    }
    with caplog.at_level(logging.WARNING, logger="sentric.importers"):
        episode = from_openai_messages(data)

    assert any("Unmapped fields" in r.message for r in caplog.records)
    assert len(episode["messages"]) == 1


def test_openai_messages_minimal():
    """Import with minimal fields."""
    data = {"messages": [{"role": "user", "content": "hello"}]}
    episode = from_openai_messages(data)

    assert episode["model"]["name"] == ""
    assert episode["total_tokens"] is None
    assert episode["task_id"] == ""


def test_openai_messages_with_task_id():
    """Import with task_id and domain."""
    data = {
        "messages": [{"role": "user", "content": "hi"}],
        "task_id": "my-task",
        "domain": "code",
    }
    episode = from_openai_messages(data)
    assert episode["task_id"] == "my-task"
    assert episode["domain"] == "code"


# --- LangSmith ---

def test_langsmith_basic():
    """Import basic LangSmith run."""
    data = {
        "id": "run-123",
        "name": "AgentExecutor",
        "run_type": "chain",
        "inputs": {"input": "Fix the bug"},
        "outputs": {"output": "Done"},
        "child_runs": [],
        "extra": {"metadata": {"task_id": "task-1", "domain": "code"}},
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-01T00:01:00Z",
    }
    episode = from_langsmith(data)

    assert episode["episode_id"] == "run-123"
    assert episode["task_id"] == "task-1"
    assert episode["domain"] == "code"
    assert episode["duration_ms"] == 60000
    assert len(episode["messages"]) == 2  # user + assistant from inputs/outputs
    assert episode["messages"][0]["content"] == "Fix the bug"
    assert episode["messages"][1]["content"] == "Done"


def test_langsmith_with_child_runs():
    """Import LangSmith run with child LLM and tool runs."""
    data = {
        "id": "run-456",
        "name": "Agent",
        "run_type": "chain",
        "inputs": {},
        "outputs": {},
        "child_runs": [
            {
                "name": "ChatOpenAI",
                "run_type": "llm",
                "inputs": {
                    "messages": [
                        [{"role": "user", "content": "Fix it"}]
                    ]
                },
                "outputs": {
                    "generations": [
                        [{"text": "Looking at it.", "message": {"role": "assistant", "content": "Looking at it."}}]
                    ]
                },
                "extra": {"invocation_params": {"model_name": "gpt-4o"}},
            },
            {
                "name": "bash",
                "run_type": "tool",
                "inputs": {"input": "ls"},
                "outputs": {"output": "file1.py"},
            },
        ],
        "extra": {},
    }
    episode = from_langsmith(data)

    assert episode["model"]["name"] == "gpt-4o"
    assert len(episode["messages"]) >= 3  # user, assistant, tool assistant+result
    assert episode["messages"][0]["role"] == "user"
    assert episode["messages"][1]["role"] == "assistant"


def test_langsmith_with_tool_calls_in_messages():
    """LangSmith messages with additional_kwargs tool calls."""
    data = {
        "id": "run-789",
        "name": "Agent",
        "run_type": "chain",
        "inputs": {},
        "outputs": {},
        "child_runs": [
            {
                "name": "ChatOpenAI",
                "run_type": "llm",
                "inputs": {"messages": []},
                "outputs": {
                    "generations": [[{
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "additional_kwargs": {
                                "tool_calls": [{
                                    "id": "call_abc",
                                    "function": {"name": "search", "arguments": '{"q":"test"}'},
                                }]
                            }
                        }
                    }]]
                },
                "extra": {},
            },
        ],
        "extra": {},
    }
    episode = from_langsmith(data)

    assistant_msgs = [m for m in episode["messages"] if m["role"] == "assistant" and "tool_calls" in m]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0]["tool_calls"][0]["name"] == "search"
    assert assistant_msgs[0]["tool_calls"][0]["id"] == "call_abc"


def test_langsmith_token_usage():
    """LangSmith token usage extraction."""
    data = {
        "id": "run-tok",
        "name": "Agent",
        "run_type": "chain",
        "inputs": {"input": "hi"},
        "outputs": {"output": "hello"},
        "child_runs": [],
        "extra": {},
        "token_usage": {"prompt_tokens": 50, "completion_tokens": 30},
    }
    episode = from_langsmith(data)
    assert episode["input_tokens"] == 50
    assert episode["output_tokens"] == 30
    assert episode["total_tokens"] == 80


def test_langsmith_unmapped_fields(caplog):
    """Warns about unmapped fields."""
    data = {
        "id": "run-unk",
        "name": "Agent",
        "run_type": "chain",
        "inputs": {},
        "outputs": {},
        "child_runs": [],
        "extra": {},
        "weird_field": True,
    }
    with caplog.at_level(logging.WARNING, logger="sentric.importers"):
        from_langsmith(data)

    assert any("Unmapped fields" in r.message for r in caplog.records)


# --- W&B ---

def test_wandb_basic():
    """Import basic W&B trace."""
    data = {
        "span_id": "span-123",
        "trace_id": "trace-456",
        "name": "AgentRun",
        "kind": "AGENT",
        "inputs": {"query": "Fix the bug"},
        "outputs": {"response": "Done"},
        "child_spans": [],
        "attributes": {"metadata": {"task_id": "task-1"}},
        "start_time_ms": 1704067200000,
        "end_time_ms": 1704067260000,
    }
    episode = from_wandb(data)

    assert episode["episode_id"] == "span-123"
    assert episode["task_id"] == "task-1"
    assert episode["duration_ms"] == 60000
    assert episode["created_at"] is not None
    assert len(episode["messages"]) == 2
    assert episode["messages"][0]["content"] == "Fix the bug"


def test_wandb_with_child_spans():
    """Import W&B trace with LLM and tool spans."""
    data = {
        "span_id": "span-789",
        "name": "Agent",
        "kind": "AGENT",
        "inputs": {},
        "outputs": {},
        "child_spans": [
            {
                "name": "llm_call",
                "kind": "LLM",
                "inputs": {"messages": [{"role": "user", "content": "Fix it"}]},
                "outputs": {
                    "choices": [{"message": {"role": "assistant", "content": "On it."}}]
                },
                "attributes": {
                    "model": "gpt-4o",
                    "token_usage": {"prompt_tokens": 50, "completion_tokens": 30},
                },
            },
            {
                "name": "tool_call",
                "kind": "TOOL",
                "inputs": {"tool_name": "bash", "tool_input": "ls"},
                "outputs": {"tool_output": "file1.py"},
            },
        ],
        "attributes": {},
    }
    episode = from_wandb(data)

    assert episode["model"]["name"] == "gpt-4o"
    assert episode["input_tokens"] == 50
    assert episode["output_tokens"] == 30
    assert episode["total_tokens"] == 80
    assert len(episode["messages"]) >= 3


def test_wandb_unmapped_fields(caplog):
    """Warns about unmapped fields."""
    data = {
        "span_id": "span-unk",
        "name": "Agent",
        "kind": "AGENT",
        "inputs": {},
        "outputs": {},
        "child_spans": [],
        "attributes": {},
        "custom_thing": "value",
    }
    with caplog.at_level(logging.WARNING, logger="sentric.importers"):
        from_wandb(data)

    assert any("Unmapped fields" in r.message for r in caplog.records)


# --- Batch Import ---

def test_import_directory():
    """Batch import from a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            data = {
                "messages": [{"role": "user", "content": f"msg {i}"}],
                "model": "gpt-4o",
            }
            Path(tmpdir, f"file_{i}.json").write_text(json.dumps(data))

        # Also add a non-JSON file that should be skipped
        Path(tmpdir, "readme.txt").write_text("not json")

        episodes = import_directory(tmpdir, format="openai")

    assert len(episodes) == 3
    for ep in episodes:
        assert ep["model"]["name"] == "gpt-4o"
        assert len(ep["messages"]) == 1


def test_import_directory_langsmith():
    """Batch import LangSmith format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {
            "id": "run-1",
            "name": "Agent",
            "run_type": "chain",
            "inputs": {"input": "hello"},
            "outputs": {"output": "world"},
            "child_runs": [],
            "extra": {},
        }
        Path(tmpdir, "run.json").write_text(json.dumps(data))

        episodes = import_directory(tmpdir, format="langsmith")

    assert len(episodes) == 1
    assert episodes[0]["episode_id"] == "run-1"


def test_import_directory_bad_format():
    """Raises on unknown format."""
    import pytest
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unknown format"):
            import_directory(tmpdir, format="invalid")


def test_import_directory_skips_bad_files(caplog):
    """Skips files with bad JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "good.json").write_text(json.dumps({
            "messages": [{"role": "user", "content": "hi"}],
        }))
        Path(tmpdir, "bad.json").write_text("{invalid json")

        with caplog.at_level(logging.WARNING, logger="sentric.importers"):
            episodes = import_directory(tmpdir, format="openai")

    assert len(episodes) == 1
    assert any("Skipping" in r.message for r in caplog.records)
