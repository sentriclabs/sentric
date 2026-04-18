"""Tests for TrajectoryCollector core functionality."""

import json
import re
import tempfile
import pytest
import sentric
from sentric import TrajectoryCollector


def test_version():
    """sentric.__version__ is a string matching semver pattern."""
    assert isinstance(sentric.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+$", sentric.__version__)


def test_basic_episode():
    """A 5-step episode saves valid JSON matching the schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="django__django-11099",
            domain="code",
            model={"name": "Qwen/Qwen2.5-Coder-7B", "version": "base", "provider": "local"},
            output_dir=tmpdir,
            metadata={"repo": "django/django", "base_commit": "abc123"},
        )

        collector.add_message(role="system", content="You are a software engineer.")
        collector.add_message(role="user", content="Fix the regex bug.")
        collector.add_message(
            role="assistant",
            content="Let me find the file.",
            tool_calls=[{"id": "call_1", "name": "bash", "arguments": '{"command": "grep -r Validator"}'}],
        )
        collector.add_message(role="tool", content="validators.py:class ASCIIUsernameValidator", tool_call_id="call_1")
        collector.add_message(
            role="assistant",
            content="Changing $ to \\Z in the regex.",
            tool_calls=[{"id": "call_2", "name": "file_write", "arguments": '{"path": "validators.py"}'}],
        )

        collector.add_tokens(input_tokens=150, output_tokens=200)
        collector.add_tokens(input_tokens=80, output_tokens=130)
        path = collector.save_episode()

        assert path.exists()
        data = json.loads(path.read_text())

        # Top-level fields
        assert data["episode_id"] == collector.episode_id
        assert data["task_id"] == "django__django-11099"
        assert data["domain"] == "code"
        assert data["created_at"] is not None
        assert data["duration_ms"] >= 0

        # Model
        assert data["model"]["name"] == "Qwen/Qwen2.5-Coder-7B"
        assert data["model"]["version"] == "base"
        assert data["model"]["provider"] == "local"

        # Reward fields null before scoring
        assert data["reward"] is None
        assert data["success"] is None
        assert data["verifier"] is None
        assert data["verified_at"] is None

        # Messages
        assert len(data["messages"]) == 5
        assert data["messages"][0]["role"] == "system"
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][2]["role"] == "assistant"
        assert data["messages"][2]["tool_calls"][0]["name"] == "bash"
        assert data["messages"][3]["role"] == "tool"
        assert data["messages"][3]["tool_call_id"] == "call_1"
        assert data["messages"][4]["tool_calls"][0]["name"] == "file_write"

        # No stray fields
        assert "tool_calls" not in data["messages"][0]
        assert "tool_calls" not in data["messages"][1]
        assert "tool_calls" not in data["messages"][3]
        assert "tool_call_id" not in data["messages"][0]
        assert "tool_call_id" not in data["messages"][2]

        # Token tracking
        assert data["total_tokens"] == 560

        # Metadata
        assert data["metadata"]["repo"] == "django/django"


def test_validation_rejects_bad_input():
    """Invalid roles, missing tool_call_id, and incomplete tool_calls raise ValueError."""
    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "test", "version": "base", "provider": "local"},
    )

    # Bad role
    try:
        collector.add_message(role="invalid", content="hello")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Tool message without tool_call_id
    try:
        collector.add_message(role="tool", content="result")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Tool call missing required fields
    try:
        collector.add_message(
            role="assistant", content="hi",
            tool_calls=[{"id": "call_1", "name": "bash"}],  # missing "arguments"
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_reset():
    """Reset clears state for a new episode while preserving model/domain."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="task_1", domain="code",
            model={"name": "test", "version": "base", "provider": "local"},
            output_dir=tmpdir,
        )

        collector.add_message(role="user", content="first episode")
        path1 = collector.save_episode()
        old_id = collector.episode_id

        collector.reset(task_id="task_2")
        assert collector.episode_id != old_id
        assert collector.task_id == "task_2"
        assert len(collector.messages) == 0

        collector.add_message(role="user", content="second episode")
        path2 = collector.save_episode()

        assert path1 != path2
        data1 = json.loads(path1.read_text())
        data2 = json.loads(path2.read_text())
        assert data1["task_id"] == "task_1"
        assert data2["task_id"] == "task_2"
        assert data1["episode_id"] != data2["episode_id"]


def test_no_tokens_saved_as_null():
    """If no tokens are tracked, total_tokens is null in the output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "test", "version": "base", "provider": "local"},
            output_dir=tmpdir,
        )
        collector.add_message(role="user", content="hello")
        path = collector.save_episode()
        data = json.loads(path.read_text())
        assert data["total_tokens"] is None


def test_add_step():
    """add_step() adds both assistant and tool messages with correct structure."""
    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "test", "version": "base", "provider": "local"},
    )

    collector.add_step(
        content="Let me run a command.",
        tool_name="bash",
        tool_args='{"command": "ls"}',
        tool_result="file1.py\nfile2.py",
        tool_call_id="call_99",
    )

    assert len(collector.messages) == 2
    assistant_msg = collector.messages[0]
    tool_msg = collector.messages[1]

    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "Let me run a command."
    assert len(assistant_msg["tool_calls"]) == 1
    assert assistant_msg["tool_calls"][0]["id"] == "call_99"
    assert assistant_msg["tool_calls"][0]["name"] == "bash"
    assert assistant_msg["tool_calls"][0]["arguments"] == '{"command": "ls"}'

    assert tool_msg["role"] == "tool"
    assert tool_msg["content"] == "file1.py\nfile2.py"
    assert tool_msg["tool_call_id"] == "call_99"


def test_add_step_auto_id():
    """add_step() auto-generates tool_call_id when not provided."""
    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "test", "version": "base", "provider": "local"},
    )

    collector.add_step(
        content="Running grep.",
        tool_name="bash",
        tool_args='{"command": "grep foo"}',
        tool_result="match found",
    )

    assert len(collector.messages) == 2
    call_id = collector.messages[0]["tool_calls"][0]["id"]
    assert call_id.startswith("call_")
    assert collector.messages[1]["tool_call_id"] == call_id


def test_to_dict():
    """to_dict() returns the same structure as save_episode JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "test", "version": "base", "provider": "local"},
            output_dir=tmpdir,
            metadata={"key": "value"},
        )
        collector.add_message(role="user", content="hello")
        collector.add_tokens(input_tokens=40, output_tokens=60)

        d = collector.to_dict()

        assert d["episode_id"] == collector.episode_id
        assert d["task_id"] == "test"
        assert d["domain"] == "code"
        assert d["model"]["name"] == "test"
        assert len(d["messages"]) == 1
        assert d["total_tokens"] == 100
        assert d["input_tokens"] == 40
        assert d["output_tokens"] == 60
        assert d["metadata"]["key"] == "value"
        assert d["reward"] is None
        assert d["success"] is None
        assert d["created_at"] is not None
        assert d["duration_ms"] >= 0


def test_to_dict_matches_save():
    """to_dict() output matches what save_episode() writes to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "test", "version": "base", "provider": "local"},
            output_dir=tmpdir,
        )
        collector.add_message(role="user", content="hello")

        d = collector.to_dict()
        path = collector.save_episode()
        saved = json.loads(path.read_text())

        # Same keys
        assert set(d.keys()) == set(saved.keys())
        # Same core data (timestamps may differ slightly)
        assert d["episode_id"] == saved["episode_id"]
        assert d["task_id"] == saved["task_id"]
        assert d["messages"] == saved["messages"]


def test_reset_preserves_metadata():
    """reset() preserves metadata when metadata is not passed."""
    collector = TrajectoryCollector(
        task_id="task_1", domain="code",
        model={"name": "test", "version": "base", "provider": "local"},
        metadata={"repo": "django/django"},
    )

    collector.add_message(role="user", content="hello")
    collector.reset(task_id="task_2")

    assert collector.task_id == "task_2"
    assert collector.metadata == {"repo": "django/django"}
    assert len(collector.messages) == 0


def test_reset_clears_metadata_when_passed():
    """reset() clears metadata when explicitly passed as None."""
    collector = TrajectoryCollector(
        task_id="task_1", domain="code",
        model={"name": "test", "version": "base", "provider": "local"},
        metadata={"repo": "django/django"},
    )

    collector.reset(task_id="task_2", metadata=None)
    assert collector.metadata == {}


def test_reset_replaces_metadata_when_passed():
    """reset() replaces metadata when a new dict is passed."""
    collector = TrajectoryCollector(
        task_id="task_1", domain="code",
        model={"name": "test", "version": "base", "provider": "local"},
        metadata={"repo": "django/django"},
    )

    collector.reset(task_id="task_2", metadata={"repo": "flask/flask"})
    assert collector.metadata == {"repo": "flask/flask"}


def test_capture_env():
    """capture_env() populates metadata._env with python version and packages."""
    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "test", "version": "base", "provider": "local"},
    )

    collector.capture_env()

    assert "_env" in collector.metadata
    env = collector.metadata["_env"]
    assert "python_version" in env
    assert "platform" in env
    assert "packages" in env
    assert "sentric" in env["packages"]
    assert "git_hash" in env


def test_auto_cost_calculation():
    """Cost is auto-calculated from tokens + built-in pricing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
            output_dir=tmpdir,
        )
        collector.add_message(role="user", content="hello")
        collector.add_tokens(input_tokens=1000, output_tokens=500)
        path = collector.save_episode()
        data = json.loads(path.read_text())

        # gpt-4o: input=$2.50/M, output=$10.00/M
        expected = 1000 * 2.50 / 1_000_000 + 500 * 10.00 / 1_000_000
        assert data["total_cost_usd"] == pytest.approx(expected)
        assert data["input_tokens"] == 1000
        assert data["output_tokens"] == 500
        assert data["total_tokens"] == 1500


def test_manual_cost():
    """add_cost() manually tracks cost in USD."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "custom-model", "version": "v1", "provider": "local"},
            output_dir=tmpdir,
        )
        collector.add_message(role="user", content="hello")
        collector.add_cost(0.05)
        collector.add_cost(0.03)
        path = collector.save_episode()
        data = json.loads(path.read_text())

        assert data["total_cost_usd"] == pytest.approx(0.08)


def test_custom_pricing():
    """Custom pricing in model dict overrides built-in pricing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={
                "name": "gpt-4o",  # has built-in pricing
                "version": "custom",
                "provider": "openai",
                "pricing": {"input": 0.001, "output": 0.002},  # custom override
            },
            output_dir=tmpdir,
        )
        collector.add_message(role="user", content="hello")
        collector.add_tokens(input_tokens=100, output_tokens=200)
        path = collector.save_episode()
        data = json.loads(path.read_text())

        expected = 100 * 0.001 + 200 * 0.002
        assert data["total_cost_usd"] == pytest.approx(expected)


def test_unknown_model_no_cost():
    """Unknown model without custom pricing results in null cost."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "totally-unknown-model", "version": "v1", "provider": "local"},
            output_dir=tmpdir,
        )
        collector.add_message(role="user", content="hello")
        collector.add_tokens(input_tokens=100, output_tokens=200)
        path = collector.save_episode()
        data = json.loads(path.read_text())

        assert data["total_cost_usd"] is None
        assert data["total_tokens"] == 300


def test_split_tokens_in_output():
    """input_tokens and output_tokens are tracked separately in JSON output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "test", "version": "base", "provider": "local"},
            output_dir=tmpdir,
        )
        collector.add_message(role="user", content="hello")
        collector.add_tokens(input_tokens=100, output_tokens=200)
        collector.add_tokens(input_tokens=50, output_tokens=75)
        path = collector.save_episode()
        data = json.loads(path.read_text())

        assert data["input_tokens"] == 150
        assert data["output_tokens"] == 275
        assert data["total_tokens"] == 425


def test_backward_compat_add_tokens():
    """add_tokens(500) as positional arg still tracks tokens."""
    collector = TrajectoryCollector(
        task_id="test", domain="code",
        model={"name": "test", "version": "base", "provider": "local"},
    )
    collector.add_tokens(500)
    assert collector._total_tokens == 500
