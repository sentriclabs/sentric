"""Tests for trajectory viewer CLI (Feature 10)."""

import json
import tempfile
from io import StringIO
from pathlib import Path

from sentric.cli import main, _view_single, _view_directory, _build_parser, _format_duration, _format_tokens


def _make_episode(task_id="test-task", n_messages=3, tokens=True):
    """Create a sample episode dict."""
    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({"role": "user", "content": "Fix the bug in models.py"})
    if n_messages > 2:
        messages.append({
            "role": "assistant",
            "content": "Let me look at the file.",
            "tool_calls": [{"id": "call_1", "name": "bash", "arguments": '{"cmd": "cat models.py"}'}],
        })
    if n_messages > 3:
        messages.append({"role": "tool", "content": "class Model:\n    pass", "tool_call_id": "call_1"})

    ep = {
        "episode_id": "ep-123",
        "task_id": task_id,
        "domain": "code",
        "model": {"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
        "messages": messages[:n_messages],
        "reward": None,
        "success": None,
        "verifier": None,
        "verified_at": None,
        "created_at": "2024-01-01T00:00:00+00:00",
        "duration_ms": 5432,
        "total_tokens": 300 if tokens else None,
        "input_tokens": 200 if tokens else None,
        "output_tokens": 100 if tokens else None,
        "metadata": {},
    }
    return ep


def _write_episode(tmpdir, episode, name="ep-123.json"):
    path = Path(tmpdir) / name
    path.write_text(json.dumps(episode, indent=2))
    return path


# --- Format helpers ---

def test_format_duration():
    assert _format_duration(None) == "n/a"
    assert _format_duration(500) == "500ms"
    assert _format_duration(1500) == "1.5s"
    assert _format_duration(65000) == "1m 5s"


def test_format_tokens():
    assert _format_tokens(None) == "n/a"
    assert _format_tokens(500) == "500"
    assert _format_tokens(1500) == "1,500"


# --- View single ---

def test_cli_view_single():
    """Default view shows stats and turns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = _make_episode()
        path = _write_episode(tmpdir, ep)

        out = StringIO()
        parser = _build_parser()
        args = parser.parse_args(["view", str(path)])
        _view_single(path, args, out)

        output = out.getvalue()
        assert "ep-123" in output
        assert "test-task" in output
        assert "gpt-4o" in output
        assert "5.4s" in output
        assert "[user]" in output
        assert "[assistant]" in output


def test_cli_view_stats_only():
    """--stats shows only summary stats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = _make_episode()
        path = _write_episode(tmpdir, ep)

        out = StringIO()
        parser = _build_parser()
        args = parser.parse_args(["view", str(path), "--stats"])
        _view_single(path, args, out)

        output = out.getvalue()
        assert "Episode Stats" in output
        assert "300" in output
        # Should not have turn-by-turn
        assert "[user]" not in output


def test_cli_view_json():
    """--json outputs raw JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = _make_episode()
        path = _write_episode(tmpdir, ep)

        out = StringIO()
        parser = _build_parser()
        args = parser.parse_args(["view", str(path), "--json"])
        _view_single(path, args, out)

        parsed = json.loads(out.getvalue())
        assert parsed["episode_id"] == "ep-123"
        assert parsed["task_id"] == "test-task"


def test_cli_view_turns():
    """--turns shows turn-by-turn messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = _make_episode(n_messages=4)
        path = _write_episode(tmpdir, ep)

        out = StringIO()
        parser = _build_parser()
        args = parser.parse_args(["view", str(path), "--turns"])
        _view_single(path, args, out)

        output = out.getvalue()
        assert "[system]" in output
        assert "[user]" in output
        assert "[assistant]" in output
        assert "[tool]" in output
        assert "call_1" in output


def test_cli_view_full():
    """--full shows full content without truncation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        long_content = "x" * 500
        ep = _make_episode()
        ep["messages"][1]["content"] = long_content
        path = _write_episode(tmpdir, ep)

        # Without --full: truncated
        out = StringIO()
        parser = _build_parser()
        args = parser.parse_args(["view", str(path)])
        _view_single(path, args, out)
        assert "500 chars" in out.getvalue()

        # With --full: not truncated
        out = StringIO()
        args = parser.parse_args(["view", str(path), "--full"])
        _view_single(path, args, out)
        assert long_content in out.getvalue()


def test_cli_view_tool_calls():
    """Tool calls are displayed with arrow notation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = _make_episode(n_messages=4)
        path = _write_episode(tmpdir, ep)

        out = StringIO()
        parser = _build_parser()
        args = parser.parse_args(["view", str(path)])
        _view_single(path, args, out)

        output = out.getvalue()
        assert "bash" in output


# --- View directory ---

def test_cli_view_directory():
    """View directory lists trajectories with stats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            ep = _make_episode(task_id=f"task-{i}")
            _write_episode(tmpdir, ep, name=f"ep-{i}.json")

        # Also add a non-JSON file
        Path(tmpdir, "readme.txt").write_text("not json")

        out = StringIO()
        _view_directory(Path(tmpdir), out)

        output = out.getvalue()
        assert "3 trajectory file(s)" in output
        assert "ep-0.json" in output
        assert "ep-1.json" in output
        assert "ep-2.json" in output


def test_cli_view_directory_empty():
    """Empty directory shows message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = StringIO()
        _view_directory(Path(tmpdir), out)
        assert "No trajectory files found" in out.getvalue()


# --- Main entry point ---

def test_cli_main_view(capsys):
    """main() dispatches to view command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = _make_episode()
        path = _write_episode(tmpdir, ep)

        main(["view", str(path), "--json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["episode_id"] == "ep-123"


def test_cli_main_no_command(capsys):
    """main() with no command shows help and exits."""
    import pytest
    with pytest.raises(SystemExit):
        main([])


def test_cli_main_missing_file(capsys):
    """main() with missing file exits with error."""
    import pytest
    with pytest.raises(SystemExit):
        main(["view", "/nonexistent/file.json"])


def test_cli_view_no_tokens():
    """View episode without token data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = _make_episode(tokens=False)
        path = _write_episode(tmpdir, ep)

        out = StringIO()
        parser = _build_parser()
        args = parser.parse_args(["view", str(path), "--stats"])
        _view_single(path, args, out)

        output = out.getvalue()
        assert "n/a" in output
