"""Tests for the episode loader utilities."""

import json
import tempfile
from pathlib import Path

import pytest
from sentric import TrajectoryCollector, load_episode, load_episodes, score_episode, export_jsonl


def _make_episode(tmpdir, task_id="test-task", domain="code", **kwargs):
    """Helper: create and save an episode, return (path, data)."""
    collector = TrajectoryCollector(
        task_id=task_id,
        domain=domain,
        model={"name": "test", "version": "base", "provider": "local"},
        output_dir=tmpdir,
        **kwargs,
    )
    collector.add_message(role="user", content="hello")
    path = collector.save_episode()
    data = json.loads(path.read_text())
    return path, data


def test_load_episode():
    """load_episode loads a saved trajectory and returns correct structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, original = _make_episode(tmpdir)

        loaded = load_episode(path)

        assert loaded["episode_id"] == original["episode_id"]
        assert loaded["task_id"] == "test-task"
        assert loaded["domain"] == "code"
        assert len(loaded["messages"]) == 1
        assert loaded["messages"][0]["role"] == "user"


def test_load_episode_missing_file():
    """load_episode raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_episode("/nonexistent/path/episode.json")


def test_load_episode_bad_json():
    """load_episode raises JSONDecodeError for invalid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / "bad.json"
        bad_path.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            load_episode(bad_path)


def test_load_episodes():
    """load_episodes loads all trajectories from a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_episode(tmpdir, task_id="task-1")
        _make_episode(tmpdir, task_id="task-2")
        _make_episode(tmpdir, task_id="task-3")

        episodes = load_episodes(tmpdir)

        assert len(episodes) == 3
        task_ids = {e["task_id"] for e in episodes}
        assert task_ids == {"task-1", "task-2", "task-3"}


def test_load_episodes_with_filter():
    """load_episodes applies filter function correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_episode(tmpdir, task_id="task-1", metadata={"domain": "code"})
        _make_episode(tmpdir, task_id="task-2", metadata={"domain": "math"})
        _make_episode(tmpdir, task_id="task-3", metadata={"domain": "code"})

        episodes = load_episodes(
            tmpdir,
            filter_fn=lambda e: e["metadata"].get("domain") == "code",
        )

        assert len(episodes) == 2
        assert all(e["metadata"]["domain"] == "code" for e in episodes)


def test_load_episodes_empty_dir():
    """load_episodes returns empty list for empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        episodes = load_episodes(tmpdir)
        assert episodes == []


def test_load_episodes_ignores_non_json():
    """load_episodes ignores non-JSON files in directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_episode(tmpdir, task_id="task-1")
        # Create a non-JSON file
        (Path(tmpdir) / "readme.txt").write_text("not a trajectory")

        episodes = load_episodes(tmpdir)
        assert len(episodes) == 1


def test_score_episode():
    """score_episode updates scoring fields on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _make_episode(tmpdir)

        updated = score_episode(path, reward=1.0, success=True, verifier="human")

        assert updated["reward"] == 1.0
        assert updated["success"] is True
        assert updated["verifier"] == "human"
        assert updated["verified_at"] is not None

        # Verify it was written to disk
        on_disk = json.loads(path.read_text())
        assert on_disk["reward"] == 1.0
        assert on_disk["success"] is True
        assert on_disk["verifier"] == "human"
        assert on_disk["verified_at"] is not None


def test_score_episode_partial():
    """score_episode only updates fields that are explicitly passed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, _ = _make_episode(tmpdir)

        # First score with reward only
        score_episode(path, reward=0.5)

        on_disk = json.loads(path.read_text())
        assert on_disk["reward"] == 0.5
        assert on_disk["success"] is None
        assert on_disk["verifier"] is None

        # Then add success without changing reward
        score_episode(path, success=False)

        on_disk = json.loads(path.read_text())
        assert on_disk["reward"] == 0.5  # preserved
        assert on_disk["success"] is False


def test_score_episode_missing_file():
    """score_episode raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        score_episode("/nonexistent/episode.json", reward=1.0)


def test_export_jsonl():
    """export_jsonl writes valid JSONL with one episode per line."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _, ep1 = _make_episode(tmpdir, task_id="task-1")
        _, ep2 = _make_episode(tmpdir, task_id="task-2")

        output_path = Path(tmpdir) / "export" / "trajectories.jsonl"
        result = export_jsonl([ep1, ep2], output_path)

        assert result == output_path
        assert output_path.exists()

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2

        parsed_1 = json.loads(lines[0])
        parsed_2 = json.loads(lines[1])
        assert parsed_1["task_id"] == "task-1"
        assert parsed_2["task_id"] == "task-2"


def test_export_jsonl_roundtrip():
    """Episodes exported to JSONL can be loaded back identically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _, ep1 = _make_episode(tmpdir, task_id="task-1")
        _, ep2 = _make_episode(tmpdir, task_id="task-2")

        output_path = Path(tmpdir) / "roundtrip.jsonl"
        export_jsonl([ep1, ep2], output_path)

        # Load back
        loaded = []
        with open(output_path) as f:
            for line in f:
                loaded.append(json.loads(line))

        assert len(loaded) == 2
        assert loaded[0] == ep1
        assert loaded[1] == ep2


def test_export_jsonl_empty():
    """export_jsonl with empty list produces empty file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "empty.jsonl"
        export_jsonl([], output_path)

        assert output_path.exists()
        assert output_path.read_text() == ""
