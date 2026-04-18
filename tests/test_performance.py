"""Tests for performance optimizations (Feature 11)."""

import json
import tempfile
from concurrent.futures import Future
from unittest import mock

from sentric import TrajectoryCollector
from sentric import _json


def test_orjson_fallback():
    """Works correctly without orjson — falls back to stdlib json."""
    _json._has_orjson.cache_clear()

    with mock.patch.dict("sys.modules", {"orjson": None}):
        _json._has_orjson.cache_clear()
        assert _json._has_orjson() is False

        data = {"key": "value", "nested": {"a": 1}}
        result = _json.dumps(data)
        assert json.loads(result) == data

        result_bytes = _json.dumps_bytes(data)
        assert json.loads(result_bytes) == data

        # No indent
        compact = _json.dumps(data, indent=False)
        assert " " not in compact
        assert json.loads(compact) == data

    _json._has_orjson.cache_clear()


def test_json_dumps_indented():
    """dumps produces indented output by default."""
    _json._has_orjson.cache_clear()

    data = {"a": 1, "b": [2, 3]}
    result = _json.dumps(data)
    assert "\n" in result
    parsed = json.loads(result)
    assert parsed == data


def test_json_dumps_bytes():
    """dumps_bytes returns bytes."""
    _json._has_orjson.cache_clear()

    data = {"x": "hello"}
    result = _json.dumps_bytes(data)
    assert isinstance(result, bytes)
    assert json.loads(result) == data


def test_save_episode_uses_bytes():
    """save_episode writes bytes to disk (via dumps_bytes)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="test", domain="code",
            model={"name": "test", "version": "1", "provider": "local"},
            output_dir=tmpdir,
        )
        collector.add_message(role="user", content="hello")
        path = collector.save_episode()

        data = json.loads(path.read_text())
        assert data["task_id"] == "test"
        assert len(data["messages"]) == 1


def test_save_episode_async():
    """save_episode_async completes correctly in background thread."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrajectoryCollector(
            task_id="async-test", domain="code",
            model={"name": "test", "version": "1", "provider": "local"},
            output_dir=tmpdir,
        )
        collector.add_message(role="user", content="async hello")
        collector.add_tokens(input_tokens=10, output_tokens=20)

        future = collector.save_episode_async()
        assert isinstance(future, Future)

        path = future.result(timeout=5)
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["task_id"] == "async-test"
        assert len(data["messages"]) == 1
        assert data["total_tokens"] == 30


def test_save_episode_async_error_logged():
    """save_episode_async logs warning on failure."""
    import logging

    logger = logging.getLogger("sentric")
    messages = []

    class _Handler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = _Handler()
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    try:
        collector = TrajectoryCollector(
            task_id="fail-test", domain="code",
            model={"name": "test", "version": "1", "provider": "local"},
            output_dir="/nonexistent/deeply/nested/path/that/should/not/exist",
        )
        collector.add_message(role="user", content="will fail")

        future = collector.save_episode_async()
        try:
            future.result(timeout=5)
        except OSError:
            pass

        assert any("save_episode_async failed" in m for m in messages)
    finally:
        logger.removeHandler(handler)


def test_save_episode_async_multiple():
    """Multiple async saves complete without interfering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        futures = []
        for i in range(5):
            collector = TrajectoryCollector(
                task_id=f"multi-{i}", domain="code",
                model={"name": "test", "version": "1", "provider": "local"},
                output_dir=tmpdir,
            )
            collector.add_message(role="user", content=f"msg {i}")
            futures.append(collector.save_episode_async())

        paths = [f.result(timeout=10) for f in futures]
        assert len(paths) == 5
        assert all(p.exists() for p in paths)

        for i, path in enumerate(paths):
            data = json.loads(path.read_text())
            assert data["task_id"] == f"multi-{i}"
