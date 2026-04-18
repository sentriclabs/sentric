"""Tests for OpenTelemetry integration.

Tests both the no-OTel case (zero overhead) and the OTel-available case (mocked).
"""

import json
import tempfile
from unittest import mock

from sentric import TrajectoryCollector
from sentric import otel as _otel


def test_otel_disabled():
    """No errors when OTel is not installed — all operations are no-ops."""
    # Clear the lru_cache to force re-check
    _otel._get_tracer.cache_clear()

    with mock.patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}):
        _otel._get_tracer.cache_clear()
        tracer = _otel._get_tracer()
        assert tracer is None

        # All otel functions should be no-ops with None span
        _otel.emit_message_event(None, "user", content="hello")
        _otel.end_episode_span(None, None)

    # Restore cache
    _otel._get_tracer.cache_clear()


def test_otel_disabled_collector():
    """Collector works normally when OTel is not installed."""
    _otel._get_tracer.cache_clear()

    with mock.patch.object(_otel, "_get_tracer", return_value=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrajectoryCollector(
                task_id="test", domain="code",
                model={"name": "test", "version": "base", "provider": "local"},
                output_dir=tmpdir,
            )
            collector.add_message(role="user", content="hello")
            path = collector.save_episode()

            data = json.loads(path.read_text())
            assert data["task_id"] == "test"
            assert len(data["messages"]) == 1


def test_otel_spans():
    """Spans are emitted when OTel is available."""
    mock_tracer = mock.MagicMock()
    mock_span = mock.MagicMock()
    mock_tracer.start_span.return_value = mock_span

    _otel._get_tracer.cache_clear()

    with mock.patch.object(_otel, "_get_tracer", return_value=mock_tracer):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrajectoryCollector(
                task_id="test-task", domain="code",
                model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
                output_dir=tmpdir,
            )

            # start_episode_span should have been called
            mock_tracer.start_span.assert_called_once()
            call_kwargs = mock_tracer.start_span.call_args
            assert "sentric.episode.test-task" in str(call_kwargs)

            # add_message should emit span event
            collector.add_message(role="user", content="Fix the bug.")
            mock_span.add_event.assert_called()
            event_call = mock_span.add_event.call_args
            assert event_call[0][0] == "message.user"

            collector.add_message(
                role="assistant",
                content="Looking at it.",
                tool_calls=[{"id": "call_1", "name": "bash", "arguments": '{"cmd": "ls"}'}],
            )
            # Should have 2 add_event calls now
            assert mock_span.add_event.call_count == 2

            # save_episode should end the span
            collector.save_episode()
            mock_span.set_attribute.assert_called()
            mock_span.end.assert_called_once()

    _otel._get_tracer.cache_clear()


def test_otel_attributes():
    """Span attributes include expected fields."""
    mock_tracer = mock.MagicMock()
    mock_span = mock.MagicMock()
    mock_tracer.start_span.return_value = mock_span

    _otel._get_tracer.cache_clear()

    with mock.patch.object(_otel, "_get_tracer", return_value=mock_tracer):
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = TrajectoryCollector(
                task_id="django__django-11099", domain="code",
                model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
                output_dir=tmpdir,
            )

            # Verify start_span was called with correct attributes
            start_call = mock_tracer.start_span.call_args
            attrs = start_call[1]["attributes"]
            assert attrs["sentric.task_id"] == "django__django-11099"
            assert attrs["sentric.domain"] == "code"
            assert attrs["sentric.model.name"] == "gpt-4o"
            assert attrs["sentric.model.provider"] == "openai"

            collector.add_message(role="user", content="Fix the regex bug.")
            collector.add_tokens(input_tokens=100, output_tokens=200)
            collector.save_episode()

            # Verify end span attributes
            set_attr_calls = {
                call[0][0]: call[0][1]
                for call in mock_span.set_attribute.call_args_list
            }
            assert set_attr_calls["sentric.message_count"] == 1
            assert set_attr_calls["sentric.total_tokens"] == 300
            assert set_attr_calls["sentric.input_tokens"] == 100
            assert set_attr_calls["sentric.output_tokens"] == 200

    _otel._get_tracer.cache_clear()


def test_otel_message_event_attributes():
    """Message events include role, content preview, and tool info."""
    mock_span = mock.MagicMock()

    # Test user message
    _otel.emit_message_event(mock_span, "user", content="Hello world")
    call = mock_span.add_event.call_args
    assert call[0][0] == "message.user"
    attrs = call[1]["attributes"]
    assert attrs["sentric.message.role"] == "user"
    assert attrs["sentric.message.content_length"] == 11
    assert attrs["sentric.message.content_preview"] == "Hello world"

    # Test assistant with tool calls
    mock_span.reset_mock()
    _otel.emit_message_event(
        mock_span, "assistant",
        content="Let me search.",
        tool_calls=[
            {"id": "call_1", "name": "bash", "arguments": '{"cmd": "ls"}'},
            {"id": "call_2", "name": "read_file", "arguments": '{"path": "foo.py"}'},
        ],
    )
    call = mock_span.add_event.call_args
    attrs = call[1]["attributes"]
    assert attrs["sentric.message.tool_count"] == 2
    assert attrs["sentric.message.tool_names"] == "bash,read_file"

    # Test tool message
    mock_span.reset_mock()
    _otel.emit_message_event(mock_span, "tool", content="result", tool_call_id="call_1")
    call = mock_span.add_event.call_args
    attrs = call[1]["attributes"]
    assert attrs["sentric.message.tool_call_id"] == "call_1"
