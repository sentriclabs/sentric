import atexit
import logging
import threading
import uuid
import time
import sys
import platform
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from sentric._json import dumps_bytes
from sentric import otel as _otel

_log = logging.getLogger("sentric")

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    """Return the module-level singleton ThreadPoolExecutor."""
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=1)
                atexit.register(_executor.shutdown, wait=True)
    return _executor


_SENTINEL = object()


class _OtelSnapshot:
    """Lightweight snapshot of collector state for async OTel span ending."""

    __slots__ = ("messages", "_input_tokens", "_output_tokens")

    def __init__(self, messages, input_tokens, output_tokens):
        self.messages = messages
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens


class TrajectoryCollector:
    """Collects agent trajectories in the conversation-based schema.

    Usage:
        from sentric import TrajectoryCollector

        collector = TrajectoryCollector(
            task_id="django__django-11099",
            domain="code",
            model={"name": "Qwen/Qwen2.5-Coder-7B", "version": "base", "provider": "local"},
        )

        collector.add_message(role="system", content="You are a software engineer...")
        collector.add_message(role="user", content="Fix the bug...")
        collector.add_message(
            role="assistant",
            content="Let me look at the file.",
            tool_calls=[{"id": "call_1", "name": "bash", "arguments": '{"command": "ls"}'}],
        )
        collector.add_message(role="tool", content="file1.py\nfile2.py", tool_call_id="call_1")

        collector.save_episode()
    """

    __slots__ = (
        "episode_id",
        "task_id",
        "domain",
        "model",
        "output_dir",
        "metadata",
        "messages",
        "_start_time",
        "_input_tokens",
        "_output_tokens",
        "_executor",
        "_otel_span",
    )

    def __init__(
        self,
        task_id: str,
        domain: str,
        model: dict,
        output_dir: str = "data/trajectories",
        metadata: dict | None = None,
    ):
        self.episode_id = str(uuid.uuid4())
        self.task_id = task_id
        self.domain = domain
        self.model = model
        self.output_dir = Path(output_dir)
        self.metadata = metadata or {}
        self.messages: list[dict] = []
        self._start_time = time.monotonic()
        self._input_tokens = 0
        self._output_tokens = 0
        self._executor = None
        self._otel_span = _otel.start_episode_span(self)

    def add_message(
        self,
        role: str,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ):
        """Append a message to the trajectory."""
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Invalid role: {role}. Must be one of: system, user, assistant, tool")

        if role == "tool" and tool_call_id is None:
            raise ValueError("tool messages must include tool_call_id")

        if role == "assistant" and tool_calls is not None:
            for tc in tool_calls:
                missing = [k for k in ("id", "name", "arguments") if k not in tc]
                if missing:
                    raise ValueError(f"tool_call missing required fields: {missing}")

        message = {"role": role, "content": content}

        if role == "assistant" and tool_calls:
            message["tool_calls"] = tool_calls

        if role == "tool":
            message["tool_call_id"] = tool_call_id

        self.messages.append(message)

        _otel.emit_message_event(
            self._otel_span, role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )

    def add_step(
        self,
        content: str,
        tool_name: str,
        tool_args: str,
        tool_result: str,
        tool_call_id: str | None = None,
    ):
        """Log a full tool-call round-trip (assistant + tool) in one call."""
        call_id = tool_call_id or f"call_{uuid.uuid4().hex[:8]}"
        self.add_message(
            role="assistant",
            content=content,
            tool_calls=[{"id": call_id, "name": tool_name, "arguments": tool_args}],
        )
        self.add_message(role="tool", content=tool_result, tool_call_id=call_id)

    def add_tokens(self, input_tokens: int = 0, output_tokens: int = 0):
        """Track token usage across model calls.

        Args:
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.
        """
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

    @property
    def _total_tokens(self) -> int:
        """Total tokens (input + output) for backward compatibility."""
        return self._input_tokens + self._output_tokens

    def to_dict(self) -> dict:
        """Return the episode as a dict without writing to disk."""
        duration_ms = int((time.monotonic() - self._start_time) * 1000)
        total_tokens = self._input_tokens + self._output_tokens
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "domain": self.domain,
            "model": self.model,
            "messages": self.messages,
            "reward": None,
            "success": None,
            "verifier": None,
            "verified_at": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "total_tokens": total_tokens if total_tokens > 0 else None,
            "input_tokens": self._input_tokens if self._input_tokens > 0 else None,
            "output_tokens": self._output_tokens if self._output_tokens > 0 else None,
            "metadata": self.metadata,
        }

    def save_episode(self, output_dir: str | None = None) -> Path:
        """Write the trajectory to disk as JSON. Returns the path to the saved file."""
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        episode = self.to_dict()

        path = out / f"{self.episode_id}.json"
        path.write_bytes(dumps_bytes(episode))

        _otel.end_episode_span(self._otel_span, self)

        return path

    def save_episode_async(self, output_dir: str | None = None) -> Future:
        """Write the trajectory to disk in a background thread.

        Returns a Future that resolves to the saved Path.
        Exceptions are logged if the Future is not checked.
        """
        episode = self.to_dict()
        episode_id = self.episode_id
        out = Path(output_dir) if output_dir else self.output_dir
        span = self._otel_span

        # Snapshot OTel-relevant state so the closure doesn't hold self alive
        # and is immune to reset() being called before _write runs.
        otel_snapshot = _OtelSnapshot(
            messages=self.messages,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

        def _write() -> Path:
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{episode_id}.json"
            path.write_bytes(dumps_bytes(episode))
            _otel.end_episode_span(span, otel_snapshot)
            return path

        future = _get_executor().submit(_write)

        def _on_done(f: Future):
            exc = f.exception()
            if exc is not None:
                _log.warning("save_episode_async failed: %s", exc)

        future.add_done_callback(_on_done)
        return future

    def capture_env(self):
        """Capture environment info into metadata under the _env key."""
        env = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": {},
        }

        for pkg_name in ("sentric", "openai", "anthropic"):
            try:
                mod = __import__(pkg_name)
                env["packages"][pkg_name] = getattr(mod, "__version__", "unknown")
            except ImportError:
                pass

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            env["git_hash"] = result.stdout.strip() if result.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            env["git_hash"] = None

        self.metadata["_env"] = env

    def reset(self, task_id: str | None = None, metadata=_SENTINEL):
        """Reset the collector for a new episode, keeping model and domain."""
        self.episode_id = str(uuid.uuid4())
        if task_id is not None:
            self.task_id = task_id
        self.messages = []
        if metadata is not _SENTINEL:
            self.metadata = metadata or {}
        self._start_time = time.monotonic()
        self._input_tokens = 0
        self._output_tokens = 0
        self._otel_span = _otel.start_episode_span(self)
