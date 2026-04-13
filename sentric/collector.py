import json
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path


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
        self._total_tokens = 0

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

    def add_tokens(self, count: int):
        """Track token usage across model calls."""
        self._total_tokens += count

    def save_episode(self, output_dir: str | None = None) -> Path:
        """Write the trajectory to disk as JSON. Returns the path to the saved file."""
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        duration_ms = int((time.monotonic() - self._start_time) * 1000)

        episode = {
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
            "total_tokens": self._total_tokens if self._total_tokens > 0 else None,
            "metadata": self.metadata,
        }

        path = out / f"{self.episode_id}.json"
        path.write_text(json.dumps(episode, indent=2))
        return path

    def reset(self, task_id: str | None = None, metadata: dict | None = None):
        """Reset the collector for a new episode, keeping model and domain."""
        self.episode_id = str(uuid.uuid4())
        if task_id is not None:
            self.task_id = task_id
        self.messages = []
        self.metadata = metadata or {}
        self._start_time = time.monotonic()
        self._total_tokens = 0
