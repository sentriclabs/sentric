"""Import trajectories from LangSmith run export format.

Expected input format: a LangSmith run dict as returned by the LangSmith API
or exported via the LangSmith UI.

Example input:
    {
        "id": "run-uuid",
        "name": "AgentExecutor",
        "run_type": "chain",
        "inputs": {"input": "Fix the bug in models.py"},
        "outputs": {"output": "I've fixed the bug..."},
        "child_runs": [
            {
                "name": "ChatOpenAI",
                "run_type": "llm",
                "inputs": {"messages": [...]},
                "outputs": {"generations": [[{"text": "...", "message": {...}}]]},
                "extra": {"invocation_params": {"model_name": "gpt-4o"}},
                "token_usage": {"prompt_tokens": 100, "completion_tokens": 50}
            },
            {
                "name": "bash",
                "run_type": "tool",
                "inputs": {"input": "ls"},
                "outputs": {"output": "file1.py\nfile2.py"}
            }
        ],
        "extra": {"metadata": {"task_id": "django__django-11099"}},
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-01T00:01:00Z"
    }
"""

import logging
import uuid
from datetime import datetime

_log = logging.getLogger("sentric.importers")

_KNOWN_FIELDS = {
    "id", "name", "run_type", "inputs", "outputs", "child_runs",
    "extra", "start_time", "end_time", "error", "status",
    "parent_run_id", "session_id", "reference_example_id",
    "serialized", "events", "tags", "feedback_stats",
    "dotted_order", "trace_id", "total_tokens", "prompt_tokens",
    "completion_tokens", "token_usage",
}


def _extract_messages_from_child_runs(child_runs: list) -> list[dict]:
    """Walk child runs and extract messages in conversation order."""
    messages = []
    for run in child_runs:
        run_type = run.get("run_type", "")

        if run_type == "llm":
            # Extract input messages from LLM calls
            input_msgs = run.get("inputs", {}).get("messages", [])
            for msg_group in input_msgs:
                if isinstance(msg_group, list):
                    for m in msg_group:
                        messages.append(_convert_langsmith_message(m))
                elif isinstance(msg_group, dict):
                    messages.append(_convert_langsmith_message(msg_group))

            # Extract output from LLM
            generations = run.get("outputs", {}).get("generations", [])
            for gen_group in generations:
                if isinstance(gen_group, list):
                    for gen in gen_group:
                        msg_data = gen.get("message", {})
                        if msg_data:
                            messages.append(_convert_langsmith_message(msg_data))
                        elif gen.get("text"):
                            messages.append({
                                "role": "assistant",
                                "content": gen["text"],
                            })

        elif run_type == "tool":
            tool_name = run.get("name", "unknown")
            tool_input = run.get("inputs", {}).get("input", "")
            tool_output = run.get("outputs", {}).get("output", "")
            call_id = f"call_{uuid.uuid4().hex[:8]}"

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "name": tool_name,
                    "arguments": str(tool_input) if not isinstance(tool_input, str) else tool_input,
                }],
            })
            messages.append({
                "role": "tool",
                "content": str(tool_output) if not isinstance(tool_output, str) else tool_output,
                "tool_call_id": call_id,
            })

    return messages


def _convert_langsmith_message(msg: dict) -> dict:
    """Convert a single LangSmith message dict to Sentric format."""
    # LangSmith messages can have type or role
    role = msg.get("role", "")
    if not role:
        msg_type = msg.get("type", "").lower()
        role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
        role = role_map.get(msg_type, msg_type)

    content = msg.get("content", "")
    converted = {"role": role, "content": content}

    # Handle tool calls from LangSmith's additional_kwargs
    additional = msg.get("additional_kwargs", {})
    tool_calls = additional.get("tool_calls", [])
    if tool_calls and role == "assistant":
        converted["tool_calls"] = [
            {
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": tc.get("function", {}).get("name", tc.get("name", "")),
                "arguments": tc.get("function", {}).get("arguments", tc.get("arguments", "")),
            }
            for tc in tool_calls
        ]

    if role == "tool":
        converted["tool_call_id"] = msg.get("tool_call_id", additional.get("tool_call_id", ""))

    return converted


def from_langsmith(data: dict) -> dict:
    """Convert a LangSmith run export to a Sentric episode dict."""
    unknown = set(data.keys()) - _KNOWN_FIELDS
    if unknown:
        _log.warning("Unmapped fields in LangSmith input: %s", ", ".join(sorted(unknown)))

    extra = data.get("extra", {})
    metadata = extra.get("metadata", {})

    # Try to extract model info from child runs
    model_name = ""
    provider = ""
    for child in data.get("child_runs", []):
        if child.get("run_type") == "llm":
            params = child.get("extra", {}).get("invocation_params", {})
            model_name = params.get("model_name", params.get("model", ""))
            provider = params.get("_type", "")
            if not provider and "openai" in str(params).lower():
                provider = "openai"
            elif not provider and "anthropic" in str(params).lower():
                provider = "anthropic"
            break

    # Extract token usage
    input_tokens = None
    output_tokens = None
    total_tokens = data.get("total_tokens")

    token_usage = data.get("token_usage", {})
    if token_usage:
        input_tokens = token_usage.get("prompt_tokens")
        output_tokens = token_usage.get("completion_tokens")
    else:
        input_tokens = data.get("prompt_tokens")
        output_tokens = data.get("completion_tokens")

    if total_tokens is None and input_tokens and output_tokens:
        total_tokens = input_tokens + output_tokens

    # Extract messages from child runs or inputs
    child_runs = data.get("child_runs", [])
    if child_runs:
        messages = _extract_messages_from_child_runs(child_runs)
    else:
        # Fallback: use top-level inputs/outputs
        messages = []
        input_text = data.get("inputs", {}).get("input", "")
        if input_text:
            messages.append({"role": "user", "content": str(input_text)})
        output_text = data.get("outputs", {}).get("output", "")
        if output_text:
            messages.append({"role": "assistant", "content": str(output_text)})

    # Calculate duration
    duration_ms = None
    start = data.get("start_time")
    end = data.get("end_time")
    if start and end:
        try:
            t0 = datetime.fromisoformat(start.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(end.replace("Z", "+00:00"))
            duration_ms = int((t1 - t0).total_seconds() * 1000)
        except (ValueError, TypeError):
            pass

    return {
        "episode_id": data.get("id", str(uuid.uuid4())),
        "task_id": metadata.get("task_id", ""),
        "domain": metadata.get("domain", ""),
        "model": {"name": model_name, "version": "", "provider": provider},
        "messages": messages,
        "reward": None,
        "success": None,
        "verifier": None,
        "verified_at": None,
        "created_at": data.get("start_time"),
        "duration_ms": duration_ms,
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "metadata": metadata,
    }
