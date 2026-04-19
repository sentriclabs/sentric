"""Import trajectories from raw OpenAI message logs.

Expected input format: a dict with a "messages" list following the OpenAI
chat completion message format, plus optional metadata fields.

Example input:
    {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!", "tool_calls": [...]},
            {"role": "tool", "content": "result", "tool_call_id": "call_1"}
        ],
        "model": "gpt-4o",
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    }
"""

import logging
import uuid

_log = logging.getLogger("sentric.importers")

_KNOWN_FIELDS = {
    "messages", "model", "usage", "id", "created", "task_id",
    "metadata", "domain",
}


def from_openai_messages(data: dict) -> dict:
    """Convert an OpenAI message log to a Sentric episode dict."""
    unknown = set(data.keys()) - _KNOWN_FIELDS
    if unknown:
        _log.warning("Unmapped fields in OpenAI messages input: %s", ", ".join(sorted(unknown)))

    messages = []
    for msg in data.get("messages", []):
        converted = {"role": msg["role"], "content": msg.get("content")}

        if msg["role"] == "assistant" and "tool_calls" in msg:
            converted["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": tc.get("function", {}).get("name", tc.get("name", "")),
                    "arguments": tc.get("function", {}).get("arguments", tc.get("arguments", "")),
                }
                for tc in msg["tool_calls"]
            ]

        if msg["role"] == "tool" and "tool_call_id" in msg:
            converted["tool_call_id"] = msg["tool_call_id"]

        messages.append(converted)

    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    if total_tokens is None and input_tokens and output_tokens:
        total_tokens = input_tokens + output_tokens

    model_name = data.get("model", "")

    return {
        "episode_id": data.get("id", str(uuid.uuid4())),
        "task_id": data.get("task_id", ""),
        "domain": data.get("domain", ""),
        "model": {"name": model_name, "version": "", "provider": "openai"},
        "messages": messages,
        "reward": None,
        "success": None,
        "verifier": None,
        "verified_at": None,
        "created_at": None,
        "duration_ms": None,
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost_usd": None,
        "metadata": data.get("metadata", {}),
    }
