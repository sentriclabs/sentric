"""Import trajectories from Weights & Biases (W&B) trace export format.

Expected input format: a W&B trace table row or run summary dict.

Example input:
    {
        "span_id": "abc-123",
        "trace_id": "trace-456",
        "name": "AgentRun",
        "kind": "AGENT",
        "inputs": {"query": "Fix the bug"},
        "outputs": {"response": "Done"},
        "child_spans": [
            {
                "name": "llm_call",
                "kind": "LLM",
                "inputs": {"messages": [{"role": "user", "content": "Fix the bug"}]},
                "outputs": {"choices": [{"message": {"role": "assistant", "content": "Looking..."}}]},
                "attributes": {"model": "gpt-4o", "token_usage": {"prompt_tokens": 50, "completion_tokens": 30}}
            },
            {
                "name": "tool_call",
                "kind": "TOOL",
                "inputs": {"tool_name": "bash", "tool_input": "ls"},
                "outputs": {"tool_output": "file1.py"}
            }
        ],
        "attributes": {"metadata": {"task_id": "task-1"}},
        "start_time_ms": 1704067200000,
        "end_time_ms": 1704067260000
    }
"""

import logging
import uuid
from datetime import datetime, timezone

_log = logging.getLogger("sentric.importers")

_KNOWN_FIELDS = {
    "span_id", "trace_id", "name", "kind", "inputs", "outputs",
    "child_spans", "attributes", "start_time_ms", "end_time_ms",
    "status_code", "status_message", "parent_id",
}


def _extract_messages_from_child_spans(child_spans: list) -> list[dict]:
    """Walk child spans and extract messages in conversation order."""
    messages = []
    for span in child_spans:
        kind = span.get("kind", "").upper()

        if kind == "LLM":
            # Extract input messages
            input_msgs = span.get("inputs", {}).get("messages", [])
            for msg in input_msgs:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content"),
                })

            # Extract output
            choices = span.get("outputs", {}).get("choices", [])
            for choice in choices:
                msg = choice.get("message", {})
                converted = {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content"),
                }
                if "tool_calls" in msg:
                    converted["tool_calls"] = [
                        {
                            "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            "name": tc.get("function", {}).get("name", tc.get("name", "")),
                            "arguments": tc.get("function", {}).get("arguments", tc.get("arguments", "")),
                        }
                        for tc in msg["tool_calls"]
                    ]
                messages.append(converted)

            # Also handle simple text output
            if not choices:
                output_text = span.get("outputs", {}).get("response", "")
                if output_text:
                    messages.append({"role": "assistant", "content": str(output_text)})

        elif kind == "TOOL":
            inputs = span.get("inputs", {})
            outputs = span.get("outputs", {})
            tool_name = inputs.get("tool_name", span.get("name", "unknown"))
            tool_input = inputs.get("tool_input", "")
            tool_output = outputs.get("tool_output", outputs.get("output", ""))
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


def from_wandb(data: dict) -> dict:
    """Convert a W&B trace export to a Sentric episode dict."""
    unknown = set(data.keys()) - _KNOWN_FIELDS
    if unknown:
        _log.warning("Unmapped fields in W&B input: %s", ", ".join(sorted(unknown)))

    attributes = data.get("attributes", {})
    metadata = attributes.get("metadata", {})

    # Extract model info from child spans
    model_name = ""
    provider = ""
    input_tokens = None
    output_tokens = None
    total_tokens = None

    for span in data.get("child_spans", []):
        if span.get("kind", "").upper() == "LLM":
            span_attrs = span.get("attributes", {})
            model_name = span_attrs.get("model", "")
            provider = span_attrs.get("provider", "")

            usage = span_attrs.get("token_usage", {})
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
            if input_tokens and output_tokens:
                total_tokens = input_tokens + output_tokens
            break

    # Extract messages from child spans or inputs
    child_spans = data.get("child_spans", [])
    if child_spans:
        messages = _extract_messages_from_child_spans(child_spans)
    else:
        messages = []
        query = data.get("inputs", {}).get("query", "")
        if query:
            messages.append({"role": "user", "content": str(query)})
        response = data.get("outputs", {}).get("response", "")
        if response:
            messages.append({"role": "assistant", "content": str(response)})

    # Calculate duration
    duration_ms = None
    start_ms = data.get("start_time_ms")
    end_ms = data.get("end_time_ms")
    if start_ms is not None and end_ms is not None:
        duration_ms = end_ms - start_ms

    # Convert start_time_ms to ISO string
    created_at = None
    if start_ms is not None:
        try:
            created_at = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
        except (ValueError, OSError):
            pass

    return {
        "episode_id": data.get("span_id", data.get("trace_id", str(uuid.uuid4()))),
        "task_id": metadata.get("task_id", ""),
        "domain": metadata.get("domain", ""),
        "model": {"name": model_name, "version": "", "provider": provider},
        "messages": messages,
        "reward": None,
        "success": None,
        "verifier": None,
        "verified_at": None,
        "created_at": created_at,
        "duration_ms": duration_ms,
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "metadata": metadata,
    }
