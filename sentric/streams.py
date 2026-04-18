"""Wrapper classes for tracing streaming LLM responses.

TracedStream and TracedAsyncStream proxy the original iterator,
accumulate content as chunks arrive, and log the complete message
to the collector when the stream is exhausted or closed.
"""

import json


def _parse_openai_chunk(chunk):
    """Extract content and tool call deltas from an OpenAI streaming chunk."""
    if not chunk.choices:
        # Usage-only chunk (final chunk with stream_options.include_usage)
        usage = getattr(chunk, "usage", None)
        if usage:
            return None, None, getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)
        return None, None, 0, 0

    delta = chunk.choices[0].delta
    content = getattr(delta, "content", None)
    tool_calls = getattr(delta, "tool_calls", None)
    return content, tool_calls, 0, 0


def _parse_anthropic_event(event):
    """Extract content from an Anthropic streaming event."""
    event_type = getattr(event, "type", None)

    if event_type == "content_block_delta":
        delta = event.delta
        if getattr(delta, "type", None) == "text_delta":
            return getattr(delta, "text", None), None, 0, 0
        if getattr(delta, "type", None) == "input_json_delta":
            return None, getattr(delta, "partial_json", ""), 0, 0
    elif event_type == "content_block_start":
        block = getattr(event, "content_block", None)
        if block and getattr(block, "type", None) == "tool_use":
            return None, {"start": True, "id": block.id, "name": block.name}, 0, 0
    elif event_type == "message_delta":
        usage = getattr(event, "usage", None)
        if usage:
            return None, None, 0, getattr(usage, "output_tokens", 0)
    elif event_type == "message_start":
        message = getattr(event, "message", None)
        if message:
            usage = getattr(message, "usage", None)
            if usage:
                return None, None, getattr(usage, "input_tokens", 0), 0

    return None, None, 0, 0


class TracedStream:
    """Wraps a sync streaming response to accumulate and log content."""

    def __init__(self, stream, collector, stream_type="openai"):
        self._stream = stream
        self._collector = collector
        self._stream_type = stream_type
        self._content_parts: list[str] = []
        self._tool_calls: list[dict] = []
        self._current_tool_args: list[str] = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._logged = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._stream)
            self._process_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise

    def _process_chunk(self, chunk):
        if self._stream_type == "openai":
            content, tool_calls, inp, out = _parse_openai_chunk(chunk)
            if content:
                self._content_parts.append(content)
            if tool_calls:
                for tc_delta in tool_calls:
                    idx = getattr(tc_delta, "index", 0)
                    fn = getattr(tc_delta, "function", None)
                    if getattr(tc_delta, "id", None):
                        # New tool call starting
                        self._tool_calls.append({
                            "id": tc_delta.id,
                            "name": fn.name if fn and fn.name else "",
                            "arguments": fn.arguments if fn and fn.arguments else "",
                        })
                    elif fn and self._tool_calls:
                        # Continuation of existing tool call
                        if fn.name:
                            self._tool_calls[-1]["name"] += fn.name
                        if fn.arguments:
                            self._tool_calls[-1]["arguments"] += fn.arguments
            self._input_tokens += inp
            self._output_tokens += out
        elif self._stream_type == "anthropic":
            content, tool_data, inp, out = _parse_anthropic_event(chunk)
            if content:
                self._content_parts.append(content)
            if isinstance(tool_data, dict) and tool_data.get("start"):
                # Finalize previous tool args if any
                if self._current_tool_args and self._tool_calls:
                    self._tool_calls[-1]["arguments"] = "".join(self._current_tool_args)
                    self._current_tool_args = []
                self._tool_calls.append({
                    "id": tool_data["id"],
                    "name": tool_data["name"],
                    "arguments": "",
                })
            elif isinstance(tool_data, str):
                self._current_tool_args.append(tool_data)
            self._input_tokens += inp
            self._output_tokens += out

    def _finalize(self):
        if self._logged:
            return
        self._logged = True

        # Finalize any pending tool args
        if self._current_tool_args and self._tool_calls:
            self._tool_calls[-1]["arguments"] = "".join(self._current_tool_args)

        content = "".join(self._content_parts) if self._content_parts else None
        tool_calls = self._tool_calls if self._tool_calls else None

        # JSON-encode tool call arguments for Anthropic
        if tool_calls and self._stream_type == "anthropic":
            for tc in tool_calls:
                if tc["arguments"] and not tc["arguments"].startswith("{"):
                    pass  # already a string

        self._collector.add_message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )

        if self._input_tokens > 0 or self._output_tokens > 0:
            self._collector.add_tokens(
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
            )

    def close(self):
        """Close the stream and log any accumulated content."""
        self._finalize()
        if hasattr(self._stream, "close"):
            self._stream.close()

    def __del__(self):
        self._finalize()


class TracedAsyncStream:
    """Wraps an async streaming response to accumulate and log content."""

    def __init__(self, stream, collector, stream_type="openai"):
        self._stream = stream
        self._collector = collector
        self._stream_type = stream_type
        self._content_parts: list[str] = []
        self._tool_calls: list[dict] = []
        self._current_tool_args: list[str] = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._logged = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._stream.__anext__()
            self._process_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise

    def _process_chunk(self, chunk):
        # Reuse the same logic as TracedStream
        if self._stream_type == "openai":
            content, tool_calls, inp, out = _parse_openai_chunk(chunk)
            if content:
                self._content_parts.append(content)
            if tool_calls:
                for tc_delta in tool_calls:
                    fn = getattr(tc_delta, "function", None)
                    if getattr(tc_delta, "id", None):
                        self._tool_calls.append({
                            "id": tc_delta.id,
                            "name": fn.name if fn and fn.name else "",
                            "arguments": fn.arguments if fn and fn.arguments else "",
                        })
                    elif fn and self._tool_calls:
                        if fn.name:
                            self._tool_calls[-1]["name"] += fn.name
                        if fn.arguments:
                            self._tool_calls[-1]["arguments"] += fn.arguments
            self._input_tokens += inp
            self._output_tokens += out
        elif self._stream_type == "anthropic":
            content, tool_data, inp, out = _parse_anthropic_event(chunk)
            if content:
                self._content_parts.append(content)
            if isinstance(tool_data, dict) and tool_data.get("start"):
                if self._current_tool_args and self._tool_calls:
                    self._tool_calls[-1]["arguments"] = "".join(self._current_tool_args)
                    self._current_tool_args = []
                self._tool_calls.append({
                    "id": tool_data["id"],
                    "name": tool_data["name"],
                    "arguments": "",
                })
            elif isinstance(tool_data, str):
                self._current_tool_args.append(tool_data)
            self._input_tokens += inp
            self._output_tokens += out

    def _finalize(self):
        if self._logged:
            return
        self._logged = True

        if self._current_tool_args and self._tool_calls:
            self._tool_calls[-1]["arguments"] = "".join(self._current_tool_args)

        content = "".join(self._content_parts) if self._content_parts else None
        tool_calls = self._tool_calls if self._tool_calls else None

        self._collector.add_message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        )

        if self._input_tokens > 0 or self._output_tokens > 0:
            self._collector.add_tokens(
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
            )

    async def close(self):
        """Close the stream and log any accumulated content."""
        self._finalize()
        if hasattr(self._stream, "close"):
            result = self._stream.close()
            if hasattr(result, "__await__"):
                await result

    def __del__(self):
        self._finalize()
