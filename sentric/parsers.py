"""Response parsers for OpenAI and Anthropic SDK types.

Each parser takes a raw SDK response and returns a list of messages
in our schema format, plus a token count. Parsers are auto-selected
based on the response type — users never call these directly.
"""

import json


def parse_openai(response) -> tuple[list[dict], int]:
    """Parse an OpenAI ChatCompletion into schema messages + token count."""
    message = response.choices[0].message
    tokens = 0

    if response.usage:
        tokens = response.usage.total_tokens

    parsed = {"role": "assistant", "content": message.content}

    if message.tool_calls:
        parsed["tool_calls"] = [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            for tc in message.tool_calls
        ]

    return [parsed], tokens


def parse_anthropic(response) -> tuple[list[dict], int]:
    """Parse an Anthropic Message into schema messages + token count."""
    tokens = 0
    if response.usage:
        tokens = response.usage.input_tokens + response.usage.output_tokens

    text_parts = []
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "arguments": json.dumps(block.input),
            })

    content = "\n".join(text_parts) if text_parts else None

    parsed = {"role": "assistant", "content": content}
    if tool_calls:
        parsed["tool_calls"] = tool_calls

    return [parsed], tokens


def parse_fallback(response) -> tuple[list[dict], int]:
    """Stringify any unrecognized response type."""
    return [{"role": "assistant", "content": str(response)}], 0


def detect_and_parse(response, normalizer=None) -> tuple[list[dict], int]:
    """Auto-detect the response type and parse it."""
    if normalizer is not None:
        result = normalizer(response)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, 0

    type_name = type(response).__name__
    module = type(response).__module__ or ""

    if type_name == "ChatCompletion" and "openai" in module:
        return parse_openai(response)

    if type_name == "Message" and "anthropic" in module:
        return parse_anthropic(response)

    return parse_fallback(response)
