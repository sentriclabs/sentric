"""Mock OpenAI and Anthropic response objects for testing.

These simulate the SDK types closely enough for auto-detection to work.
The detector checks type().__name__ and type().__module__, so we set
__module__ on each class to match the real SDK.
"""

import sys
import types


def make_openai_response(content="Hello!", tool_calls=None, total_tokens=150):
    """Build a fake OpenAI ChatCompletion.

    Args:
        content: Text content of the assistant message.
        tool_calls: List of dicts with "id", "name", "arguments" keys.
        total_tokens: Total token count for the response.
    """
    # Register a fake module so __module__ contains "openai"
    mod_name = "openai.types.chat.chat_completion"
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

    class Function:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class ToolCall:
        def __init__(self, id, function):
            self.id = id
            self.function = function

    class Message:
        def __init__(self, content, tool_calls):
            self.role = "assistant"
            self.content = content
            self.tool_calls = tool_calls

    class Choice:
        def __init__(self, message):
            self.message = message
            self.finish_reason = "tool_calls" if message.tool_calls else "stop"
            self.index = 0

    class Usage:
        def __init__(self, total_tokens):
            self.prompt_tokens = total_tokens // 3
            self.completion_tokens = total_tokens - total_tokens // 3
            self.total_tokens = total_tokens

    tc_objects = None
    if tool_calls:
        tc_objects = [
            ToolCall(id=tc["id"], function=Function(tc["name"], tc["arguments"]))
            for tc in tool_calls
        ]

    class ChatCompletion:
        __module__ = mod_name

        def __init__(self):
            self.choices = [Choice(Message(content, tc_objects))]
            self.usage = Usage(total_tokens)
            self.model = "gpt-4o"

    return ChatCompletion()


def make_anthropic_response(text="Hello!", tool_uses=None, input_tokens=50, output_tokens=100):
    """Build a fake Anthropic Message.

    Args:
        text: Text content (or None for tool-only responses).
        tool_uses: List of dicts with "id", "name", "input" keys.
        input_tokens: Input token count.
        output_tokens: Output token count.
    """
    mod_name = "anthropic.types.message"
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

    class TextBlock:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class ToolUseBlock:
        def __init__(self, id, name, input):
            self.type = "tool_use"
            self.id = id
            self.name = name
            self.input = input

    class Usage:
        def __init__(self, input_tokens, output_tokens):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    content_blocks = []
    if text:
        content_blocks.append(TextBlock(text))
    if tool_uses:
        for tu in tool_uses:
            content_blocks.append(ToolUseBlock(tu["id"], tu["name"], tu["input"]))

    class Message:
        __module__ = mod_name

        def __init__(self):
            self.role = "assistant"
            self.content = content_blocks
            self.usage = Usage(input_tokens, output_tokens)
            self.stop_reason = "tool_use" if tool_uses else "end_turn"

    return Message()


# --- Streaming mock helpers ---


def make_openai_stream_chunks(content="Hello!", input_tokens=50, output_tokens=100):
    """Build a list of fake OpenAI streaming chunks.

    Returns a list of chunk objects that simulate ChatCompletionChunk.
    """
    mod_name = "openai.types.chat.chat_completion_chunk"
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

    class Delta:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.role = "assistant" if content else None
            self.tool_calls = tool_calls

    class ChoiceDelta:
        def __init__(self, delta):
            self.delta = delta
            self.index = 0
            self.finish_reason = None

    class Usage:
        def __init__(self, prompt_tokens, completion_tokens):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    chunks = []

    # Split content into word-level chunks
    if content:
        words = content.split(" ")
        for i, word in enumerate(words):
            text = word if i == len(words) - 1 else word + " "

            class Chunk:
                __module__ = mod_name

                def __init__(self, text):
                    self.choices = [ChoiceDelta(Delta(content=text))]
                    self.usage = None

            chunks.append(Chunk(text))

    # Final chunk with usage
    class FinalChunk:
        __module__ = mod_name

        def __init__(self):
            self.choices = []
            self.usage = Usage(input_tokens, output_tokens)

    chunks.append(FinalChunk())

    return chunks


def make_openai_stream(content="Hello!", input_tokens=50, output_tokens=100):
    """Build a fake OpenAI streaming iterator."""
    mod_name = "openai.lib.streaming"
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

    chunks = make_openai_stream_chunks(content, input_tokens, output_tokens)

    class Stream:
        __module__ = mod_name

        def __init__(self, chunks):
            self._chunks = iter(chunks)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._chunks)

    return Stream(chunks)


def make_anthropic_stream_events(text="Hello!", input_tokens=50, output_tokens=100):
    """Build a list of fake Anthropic streaming events."""

    class Event:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MessageUsage:
        def __init__(self, input_tokens):
            self.input_tokens = input_tokens

    class DeltaUsage:
        def __init__(self, output_tokens):
            self.output_tokens = output_tokens

    class MessageObj:
        def __init__(self, input_tokens):
            self.usage = MessageUsage(input_tokens)

    events = []

    # message_start
    events.append(Event(type="message_start", message=MessageObj(input_tokens)))

    # content_block_start for text
    events.append(Event(
        type="content_block_start",
        content_block=Event(type="text", text=""),
        index=0,
    ))

    # Text deltas
    if text:
        words = text.split(" ")
        for i, word in enumerate(words):
            t = word if i == len(words) - 1 else word + " "
            events.append(Event(
                type="content_block_delta",
                delta=Event(type="text_delta", text=t),
                index=0,
            ))

    # content_block_stop
    events.append(Event(type="content_block_stop", index=0))

    # message_delta with usage
    events.append(Event(
        type="message_delta",
        delta=Event(type="message_delta", stop_reason="end_turn"),
        usage=DeltaUsage(output_tokens),
    ))

    return events


def make_anthropic_stream(text="Hello!", input_tokens=50, output_tokens=100):
    """Build a fake Anthropic streaming iterator."""
    mod_name = "anthropic.lib.streaming"
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

    events = make_anthropic_stream_events(text, input_tokens, output_tokens)

    class MessageStream:
        __module__ = mod_name

        def __init__(self, events):
            self._events = iter(events)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._events)

    return MessageStream(events)
