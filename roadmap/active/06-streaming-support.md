# Feature 06: Streaming Response Support

**Priority**: P1
**Effort**: L
**Status**: planned

## Summary
Add ability to trace streaming LLM responses (OpenAI `stream=True`, Anthropic streaming). Currently the decorator only handles complete response objects. Production code frequently uses streaming for lower time-to-first-token.

## Requirements
- `@trace` and `@atrace` detect streaming responses (generators/async generators)
- Accumulate streamed chunks into a complete message before logging
- Track token usage from streaming responses (OpenAI includes usage in final chunk with `stream_options={"include_usage": True}`)
- Return the original stream to the caller transparently (don't buffer everything before returning)
- Add `parse_openai_stream()` and `parse_anthropic_stream()` to parsers

## Design Notes
**Performance**: Must not add latency to streaming. Wrap the stream with a pass-through that tees content to the collector as chunks arrive. Log the complete message only after the stream is exhausted.

**Architecture**: Create wrapper classes `TracedStream` and `TracedAsyncStream` that proxy the original iterator, accumulate content, and call `collector.add_message()` on completion.

## Files to Modify
- `sentric/parsers.py` — streaming parsers
- `sentric/trace.py` — streaming detection and wrapper
- `sentric/streams.py` — new file for TracedStream/TracedAsyncStream

## Tests Needed
- `test_streaming_openai()` — mock streaming response
- `test_streaming_anthropic()` — mock Anthropic streaming
- `test_streaming_async()` — async streaming
- `test_streaming_passthrough()` — verify original stream behavior preserved
- `test_streaming_token_count()` — verify token tracking from stream

## Notes
- This is the most complex feature. Need to handle edge cases: stream interrupted mid-way, stream consumed partially, etc.
- Eng review: TracedStream must log partial content on close()/__del__ when stream is abandoned mid-way
- Eng review: add test for partial consumption (iterate 3 of 10 chunks, verify partial content logged)
