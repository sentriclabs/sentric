# sentric

Open-source trajectory logging for LLM agents. Two lines of code to get structured, replayable records of every agent run — ready for evaluation and fine-tuning.

No backend. No API key. No vendor lock-in. Just local JSON files.

## Quickstart

```bash
pip install sentric
```

### OpenAI

```python
from sentric import TrajectoryCollector, trace

collector = TrajectoryCollector(
    task_id="fix-auth-bug",
    domain="code",
    model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
)

@trace(collector)
def call_llm(messages):
    return openai.chat.completions.create(model="gpt-4o", messages=messages)

# Use call_llm as normal — every call is automatically logged
response = call_llm(messages=[
    {"role": "system", "content": "You are a software engineer."},
    {"role": "user", "content": "Fix the authentication bug in auth.py"},
])

# When the agent finishes, save the trajectory
collector.save_episode()
# -> data/trajectories/f47ac10b-58cc-4372-a567-0e02b2c3d479.json
```

That's it. The `@trace` decorator auto-detects the OpenAI response, extracts the assistant message, tool calls, and token counts, and logs everything in a structured format.

### Anthropic

```python
@trace(collector)
def call_llm(messages):
    return anthropic.messages.create(model="claude-sonnet-4-20250514", messages=messages, max_tokens=4096)
```

Same decorator. Auto-detected. Handles text blocks and tool use blocks.

### Any other LLM

```python
def my_normalizer(response):
    return [{"role": "assistant", "content": response.text}], response.token_count

@trace(collector, normalizer=my_normalizer)
def call_llm(messages):
    return my_custom_client.generate(messages)
```

Write a 5-line normalizer function that tells us how to extract fields from your response type. Or skip it entirely — unknown types fall back to stringifying the response.

## What gets logged

Each trajectory is a JSON file containing the full conversation:

```json
{
  "episode_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "task_id": "fix-auth-bug",
  "domain": "code",
  "model": {
    "name": "gpt-4o",
    "version": "2024-08-06",
    "provider": "openai"
  },
  "messages": [
    {"role": "system", "content": "You are a software engineer."},
    {"role": "user", "content": "Fix the authentication bug in auth.py"},
    {
      "role": "assistant",
      "content": "Let me look at the auth module.",
      "tool_calls": [
        {"id": "call_1", "name": "bash", "arguments": "{\"command\": \"cat auth.py\"}"}
      ]
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "class AuthHandler:\n    ..."}
  ],
  "reward": null,
  "success": null,
  "created_at": "2026-04-13T10:30:00Z",
  "duration_ms": 45200,
  "total_tokens": 8430,
  "metadata": {}
}
```

Reward fields start as `null` — they get filled in later by a verifier that scores whether the agent succeeded.

## Manual logging

If you need more control, use the collector directly:

```python
from sentric import TrajectoryCollector

collector = TrajectoryCollector(
    task_id="extract-invoice",
    domain="extraction",
    model={"name": "gpt-4o-mini", "version": "2024-07-18", "provider": "openai"},
)

collector.add_message(role="system", content="Extract fields from this invoice.")
collector.add_message(role="user", content="Invoice #1234...")
collector.add_message(role="assistant", content='{"vendor": "Acme", "amount": 150.00}')
collector.add_tokens(500)
collector.save_episode()
```

## Multiple episodes

Use `reset()` to reuse a collector across tasks:

```python
for task in tasks:
    collector.reset(task_id=task.id)
    run_agent(task, collector)
    collector.save_episode()
```

## License

MIT
