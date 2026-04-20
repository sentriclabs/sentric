# sentric

Open-source trajectory logging for LLM agents. Two lines of code to get structured, replayable records of every agent run -- ready for evaluation and fine-tuning.

No backend. No API key. No vendor lock-in. Just local JSON files.

## Quick Start

```bash
pip install sentric
```

```python
from openai import OpenAI
from sentric import TrajectoryCollector, trace

client = OpenAI()

collector = TrajectoryCollector(
    task_id="fix-auth-bug",
    domain="code",
    model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
)

@trace(collector)
def call_llm(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)

response = call_llm(messages=[
    {"role": "system", "content": "You are a software engineer."},
    {"role": "user", "content": "Fix the authentication bug in auth.py"},
])

collector.save_episode()
# -> data/trajectories/f47ac10b-58cc-4372-a567-0e02b2c3d479.json
```

The `@trace` decorator auto-detects the OpenAI response, extracts the assistant message, tool calls, and token counts, and logs everything in a structured format.

## Installation

```bash
# Core (zero dependencies)
pip install sentric

# With provider SDKs (for auto-detection in @trace)
pip install sentric[openai]
pip install sentric[anthropic]

# Performance (orjson for faster JSON serialization)
pip install sentric[fast]

# OpenTelemetry integration
pip install sentric[otel]

# Everything
pip install sentric[all]

# Development
pip install sentric[dev]
```

## Core Concepts

### TrajectoryCollector

The collector records messages exchanged between user, assistant, and tools during an agent run. Each run is an **episode**.

```python
from sentric import TrajectoryCollector

collector = TrajectoryCollector(
    task_id="django__django-11099",
    domain="code",
    model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
    output_dir="data/trajectories",  # default
    metadata={"repo": "django/django", "branch": "main"},
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_id` | `str` | Unique identifier for the task |
| `domain` | `str` | Task domain (e.g. `"code"`, `"extraction"`, `"qa"`) |
| `model` | `dict` | Model info with `name`, `version`, `provider` keys |
| `output_dir` | `str` | Directory for saved trajectories (default: `"data/trajectories"`) |
| `metadata` | `dict \| None` | Optional metadata attached to the episode |

### Adding Messages

```python
# System prompt
collector.add_message(role="system", content="You are a software engineer.")

# User message
collector.add_message(role="user", content="Fix the bug in models.py")

# Assistant with tool calls
collector.add_message(
    role="assistant",
    content="Let me look at the file.",
    tool_calls=[{"id": "call_1", "name": "bash", "arguments": '{"command": "cat models.py"}'}],
)

# Tool result
collector.add_message(role="tool", content="class Model:\n    pass", tool_call_id="call_1")
```

Roles must be one of: `system`, `user`, `assistant`, `tool`. Tool messages require `tool_call_id`.

### Shorthand: add_step

Log a full tool-call round-trip in one call:

```python
collector.add_step(
    content="Let me check the file.",
    tool_name="bash",
    tool_args='{"command": "cat models.py"}',
    tool_result="class Model:\n    pass",
)
```

### Saving Episodes

```python
path = collector.save_episode()
# -> Path('data/trajectories/<episode_id>.json')
```

### Async Save

For non-blocking writes in production agents:

```python
future = collector.save_episode_async()
# Returns concurrent.futures.Future[Path]

# Optionally wait for completion
path = future.result()
```

Uses a module-level singleton `ThreadPoolExecutor(max_workers=1)` with `atexit` cleanup. Exceptions are logged via `logging.getLogger('sentric')` if the `Future` is not checked.

### Multiple Episodes

Reuse a collector across tasks with `reset()`:

```python
for task in tasks:
    collector.reset(task_id=task.id)
    run_agent(task, collector)
    collector.save_episode()
```

`reset()` preserves `model`, `domain`, and `output_dir`. Pass `metadata={}` to clear metadata, or omit it to keep existing metadata.

## The @trace Decorator

Auto-detect provider responses and log messages + tokens without manual extraction.

### OpenAI

```python
from openai import OpenAI
from sentric import TrajectoryCollector, trace

client = OpenAI()
collector = TrajectoryCollector(
    task_id="my-task", domain="code",
    model={"name": "gpt-4o", "version": "2024-08-06", "provider": "openai"},
)

@trace(collector)
def call_llm(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)
```

### Anthropic

```python
import anthropic
from sentric import TrajectoryCollector, trace

client = anthropic.Anthropic()
collector = TrajectoryCollector(
    task_id="my-task", domain="code",
    model={"name": "claude-sonnet-4-20250514", "version": "2025-05-14", "provider": "anthropic"},
)

@trace(collector)
def call_llm(messages):
    return client.messages.create(
        model="claude-sonnet-4-20250514", messages=messages, max_tokens=4096,
    )
```

### Async

```python
from sentric import TrajectoryCollector, atrace

@atrace(collector)
async def call_llm(messages):
    return await client.chat.completions.create(model="gpt-4o", messages=messages)
```

### Custom Normalizer

For any LLM not auto-detected, provide a normalizer function:

```python
def my_normalizer(response):
    """Return (messages, input_tokens, output_tokens)."""
    return (
        [{"role": "assistant", "content": response.text}],
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

@trace(collector, normalizer=my_normalizer)
def call_llm(messages):
    return custom_client.generate(messages)
```

The normalizer returns a 3-tuple: `(messages: list[dict], input_tokens: int, output_tokens: int)`.

### Streaming

The `@trace` decorator auto-detects streaming responses from OpenAI and Anthropic. It wraps the stream transparently -- your code iterates normally while sentric accumulates the response:

```python
@trace(collector)
def call_llm_stream(messages):
    return client.chat.completions.create(
        model="gpt-4o", messages=messages, stream=True,
    )

# Use as normal -- iteration is transparent
for chunk in call_llm_stream(messages):
    print(chunk.choices[0].delta.content or "", end="")
```

The full message is logged when the stream completes.

## Token & Cost Tracking

### Automatic (via @trace)

Token counts are extracted automatically from OpenAI and Anthropic responses.

### Manual

```python
collector.add_tokens(input_tokens=1500, output_tokens=500)
```

### Cost Calculation

Costs are calculated automatically for known models using built-in pricing tables:

- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini, o3-mini
- **Anthropic**: claude-opus-4-20250514, claude-sonnet-4-20250514, claude-haiku-3-5-20241022

For custom pricing:

```python
collector = TrajectoryCollector(
    task_id="my-task", domain="code",
    model={
        "name": "my-fine-tune",
        "version": "v1",
        "provider": "openai",
        "pricing": {"input": 5.0 / 1_000_000, "output": 15.0 / 1_000_000},
    },
)
```

Or add cost manually:

```python
collector.add_cost(0.0023)  # USD
```

## Loading & Scoring

### Load Episodes

```python
from sentric import load_episode, load_episodes

# Single file
episode = load_episode("data/trajectories/abc123.json")

# All episodes in a directory
episodes = load_episodes("data/trajectories/")

# With filtering
code_episodes = load_episodes(
    "data/trajectories/",
    filter_fn=lambda ep: ep["domain"] == "code",
)
```

### Score Episodes

After running a verifier (unit tests, human review, etc.), update the trajectory:

```python
from sentric import score_episode

score_episode(
    "data/trajectories/abc123.json",
    reward=1.0,
    success=True,
    verifier="unit_tests",
)
```

This updates the file in place and sets `verified_at` to the current timestamp.

### Export to JSONL

For fine-tuning pipelines that expect JSONL:

```python
from sentric import load_episodes, export_jsonl

episodes = load_episodes("data/trajectories/")
export_jsonl(episodes, "training_data.jsonl")
```

## Environment Capture

Record Python version, platform, package versions, and git hash:

```python
collector.capture_env()
collector.save_episode()
```

Adds an `_env` key to metadata:

```json
{
  "metadata": {
    "_env": {
      "python_version": "3.12.0",
      "platform": "macOS-14.0-arm64",
      "packages": {"sentric": "0.2.0", "openai": "1.12.0"},
      "git_hash": "abc123..."
    }
  }
}
```

## Format Importers

Migrate historical trajectory data from other tools:

```python
from sentric.importers import from_openai_messages, from_langsmith, from_wandb, import_directory

# Single record
episode = from_openai_messages({
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "gpt-4o",
    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
})

# LangSmith run export
episode = from_langsmith(langsmith_run_dict)

# W&B trace export
episode = from_wandb(wandb_trace_dict)

# Batch import a directory
episodes = import_directory("exports/langsmith/", format="langsmith")
# format options: "langsmith", "openai", "openai_messages", "wandb"
```

All importers are pure dict transforms with zero external dependencies. Unknown fields generate a warning via `logging.getLogger('sentric.importers')`.

## OpenTelemetry Integration

Opt-in distributed tracing with zero overhead when OTel is not installed:

```bash
pip install sentric[otel]
```

```python
# No code changes needed. If opentelemetry-api is installed,
# sentric automatically emits spans and events:
#   - Parent span per episode with task_id, model, domain attributes
#   - Span events per message with role, content preview, tool info
#   - Final attributes: message_count, total_tokens, cost
```

The OTel code path uses `functools.lru_cache` to check for the package once. If not installed, all OTel functions are no-ops that return immediately.

## CLI: Trajectory Viewer

Pretty-print trajectories in the terminal:

```bash
# View a single trajectory (stats + messages)
sentric view data/trajectories/abc123.json

# Stats summary only
sentric view data/trajectories/abc123.json --stats

# Raw JSON (for piping)
sentric view data/trajectories/abc123.json --json

# Full content (no truncation)
sentric view data/trajectories/abc123.json --full

# List all trajectories in a directory
sentric view data/trajectories/
```

Color-coded roles: system (gray), user (blue), assistant (green), tool (yellow). Respects `NO_COLOR` environment variable.

## Output Schema

Every saved trajectory follows this schema:

```json
{
  "episode_id": "uuid",
  "task_id": "string",
  "domain": "string",
  "model": {
    "name": "string",
    "version": "string",
    "provider": "string"
  },
  "messages": [
    {
      "role": "system | user | assistant | tool",
      "content": "string | null",
      "tool_calls": [{"id": "string", "name": "string", "arguments": "string"}],
      "tool_call_id": "string"
    }
  ],
  "reward": "float | null",
  "success": "bool | null",
  "verifier": "string | null",
  "verified_at": "ISO 8601 | null",
  "created_at": "ISO 8601",
  "duration_ms": "int",
  "total_tokens": "int | null",
  "input_tokens": "int | null",
  "output_tokens": "int | null",
  "total_cost_usd": "float | null",
  "metadata": {}
}
```

| Field | Description |
|-------|-------------|
| `episode_id` | UUID generated per episode |
| `task_id` | Your identifier for the task |
| `domain` | Task category |
| `model` | Model name, version, and provider |
| `messages` | Full conversation in OpenAI message format |
| `reward` | Numeric score (set via `score_episode()`) |
| `success` | Boolean pass/fail (set via `score_episode()`) |
| `verifier` | What verified the result (set via `score_episode()`) |
| `verified_at` | When verification happened (auto-set by `score_episode()`) |
| `created_at` | Episode creation timestamp (UTC ISO 8601) |
| `duration_ms` | Wall-clock time from collector creation to save |
| `total_tokens` | Input + output tokens (null if not tracked) |
| `input_tokens` | Prompt/input tokens (null if not tracked) |
| `output_tokens` | Completion/output tokens (null if not tracked) |
| `total_cost_usd` | Calculated or manual cost in USD (null if unknown) |
| `metadata` | Arbitrary dict for your use |

## Performance

Sentric is designed for near-zero overhead:

- `add_message()`: ~0.4 us per call
- `save_episode()`: ~0.35 ms for 100 messages
- `save_episode_async()`: ~9 us submit time (non-blocking)
- Memory: ~275 bytes per message

Install `sentric[fast]` for orjson-based JSON serialization (3-10x faster than stdlib json). Falls back to stdlib json automatically when orjson is not installed.

## Testing

```bash
pip install sentric[dev]
pytest tests/ -v
```

## License

MIT
