"""Trajectory viewer CLI.

Usage:
    sentric view <path>           Pretty-print a single trajectory
    sentric view <directory>      List trajectories with summary stats
    sentric view <path> --turns   Show turn-by-turn with role colors
    sentric view <path> --stats   Show token/duration summary only
    sentric view <path> --json    Raw JSON output (for piping)
    sentric view <path> --full    Show full content (no truncation)

No external dependencies — uses ANSI escape codes directly.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GRAY = "\033[90m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"

_ROLE_COLORS = {
    "system": _GRAY,
    "user": _BLUE,
    "assistant": _GREEN,
    "tool": _YELLOW,
}

_MAX_CONTENT_LEN = 200


def _supports_color(stream=None) -> bool:
    """Check if the output stream supports color output."""
    if os.environ.get("NO_COLOR"):
        return False
    stream = stream or sys.stdout
    if not hasattr(stream, "isatty"):
        return False
    return stream.isatty()


def _color(text: str, code: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{code}{text}{_RESET}"


def _truncate(text: str, full: bool) -> str:
    if full or len(text) <= _MAX_CONTENT_LEN:
        return text
    return text[:_MAX_CONTENT_LEN] + f"... ({len(text)} chars)"


def _load_episode(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _format_duration(ms: int | None) -> str:
    if ms is None:
        return "n/a"
    if ms < 1000:
        return f"{ms}ms"
    secs = ms / 1000
    if secs < 60:
        return f"{secs:.1f}s"
    mins = int(secs // 60)
    remaining = secs % 60
    return f"{mins}m {remaining:.0f}s"


def _format_tokens(tokens: int | None) -> str:
    if tokens is None:
        return "n/a"
    if tokens >= 1000:
        return f"{tokens:,}"
    return str(tokens)


def _view_stats(episode: dict, out, use_color: bool):
    """Print summary statistics for an episode."""
    write = out.write

    write(_color("Episode Stats", _BOLD + _CYAN, use_color) + "\n")
    write(_color("─" * 40, _DIM, use_color) + "\n")

    write(f"  Episode ID:    {episode.get('episode_id', 'n/a')}\n")
    write(f"  Task ID:       {episode.get('task_id', 'n/a')}\n")
    write(f"  Domain:        {episode.get('domain', 'n/a')}\n")

    model = episode.get("model", {})
    if isinstance(model, dict):
        write(f"  Model:         {model.get('name', 'n/a')}\n")
        write(f"  Provider:      {model.get('provider', 'n/a')}\n")
    else:
        write(f"  Model:         {model}\n")

    write(f"  Messages:      {len(episode.get('messages', []))}\n")
    write(f"  Duration:      {_format_duration(episode.get('duration_ms'))}\n")
    write(f"  Input tokens:  {_format_tokens(episode.get('input_tokens'))}\n")
    write(f"  Output tokens: {_format_tokens(episode.get('output_tokens'))}\n")
    write(f"  Total tokens:  {_format_tokens(episode.get('total_tokens'))}\n")

    reward = episode.get("reward")
    success = episode.get("success")
    if reward is not None:
        write(f"  Reward:        {reward}\n")
    if success is not None:
        write(f"  Success:       {success}\n")

    write("\n")


def _view_turns(episode: dict, out, use_color: bool, full: bool):
    """Print turn-by-turn message view."""
    write = out.write
    messages = episode.get("messages", [])

    for msg in messages:
        role = msg.get("role", "unknown")
        color = _ROLE_COLORS.get(role, _WHITE)

        write(_color(f"[{role}]", _BOLD + color, use_color))

        content = msg.get("content")
        if content:
            write(f" {_truncate(content, full)}")

        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name", "?")
                write(f"\n  " + _color(f"→ {name}", _YELLOW + _BOLD, use_color))
                args = tc.get("arguments", "")
                if args:
                    write(f"({_truncate(args, full)})")

        tool_call_id = msg.get("tool_call_id")
        if tool_call_id:
            write(f" " + _color(f"[{tool_call_id}]", _DIM, use_color))

        write("\n")

    write("\n")


def _view_single(path: Path, args, out):
    """View a single trajectory file."""
    episode = _load_episode(path)
    use_color = _supports_color(out) and not args.json

    if args.json:
        out.write(json.dumps(episode, indent=2) + "\n")
        return

    if args.stats:
        _view_stats(episode, out, use_color)
        return

    # Default: show stats header then turns
    _view_stats(episode, out, use_color)
    _view_turns(episode, out, use_color, args.full)


def _view_directory(dirpath: Path, out):
    """List trajectories in a directory with summary stats."""
    use_color = _supports_color(out)
    write = out.write

    write(_color(f"Trajectories in {dirpath}", _BOLD + _CYAN, use_color) + "\n")
    write(_color("─" * 60, _DIM, use_color) + "\n")

    entries = sorted(
        (e for e in os.scandir(dirpath) if e.name.endswith(".json") and e.is_file()),
        key=lambda e: e.name,
    )

    if not entries:
        write("  No trajectory files found.\n")
        return

    write(f"  {'File':<40} {'Messages':>8} {'Tokens':>8}\n")
    write(f"  {'─' * 40} {'─' * 8} {'─' * 8}\n")

    for entry in entries:
        try:
            with open(entry.path) as f:
                ep = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        name = entry.name
        if len(name) > 40:
            name = name[:37] + "..."

        n_msgs = len(ep.get("messages", []))
        tokens = _format_tokens(ep.get("total_tokens"))

        write(f"  {name:<40} {n_msgs:>8} {tokens:>8}\n")

    write(f"\n  {len(entries)} trajectory file(s)\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentric",
        description="Sentric trajectory toolkit",
    )
    subparsers = parser.add_subparsers(dest="command")

    view = subparsers.add_parser("view", help="View trajectory files")
    view.add_argument("path", help="Path to a trajectory JSON file or directory")
    view.add_argument("--turns", action="store_true", help="Show turn-by-turn messages")
    view.add_argument("--stats", action="store_true", help="Show summary stats only")
    view.add_argument("--json", action="store_true", help="Raw JSON output")
    view.add_argument("--full", action="store_true", help="Show full content (no truncation)")

    return parser


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "view":
        path = Path(args.path)
        if not path.exists():
            sys.stderr.write(f"Error: {path} not found\n")
            sys.exit(1)

        if path.is_dir():
            _view_directory(path, sys.stdout)
        else:
            _view_single(path, args, sys.stdout)


if __name__ == "__main__":
    main()
