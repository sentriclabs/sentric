# Feature 10: Trajectory Viewer CLI

**Priority**: P2
**Effort**: M
**Status**: planned

## Summary
A `sentric view` CLI command that pretty-prints trajectories in the terminal. Developers debugging agent runs shouldn't need to open JSON files in an editor. Color-coded roles, collapsible tool calls, token/cost summaries.

## Requirements
- `sentric view <path>` — pretty-print a single trajectory
- `sentric view <directory>` — list trajectories with summary stats
- `sentric view <path> --turns` — show turn-by-turn with role colors
- `sentric view <path> --stats` — show token/cost/duration summary only
- `sentric view <path> --json` — raw JSON output (for piping)
- Color-coded roles: system=gray, user=blue, assistant=green, tool=yellow
- Truncate long content with `--full` flag to show everything

## Design Notes
**Performance**: Use `sys.stdout.write()` directly, not `print()`. For large trajectories, stream output rather than building full string.

**No external deps**: Use ANSI escape codes directly — no `rich` or `click` dependency. Keep the package zero-dep.

## Files to Modify
- `sentric/cli.py` — new file
- `pyproject.toml` — add `[project.scripts]` entry point

## Tests Needed
- `test_cli_view_single()` — verify output for a sample trajectory
- `test_cli_view_stats()` — verify stats mode
- `test_cli_view_json()` — verify JSON passthrough

## Notes
