# Feature 04: Episode Loader — load, score, export

**Priority**: P0
**Effort**: M
**Status**: planned

## Summary
Add utilities to load saved trajectories, annotate them with scores, and export to JSONL. These are the basic data operations anyone needs after collecting trajectories. Without them, users have to write boilerplate `json.load()` loops.

## Requirements
- `load_episode(path) -> dict` — load a single trajectory JSON file
- `load_episodes(directory, filter_fn=None) -> list[dict]` — load all trajectories from a directory, optionally filtered
- `score_episode(path, reward=None, success=None, verifier=None) -> dict` — update scoring fields on a saved trajectory, return updated dict
- `export_jsonl(episodes, output_path) -> Path` — write a list of episode dicts to a JSONL file

## Design Notes
**Performance**: 
- `load_episodes()` should use `os.scandir()` not `glob` for directory listing (faster, fewer syscalls)
- For large directories (1000+ files), consider using a generator with `yield` to avoid loading everything into memory
- `export_jsonl()` should write line-by-line, not build a giant string
- All file I/O should use buffered reads/writes

**`score_episode()`**: Read-modify-write with `json.load` + `json.dump`. Updates `reward`, `success`, `verifier`, and sets `verified_at` to current UTC timestamp. Only updates fields that are explicitly passed (non-None).

**`filter_fn`**: Takes an episode dict, returns bool. Applied after loading. Examples: `lambda e: e["success"] is True`, `lambda e: e["domain"] == "code"`.

## Files to Modify
- `sentric/loader.py` — new file
- `sentric/__init__.py` — export new functions

## Tests Needed
- `test_load_episode()` — load a saved trajectory, verify structure
- `test_load_episodes()` — load directory of trajectories
- `test_load_episodes_with_filter()` — verify filter function works
- `test_load_episodes_empty_dir()` — empty directory returns empty list
- `test_score_episode()` — verify scoring fields updated on disk
- `test_score_episode_partial()` — only updates passed fields
- `test_export_jsonl()` — verify JSONL format, line-by-line
- `test_export_jsonl_roundtrip()` — export then load back

## Notes
- Eng review: add error-path tests: load_episode(missing), load_episode(bad JSON), score_episode(missing)
- Eng review: score_episode() race condition is a documented limitation — no file locking at v0.2
