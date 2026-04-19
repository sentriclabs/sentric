"""Batch import utilities."""

import json
import logging
import os
from pathlib import Path

_log = logging.getLogger("sentric.importers")

_FORMAT_MAP = {
    "langsmith": "from_langsmith",
    "openai": "from_openai_messages",
    "openai_messages": "from_openai_messages",
    "wandb": "from_wandb",
}


def import_directory(path: str | Path, format: str) -> list[dict]:
    """Batch import all JSON files in a directory.

    Args:
        path: Directory containing JSON files to import.
        format: One of "langsmith", "openai", "openai_messages", "wandb".

    Returns:
        List of Sentric episode dicts.
    """
    if format not in _FORMAT_MAP:
        raise ValueError(f"Unknown format: {format!r}. Must be one of: {', '.join(sorted(_FORMAT_MAP))}")

    # Lazy import to avoid circular deps
    from sentric import importers
    importer_fn = getattr(importers, _FORMAT_MAP[format])

    path = Path(path)
    episodes = []

    for entry in sorted(os.scandir(path), key=lambda e: e.name):
        if not entry.name.endswith(".json") or not entry.is_file():
            continue
        try:
            with open(entry.path) as f:
                data = json.load(f)
            episodes.append(importer_fn(data))
        except (json.JSONDecodeError, KeyError) as exc:
            _log.warning("Skipping %s: %s", entry.name, exc)

    return episodes
