"""Utilities to load, score, and export saved trajectories."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def load_episode(path) -> dict:
    """Load a single trajectory JSON file.

    Args:
        path: Path to the trajectory JSON file.

    Returns:
        The episode as a dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(path)
    with open(path) as f:
        return json.load(f)


def load_episodes(directory, filter_fn=None) -> list[dict]:
    """Load all trajectory JSON files from a directory.

    Uses os.scandir() for efficient directory listing. Optionally filters
    episodes using a predicate function applied after loading.

    Args:
        directory: Path to the directory containing trajectory JSON files.
        filter_fn: Optional function that takes an episode dict and returns
            bool. Only episodes where filter_fn returns True are included.

    Returns:
        List of episode dicts, sorted by filename.
    """
    directory = Path(directory)
    episodes = []

    entries = sorted(
        (e for e in os.scandir(directory) if e.is_file() and e.name.endswith(".json")),
        key=lambda e: e.name,
    )

    for entry in entries:
        with open(entry.path) as f:
            episode = json.load(f)
        if filter_fn is None or filter_fn(episode):
            episodes.append(episode)

    return episodes


def score_episode(path, reward=None, success=None, verifier=None) -> dict:
    """Update scoring fields on a saved trajectory.

    Read-modify-write: loads the file, updates the specified fields,
    writes it back, and returns the updated dict. Only updates fields
    that are explicitly passed (non-None).

    Args:
        path: Path to the trajectory JSON file.
        reward: Numeric reward value (e.g. 0.0 to 1.0).
        success: Boolean indicating task success.
        verifier: String identifying the verification method.

    Returns:
        The updated episode dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    with open(path) as f:
        episode = json.load(f)

    if reward is not None:
        episode["reward"] = reward
    if success is not None:
        episode["success"] = success
    if verifier is not None:
        episode["verifier"] = verifier

    if any(v is not None for v in (reward, success, verifier)):
        episode["verified_at"] = datetime.now(timezone.utc).isoformat()

    with open(path, "w") as f:
        json.dump(episode, f, indent=2)

    return episode


def export_jsonl(episodes, output_path) -> Path:
    """Write a list of episode dicts to a JSONL file, line-by-line.

    Args:
        episodes: List of episode dicts.
        output_path: Path for the output JSONL file.

    Returns:
        Path to the written JSONL file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for episode in episodes:
            f.write(json.dumps(episode))
            f.write("\n")

    return output_path
