"""JSON serialization abstraction.

Uses orjson when available (3-10x faster), falls back to stdlib json.
"""

import functools


@functools.lru_cache(maxsize=1)
def _has_orjson() -> bool:
    try:
        import orjson  # noqa: F401
        return True
    except ImportError:
        return False


def dumps(obj: dict, *, indent: bool = True) -> str:
    """Serialize obj to a JSON string."""
    if _has_orjson():
        import orjson
        option = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=option).decode()
    else:
        import json
        if indent:
            return json.dumps(obj, indent=2)
        return json.dumps(obj, separators=(",", ":"))


def dumps_bytes(obj: dict, *, indent: bool = True) -> bytes:
    """Serialize obj to JSON bytes (avoids decode step for file writes)."""
    if _has_orjson():
        import orjson
        option = orjson.OPT_INDENT_2 if indent else 0
        return orjson.dumps(obj, option=option)
    else:
        return dumps(obj, indent=indent).encode()
