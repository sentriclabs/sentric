# Feature 01: Version Export & Package Metadata

**Priority**: P0
**Effort**: S
**Status**: planned

## Summary
Expose `__version__` in the package and add basic package metadata. Table-stakes for any published Python package — lets users check what version they're running, enables version-gated behavior downstream.

## Requirements
- `sentric.__version__` returns the version string (synced with pyproject.toml)
- Single source of truth for version (read from pyproject.toml at build time or hardcode and sync)

## Design Notes
Simplest approach: hardcode `__version__ = "0.2.0"` in `__init__.py`. Avoids importlib overhead. Update in lockstep with pyproject.toml on release.

## Files to Modify
- `sentric/__init__.py` — add `__version__`
- `pyproject.toml` — bump version to 0.2.0

## Tests Needed
- `test_version()` — assert `sentric.__version__` is a string matching semver pattern

## Notes
