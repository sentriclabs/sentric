# Feature 09: Trajectory Format Importers

**Priority**: P2
**Effort**: L
**Status**: planned

## Summary
Import trajectories from other logging formats (LangSmith, Weights & Biases, raw OpenAI message logs) into Sentric's schema. Reduces onboarding friction for companies already using other tools — they can migrate historical data without re-running agents.

## Requirements
- `from sentric.importers import from_langsmith, from_openai_messages, from_wandb`
- Each importer takes the foreign format and returns a Sentric episode dict
- Batch import: `import_directory(path, format="langsmith") -> list[dict]`
- Best-effort field mapping with clear warnings for unmappable fields
- Zero required dependencies — importers work on raw dicts/JSON, not SDK objects

## Design Notes
**Performance**: Pure dict transformation — no external calls, no SDK imports needed. Should handle 10k+ trajectories without memory issues (process one at a time).

## Files to Modify
- `sentric/importers/` — new subpackage
- `sentric/importers/langsmith.py`
- `sentric/importers/openai_messages.py`
- `sentric/importers/wandb.py`

## Tests Needed
- Test each importer with sample data
- Test unmappable field warnings
- Test batch import

## Notes
Need to research exact formats of LangSmith and W&B trace exports.
