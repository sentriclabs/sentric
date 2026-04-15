# Feature 05: README & Documentation Overhaul

**Priority**: P0
**Effort**: M
**Status**: planned

## Summary
The README has missing imports in examples, undocumented schema fields, no async examples, and no guidance on loading/scoring. Rewrite to be complete, correct, and enterprise-ready.

## Requirements
- All code examples must be copy-pasteable (include imports, handle missing deps)
- Document every public API method with signature and example
- Document the output JSON schema (every field, especially reward/success/verifier/verified_at)
- Add async usage section
- Add loading/scoring/export section
- Add "Who uses this" section for enterprise context
- Add normalizer contract documentation (tuple vs single return)
- Add performance notes section
- Add environment capture documentation

## Design Notes
Structure: Quick Start → Installation → Core Concepts → API Reference → Advanced Usage → Schema Reference → Contributing

## Files to Modify
- `README.md` — full rewrite

## Tests Needed
- N/A (documentation)

## Notes
