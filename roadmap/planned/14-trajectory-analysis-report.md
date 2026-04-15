# Feature 14: Trajectory Analysis & PDF Report

**Priority**: P0
**Effort**: L
**Status**: planned

## Summary
Load all collected trajectories from the SWE-bench PoC, analyze them, and generate a comprehensive PDF report. This is the final deliverable proving the SDK and workflow.

## Requirements
- Load trajectories from both mini-swe-agent and custom harness runs
- Compare: turn count, token usage, tool call patterns, success/failure rates
- Identify failure patterns per business plan Section 6.5 (brittle patches, wrong-file edits, incomplete test coverage)
- Generate PDF with:
  1. SDK audit findings and improvements made
  2. Architecture overview of the SDK
  3. SWE-bench PoC methodology
  4. Results comparison (framework vs custom harness)
  5. Trajectory examples (annotated)
  6. Failure pattern taxonomy
  7. Recommendations for the 2-week demo execution

## Design Notes
Use `reportlab` or `weasyprint` for PDF generation. Keep it simple — markdown-to-PDF with basic formatting.

## Files to Modify
- `examples/analyze.py` — new file, analysis script
- `examples/generate_report.py` — new file, PDF generation

## Tests Needed
- N/A (one-time deliverable)

## Notes
