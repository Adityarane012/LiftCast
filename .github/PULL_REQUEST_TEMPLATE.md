## Description

Please include a summary of the change and which issue or feature it relates to.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Documentation update
- [ ] Test suite addition

## Hacktoberfest & Privacy Checklist
- [ ] **Zero Data Leakage:** Confirmed no personal workout data (`*.csv`, `*.db`, personal notes) is tracked or committed.
- [ ] **Local-First Verification:** No external cloud endpoints or API dependencies added that violate offline/privacy constraints.
- [ ] **Strict Numeric Guard:** Gemma narrates computed numbers only; all numbers are calculated by deterministic code.
- [ ] **Test Suite:** All unit tests pass locally (`pytest -q` passes 51/51 tests).
- [ ] **Code Quality:** Type hints and docstrings maintained for any modified pure functions.
