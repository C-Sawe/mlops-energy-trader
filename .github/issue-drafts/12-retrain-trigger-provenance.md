title: "Record what triggered each training run (CT decay / bootstrap / manual sweep) in model_run"
labels: enhancement, traceability, NFR-07
---
## Problem
`model_run` can't tell an autonomous `CTOrchestrator` retrain apart from a `scripts/train_agent.py` sweep, so the dashboard has to hedge with "Training runs, last N days" instead of "Autonomous retrains". For the thesis, "the loop retrained itself N times" is the core claim, and today it can't be queried precisely.

## Proposal
Add `trigger` (`ct_decay`, `ct_bootstrap`, `manual_sweep`, `analysis`) plus, for CT runs, the `rolling_sharpe` and VIX that caused it. Default existing rows to `NULL` (unknown), not guessed.

## Acceptance criteria
- [ ] Column + migration-safe default
- [ ] `_trigger_retrain` passes the reason through; tests cover decay vs bootstrap
- [ ] `get_cycle_stats()` and the Cycle-history card can report autonomous retrains separately
