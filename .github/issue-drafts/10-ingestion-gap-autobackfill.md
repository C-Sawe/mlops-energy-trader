title: "Ingestion scheduler can't close gaps longer than its 120-day window; detect them and backfill automatically"
labels: enhancement, dataops, FR-01
---
## Problem
`run_ingestion_tick()` refetches a fixed 120-day trailing window (`DEFAULT_TRAILING_WINDOW_DAYS`). A downtime longer than that leaves a permanent hole. A ~4.5-month gap (2025-12-30 → 2026-05-20) happened for real and needed a manual `run_ingestion.py` backfill (CLAUDE.md §13). A cloud host that's down for a long stretch hits the same problem.

## Proposal
On each tick, read `latest_ingest_info()`. If the last stored date is more than `window − 20` days old, widen the fetch start to `last_stored − 40 days` (warm-up for SMA_20/RSI_14; keep the existing drop of the first `_INDICATOR_WARMUP` rows per ticker so warm history is never clobbered, per the 2026-09-20 fix).

## Acceptance criteria
- [ ] Test: seed data ending 200 days before `end`, tick once, assert no missing trading days and no NaN `sma_20`/`rsi_14` in the gap
- [ ] Existing `test_ingestion_tick_does_not_clobber_already_warm_history_behind_it` still passes
- [ ] Logged clearly when a gap backfill happens (FR-20 visibility)
