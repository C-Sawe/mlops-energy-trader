title: "Persist a buy-and-hold benchmark during CT evaluation so the live equity chart can show it"
labels: enhancement, dashboard, FR-18
---
## Problem
The equity chart's buy-and-hold series only exists in mock data. `CTOrchestrator.evaluate()` replays only the incumbent, never a baseline, so the live view has nothing to compare against. CLAUDE.md §10.1 says "a Sharpe Ratio is meaningless alone".

## Proposal
In `evaluate()`, replay `baselines.buy_and_hold` over the same window (cheap: no model) and store `benchmark_equity` alongside each `performance_snapshot`. Expose it in `/telemetry`; `adaptTelemetry` already handles a two-series shape from the mock path.

## Acceptance criteria
- [ ] Schema change + test that the benchmark and incumbent use the identical window and cost settings
- [ ] Frontend test for the live two-series case
- [ ] Mock-data label remains; no client-side fabrication
