# RLOps Module

**Sprint 2 — Status: ✅ Complete.** **Sprint 3 (PPO agent) — Status: ⏳ Not started.**

- `environment.py` — `TradingEnvironment`, a Gymnasium-compatible MDP implemented directly
  to spec (FR-06, FR-07, FR-11), rather than via FinRL. See the recorded deviation in
  `CLAUDE.md` §8: FinRL's `StockTradingEnv` imports successfully but its reward has no
  override hook and its actions are hmax-scaled share counts, not continuous [-1, 1] target
  weights.
- `baselines.py` — buy-and-hold, equal-weight-rebalanced, all-cash, and random policies, run
  through the same environment accounting as any learned agent (CLAUDE.md §10.1).

## Planned — Sprint 3

```
rlops/
├── agent.py       # PPOAgent — Stable Baselines3 wrapper, train/predict/save/load
└── registry.py     # MLflow run logging (FR-08) and artifact registration (FR-09)
```
