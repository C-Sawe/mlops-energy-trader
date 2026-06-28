# RLOps Module

**Sprint 2 & 3 — Status: ⏳ Awaiting Sprint 1**

This module contains the Reinforcement Learning training environment and PPO agent:

- FinRL simulated MDP environment over global energy equities
- PPO agent via Stable Baselines3 (PyTorch backend)
- Risk-averse reward function — penalises Maximum Drawdown (MDD)
- Walk-forward time-series cross-validation to prevent look-ahead bias

## Planned Files

```
rlops/
├── environment.py       # FinRLEnvironment class — MDP wrapper, state/action space
├── agent.py             # PPOAgent class — training, inference, artifact serialisation
├── reward.py            # Custom reward function — Sharpe-weighted, MDD-penalised
├── train.py             # Training entry point — hyperparameter config
└── validate.py          # Walk-forward backtesting evaluation
```

## Implementation begins: Sprint 2 (September 2026)
