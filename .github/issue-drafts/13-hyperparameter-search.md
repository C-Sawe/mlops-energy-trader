title: "Hyperparameter search (n_steps, learning rate, net size) with honest trial counting"
labels: research, rlops, needs-supervisor
---
## Context
Every study so far (180-run PPO sweep, 720-run algorithm comparison) keeps SB3 defaults fixed and varies only the seed (CLAUDE.md §10, "Not yet done"). That answers "how variable is this configuration", not "is there a better one".

## Scope guardrails
- Scope extension: check with the supervisor first (§2).
- **Every configuration tried counts toward `n_trials` in `deflated_sharpe_ratio()`**, including abandoned ones (§10.3). A search makes the selection-bias problem *worse*, not better.
- Tune on a validation slice carved from the training window (as the ensemble study did). Never on eval (I1).

## Acceptance criteria
- [ ] Search space and budget written down *before* running
- [ ] Results reported as pooled mean ± std and DSR with the full trial count, alongside the untuned baseline
- [ ] Framed per §1: evidence about the architecture, not a profitability claim
