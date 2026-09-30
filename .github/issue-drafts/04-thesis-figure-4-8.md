title: "Figure 4.8 dashboard wireframe contradicts the spec (tech tickers, FR-11 actions, threshold at ~0.2)"
labels: thesis, figures, priority:medium
---
## Problem
The Figure 4.8 wireframe:
1. shows **AAPL, MSFT, NVDA…** instead of the approved energy universe **XOM, CVX, SHEL, BP, NEE** (changing the universe needs supervisor sign-off, CLAUDE.md §2);
2. shows action values that contradict **FR-11** (BUY ≥ 0.5, SELL ≤ −0.5, otherwise HOLD; LIQUIDATE only from the FR-12 fail-safe);
3. draws the retraining threshold line at **~0.2** instead of **1.0** (`target_sharpe_threshold`, FR-14).

## Suggested fix
The real dashboard now exists (`frontend/`). A screenshot of it in mock-data mode ("Example data" banner visible) is more honest than a redrawn wireframe, and it matches the implementation by construction. If the chapter needs a *design-stage* wireframe, redraw it with the three corrections above.

## Acceptance criteria
- [ ] Only the five approved tickers appear
- [ ] Actions shown are consistent with FR-11/FR-12
- [ ] Threshold line is at 1.0 and labelled
