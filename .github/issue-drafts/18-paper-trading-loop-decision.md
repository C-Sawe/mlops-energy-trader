title: "Decision needed: continuous Alpaca *paper* trading, or keep it as a one-off validation?"
labels: decision, needs-supervisor, execution
---
## Current state (verified 2026-09-30)
There is **no live or continuous trading of any kind**. The Alpaca integration (`src/execution/alpaca_broker.py`) was exercised once, on 2026-09-17, by `scripts/broker_paper_trade_test.py`:
- the model's decision was HOLD × 5, so no orders were placed;
- a separate manual $10 notional BUY of XOM filled at $162.876 (0.0613 sh), then LIQUIDATE closed it back to flat.

Nothing routes decisions to Alpaca on a schedule, and the cloud deployment doesn't include it either.

## Options
1. **Keep as-is** (recommended for the thesis): a one-off validation of the two order shapes, recorded as the §7 deviation.
2. **Scheduled paper routing**: a daily job that routes the served decision to the paper account and records fills next to `trading_decision`. More demonstration value, but it's new scope and invites "is it profitable?" framing (§1).
3. **Real money / broader universe**: explicitly out of scope; needs the supervisor, and deferred by the owner.

## If option 2 is chosen
- [ ] Supervisor approval recorded
- [ ] Paper-only guarantee kept (fixed `base_url`, no env override)
- [ ] Fills stored with `version_id` (I6) so every order traces to its weights
- [ ] Dashboard labels it "paper account", never P&L as a result
