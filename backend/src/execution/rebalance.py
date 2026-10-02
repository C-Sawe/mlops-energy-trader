"""Target weights -> broker orders for the forward paper account.

Pure arithmetic, no I/O, so it is testable without a broker. Mirrors
`TradingEnvironment._rebalance` (FR-06) where the broker allows it, and
departs from it in two recorded ways:

1. **Long-only.** The environment accepts target weights in [-1, 1], so a
   negative weight is a short position. Alpaca does not open short
   positions with fractional/notional orders, so here a negative target is
   clamped to 0 — "be flat in this ticker". The paper account can
   therefore never reproduce the simulator's short exposure; this is a
   Sim2Real gap of the kind CLAUDE.md §1 already concedes, not a bug.
2. **No leverage.** Positive targets summing above 1.0 are scaled down
   proportionally, and buys are capped at cash plus sell proceeds — the
   same "scale back unaffordable buys, never reject" rule CLAUDE.md §7
   records for the environment, without the paper account's 4x margin.

Sells are planned in full (they fund themselves); only buys are scaled.
Alpaca charges no commission, so unlike the environment there is no fee
term in the affordability check.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderIntent:
    ticker: str
    side: str  # "buy" | "sell"
    notional: float | None  # None for a full close
    close: bool = False  # True -> close the whole position, not a sized sell


def clamp_long_only(raw_weights: dict[str, float]) -> dict[str, float]:
    """Negative -> 0, and the positive remainder scaled to sum to at most 1."""
    clamped = {t: max(0.0, min(1.0, float(w))) for t, w in raw_weights.items()}
    total = sum(clamped.values())
    if total > 1.0:
        clamped = {t: w / total for t, w in clamped.items()}
    return clamped


def plan_rebalance(
    raw_weights: dict[str, float],
    holdings: dict[str, float],
    equity: float,
    cash: float,
    min_notional: float = 1.0,
) -> list[OrderIntent]:
    """Orders that move `holdings` (market value per ticker) toward the
    clamped targets. Deltas under `min_notional` are skipped — Alpaca's
    minimum notional order is $1, and below it the "hold" round-trip error
    CLAUDE.md §7's `_DUST_FRACTION` note describes would otherwise produce
    a trade every day."""
    if equity <= 0:
        return []
    targets = clamp_long_only(raw_weights)

    sells: list[OrderIntent] = []
    buys: list[tuple[str, float]] = []
    for ticker, weight in targets.items():
        held = max(0.0, holdings.get(ticker, 0.0))
        delta = weight * equity - held
        if weight == 0.0 and held > 0.0:
            sells.append(OrderIntent(ticker, "sell", None, close=True))
        elif delta <= -min_notional:
            sells.append(OrderIntent(ticker, "sell", round(min(-delta, held), 2)))
        elif delta >= min_notional:
            buys.append((ticker, delta))

    proceeds = sum(
        holdings.get(o.ticker, 0.0) if o.close else (o.notional or 0.0) for o in sells
    )
    budget = max(0.0, cash) + proceeds
    wanted = sum(d for _, d in buys)
    scale = 1.0 if wanted <= budget or wanted == 0 else budget / wanted

    planned = list(sells)
    for ticker, delta in buys:
        notional = round(delta * scale, 2)
        if notional >= min_notional:
            planned.append(OrderIntent(ticker, "buy", notional))
    return planned


def liquidate_all(holdings: dict[str, float]) -> list[OrderIntent]:
    """FR-12/I5's capital preservation on the paper account: close every
    held position outright rather than sizing sells."""
    return [OrderIntent(t, "sell", None, close=True) for t, v in holdings.items() if v > 0.0]
