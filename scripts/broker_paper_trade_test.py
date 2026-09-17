#!/usr/bin/env python3
"""Validation experiment: does a real InferenceService decision (FR-10)
reach a real execution venue's order API?

This is the same kind of one-off validation exercise as
`scripts/finrl_crosscheck.py` — not a permanent part of the serving path.
`InferenceService.predict()` is called exactly the way `/predict` calls it
in production; the only new thing this script does is take that output and
route it to Alpaca's *paper* trading API (CLAUDE.md §7's recorded
deviation) instead of just returning it as JSON.

Requires ALPACA_API_KEY / ALPACA_SECRET_KEY (a free Alpaca paper account —
see CLAUDE.md §7 for setup). Never touches Alpaca's live endpoint;
`BrokerConfig.base_url` (src/config.py) is a fixed constant, not something
this script or any env var can override.

Usage:
    python scripts/broker_paper_trade_test.py
    python scripts/broker_paper_trade_test.py --notional 25
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA  # noqa: E402
from src.dataops.repository import MarketRepository  # noqa: E402
from src.execution.alpaca_broker import AlpacaBroker, BrokerError, route_decision  # noqa: E402
from src.serving.inference import InferenceService, NoActiveModelError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("broker_paper_trade_test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Route one real inference decision to Alpaca paper trading")
    parser.add_argument("--notional", type=float, default=50.0, help="dollar amount per BUY/SELL order")
    args = parser.parse_args()

    try:
        broker = AlpacaBroker()
    except BrokerError as exc:
        log.error("%s", exc)
        return 2

    account = broker.get_account()
    log.info(
        "connected to Alpaca paper account %s: equity=$%s buying_power=$%s status=%s",
        account["id"], account["equity"], account["buying_power"], account["status"],
    )

    repo = MarketRepository()
    service = InferenceService(repo=repo)
    if service.active_version_id is None:
        log.error("no active model — promote one first (see CLAUDE.md §10's bootstrap walkthrough)")
        return 1

    try:
        # A flat starting state, deliberately: this validates that a
        # decision reaches the venue, not that Alpaca's actual position
        # state is kept in sync with the RL environment's own weights —
        # that would be a separate, larger feature.
        flat_positions = {t: 0.0 for t in DATA.tickers}
        result = service.predict(flat_positions, cash_weight=1.0)
    except NoActiveModelError as exc:
        log.error("%s", exc)
        return 1

    log.info(
        "decision from version %s (failsafe_triggered=%s, vix=%s)",
        result["version_id"], result["failsafe_triggered"], result["vix_at_decision"],
    )

    for decision in result["decisions"]:
        ticker, action = decision["ticker"], decision["discrete_action"]
        try:
            response = route_decision(broker, ticker, action, args.notional)
        except Exception:  # noqa: BLE001 - one ticker's rejection shouldn't abort the rest
            log.exception("%s (%s): order routing failed", ticker, action)
            continue

        if response is None:
            log.info("%s: %s (no order placed)", ticker, action)
            continue

        order_id = response.get("id")
        log.info("%s: %s -> Alpaca order %s (status=%s)", ticker, action, order_id, response.get("status"))

        if order_id and action != "LIQUIDATE":
            time.sleep(2.0)  # market orders usually fill fast in paper trading during market hours
            order = broker.get_order(order_id)
            log.info("%s: order %s settled at status=%s filled_qty=%s", ticker, order_id, order["status"], order.get("filled_qty"))

    after = broker.get_account()
    log.info("account after: equity=$%s buying_power=$%s", after["equity"], after["buying_power"])
    positions = broker.get_positions()
    log.info("open positions: %s", [(p["symbol"], p["qty"]) for p in positions] or "none")

    broker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
