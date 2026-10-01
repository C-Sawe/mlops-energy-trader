#!/usr/bin/env python3
"""Sprint 1 entry point: ingest, enrich, persist.

Usage:
    python scripts/run_ingestion.py --start 2015-01-01 --end 2025-12-31
    python scripts/run_ingestion.py --start 2024-01-01 --end 2024-03-01 --dry-run

Exit codes:
    0  success
    1  ingestion failed (IR-02: nothing was written)
    2  configuration or database error
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA, DATABASE  # noqa: E402
from src.dataops.ingestion import IngestionError, fetch_market_data  # noqa: E402
from src.dataops.processing import build_feature_frame  # noqa: E402
from src.dataops.repository import MarketRepository  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingestion")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest and enrich market data")
    parser.add_argument("--start", default=DATA.train_start)
    parser.add_argument("--end", default=DATA.eval_end)
    parser.add_argument(
        "--tickers", nargs="*", default=None, help="override the configured universe"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="fetch and transform but do not persist"
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="continue when individual tickers fail (exploratory use only)",
    )
    args = parser.parse_args()

    tickers = tuple(args.tickers) if args.tickers else DATA.tickers
    log.info("universe: %s", ", ".join(tickers))
    log.info("range: %s to %s", args.start, args.end)
    log.info("target: %r", DATABASE)

    # --- fetch (FR-01, IR-01, IR-02) ------------------------------------
    try:
        result = fetch_market_data(
            args.start, args.end, tickers=tickers, strict=not args.lenient
        )
    except IngestionError as exc:
        log.error("ingestion aborted, nothing written: %s", exc)
        return 1

    if result.failed_tickers:
        log.warning("failed tickers: %s", ", ".join(result.failed_tickers))
    log.info("fetched %d raw equity rows", len(result.equities))

    # --- transform (FR-02, FR-03, FR-04) --------------------------------
    frame = build_feature_frame(result.equities, result.vix)
    imputed = int(frame["is_imputed"].sum())
    log.info(
        "enriched to %d rows across %d tickers (%d imputed)",
        len(frame), frame["ticker"].nunique(), imputed,
    )
    if imputed:
        pct = 100.0 * imputed / len(frame)
        log.warning("%.2f%% of rows are forward-filled", pct)

    ready = frame["close_z"].notna().sum()
    log.info("%d rows have a complete normalised feature vector", ready)

    if args.dry_run:
        log.info("dry run: no rows persisted")
        print(frame.tail(5).to_string(index=False))
        return 0

    # --- persist (FR-05, DR-05) -----------------------------------------
    try:
        repo = MarketRepository()
        repo.create_schema()
        written = repo.persist(frame)
    except Exception as exc:  # noqa: BLE001
        log.error("database error: %s", exc)
        log.error("is PostgreSQL running?  docker compose up -d postgres")
        return 2

    log.info("persisted %d rows; store now holds %d", written, repo.count())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
