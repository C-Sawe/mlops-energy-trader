#!/usr/bin/env python3
"""Remove the blank performance days left by D-20 (one-off repair).

Until 2026-11-24 every check rewrote its whole replay window, and each day,
as it aged out, was left at exactly the initial cash with no Sharpe and no
drawdown. The real values were overwritten and can't be recovered, so this
only removes the blanks, which the charts would otherwise draw as a flat
line. New checks no longer create them.

Usage (a dry run unless --apply is given; back up first, deploy/backup.sh):
    python scripts/clean_snapshot_history.py
    python scripts/clean_snapshot_history.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ENVIRONMENT  # noqa: E402
from src.dataops.repository import MarketRepository  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="delete them; without this nothing is changed")
    args = parser.parse_args(argv)

    repo = MarketRepository()
    dates = repo.placeholder_snapshot_dates(ENVIRONMENT.initial_cash)
    if not dates:
        print("no blank days found")
        return 0
    print(f"{len(dates)} blank days between {dates[0]} and {dates[-1]}")
    if not args.apply:
        print("dry run: nothing changed. Re-run with --apply to delete them.")
        return 0
    print(f"deleted {repo.delete_snapshots(dates)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
