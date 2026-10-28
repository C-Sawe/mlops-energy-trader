"""Pipeline activity events (FR-20 as a history, NFR-07): each autonomous
step records what it did, in words an operator can read, at the moment it
does it. The dashboard's activity timeline shows these and nothing else.

Recording is observability, not part of the pipeline's job. A failed write
is logged and swallowed here, so a database hiccup while recording "model
kept serving" can never interrupt ingestion, evaluation or a retrain.
"""
from __future__ import annotations

import logging
import math

from src.dataops.repository import MarketRepository

logger = logging.getLogger(__name__)


def fmt(value: float | None, places: int = 2) -> str:
    """Numbers as the timeline shows them; `n/a` rather than `nan`/`inf`."""
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{places}f}"


def finite_or_none(value: float | None) -> float | None:
    """JSON can't carry inf/nan, and the bootstrap's incumbent Sharpe is -inf."""
    return float(value) if value is not None and math.isfinite(value) else None


def emit(
    repo: MarketRepository | None,
    kind: str,
    summary: str,
    *,
    source: str = "scheduled",
    detail: dict | None = None,
    run_id: str | None = None,
    version_id: str | None = None,
) -> None:
    try:
        (repo or MarketRepository()).record_event(
            kind, summary, source=source, detail=detail, run_id=run_id, version_id=version_id
        )
    except Exception:  # noqa: BLE001 - recording must never break the step being recorded
        logger.exception("could not record pipeline event %s: %s", kind, summary)
