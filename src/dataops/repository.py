"""Persistence layer (FR-05, DR-05, IR-03).

Writes go through a single transaction so that a failure mid-batch rolls
back rather than leaving the feature store partially updated (IR-02).

Also owns access to the serving-layer tables (`trading_decision`,
`portfolio_snapshot`, `model_version`) — they live in the same database, and
this project has one repository class per database, not one per table, so
`src/serving` and `src/orchestration` depend on `dataops.repository` rather
than opening their own connections (NFR-06: dependencies flow one way).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, select, func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DATABASE
from src.dataops.models import (
    Base,
    MarketObservation,
    ModelVersion,
    PortfolioSnapshot,
    TradingDecision,
)

logger = logging.getLogger(__name__)

PERSISTED_COLUMNS = [
    "observation_date",
    "ticker",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "sma_20",
    "rsi_14",
    "vix",
    "is_imputed",
]

_FRAME_TO_COLUMN = {
    "date": "observation_date",
    "open": "open_price",
    "high": "high_price",
    "low": "low_price",
    "close": "close_price",
}


class MarketRepository:
    """Owns all access to the market observation store."""

    def __init__(self, url: str | None = None, echo: bool = False):
        self.url = url or DATABASE.url
        self.engine: Engine = create_engine(self.url, echo=echo, future=True)
        self._Session = sessionmaker(bind=self.engine, future=True)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def drop_schema(self) -> None:
        Base.metadata.drop_all(self.engine)

    def session(self) -> Session:
        return self._Session()

    @staticmethod
    def _to_records(df: pd.DataFrame) -> list[dict]:
        frame = df.rename(columns=_FRAME_TO_COLUMN).copy()

        if "is_imputed" not in frame.columns:
            frame["is_imputed"] = False

        for col in PERSISTED_COLUMNS:
            if col not in frame.columns:
                frame[col] = None

        frame = frame[PERSISTED_COLUMNS]
        frame["observation_date"] = pd.to_datetime(frame["observation_date"]).dt.date
        frame["is_imputed"] = frame["is_imputed"].astype(bool)
        frame["volume"] = frame["volume"].fillna(0).astype("int64")

        records = frame.to_dict(orient="records")
        for rec in records:
            for key, value in rec.items():
                if pd.isna(value):
                    rec[key] = None
        return records

    def persist(self, df: pd.DataFrame) -> int:
        """FR-05: write observations idempotently.

        Re-ingesting an overlapping date range is a normal event — the
        scheduled job refetches a trailing window to pick up late
        corrections — so an existing (date, ticker) row is updated rather
        than raising. DR-05 still holds: exactly one row per key survives.
        """
        records = self._to_records(df)
        if not records:
            return 0

        written = 0
        with self.session() as session:
            with session.begin():
                for rec in records:
                    existing = session.get(
                        MarketObservation,
                        (rec["observation_date"], rec["ticker"]),
                    )
                    if existing is None:
                        session.add(MarketObservation(**rec))
                    else:
                        for key, value in rec.items():
                            setattr(existing, key, value)
                    written += 1

        logger.info("persisted %d observations", written)
        return written

    def load_partition(
        self, start: date | str, end: date | str, tickers: tuple[str, ...] | None = None
    ) -> pd.DataFrame:
        """Load a chronological slice for training or evaluation."""
        start = pd.Timestamp(start).date()
        end = pd.Timestamp(end).date()

        stmt = select(MarketObservation).where(
            MarketObservation.observation_date >= start,
            MarketObservation.observation_date <= end,
        )
        if tickers:
            stmt = stmt.where(MarketObservation.ticker.in_(tickers))
        stmt = stmt.order_by(
            MarketObservation.ticker, MarketObservation.observation_date
        )

        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            data = [
                {
                    "date": r.observation_date,
                    "ticker": r.ticker,
                    "open": float(r.open_price),
                    "high": float(r.high_price),
                    "low": float(r.low_price),
                    "close": float(r.close_price),
                    "volume": int(r.volume),
                    "sma_20": float(r.sma_20) if r.sma_20 is not None else None,
                    "rsi_14": float(r.rsi_14) if r.rsi_14 is not None else None,
                    "vix": float(r.vix) if r.vix is not None else None,
                    "is_imputed": r.is_imputed,
                }
                for r in rows
            ]

        if not data:
            # pd.DataFrame([]) has zero *columns*, not just zero rows —
            # callers (normalize_rolling's column check, in particular)
            # need a correctly-shaped empty frame to fail gracefully rather
            # than raising a confusing "missing required columns" error
            # that has nothing to do with the actual problem (no data in
            # the requested range, e.g. ingestion hasn't caught up yet).
            return pd.DataFrame(
                columns=[
                    "date", "ticker", "open", "high", "low", "close",
                    "volume", "sma_20", "rsi_14", "vix", "is_imputed",
                ]
            )

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def latest_state(self, ticker: str) -> dict | None:
        """Most recent observation for a ticker, for live inference."""
        stmt = (
            select(MarketObservation)
            .where(MarketObservation.ticker == ticker)
            .order_by(MarketObservation.observation_date.desc())
            .limit(1)
        )
        with self.session() as session:
            row = session.execute(stmt).scalars().first()
            if row is None:
                return None
            return {
                "date": row.observation_date,
                "ticker": row.ticker,
                "open": float(row.open_price),
                "high": float(row.high_price),
                "low": float(row.low_price),
                "close": float(row.close_price),
                "volume": int(row.volume),
                "sma_20": float(row.sma_20) if row.sma_20 is not None else None,
                "rsi_14": float(row.rsi_14) if row.rsi_14 is not None else None,
                "vix": float(row.vix) if row.vix is not None else None,
            }

    def count(self) -> int:
        with self.session() as session:
            return session.execute(
                select(func.count()).select_from(MarketObservation)
            ).scalar_one()

    # ----------------------------------------------------------- decisions
    def record_decision(
        self,
        version_id: str,
        ticker: str,
        raw_weight: float | None,
        discrete_action: str,
        vix_at_decision: float | None,
        failsafe_triggered: bool = False,
    ) -> str:
        """FR-19, NFR-07, NFR-11, I6: one autonomous action, attributable to
        the version that produced it. `failsafe_triggered` is stored rather
        than inferred, so a fail-safe activation is never confused with a
        model decision in later analysis."""
        decision = TradingDecision(
            version_id=version_id,
            ticker=ticker,
            raw_weight=raw_weight,
            discrete_action=discrete_action,
            vix_at_decision=vix_at_decision,
            failsafe_triggered=failsafe_triggered,
        )
        with self.session() as session:
            with session.begin():
                session.add(decision)
                session.flush()
                return decision.decision_id

    def list_decisions(self, page: int = 1, page_size: int = 20) -> tuple[list[dict], int]:
        """FR-19: a paginated decision log, most recent first."""
        with self.session() as session:
            total = session.execute(
                select(func.count()).select_from(TradingDecision)
            ).scalar_one()

            stmt = (
                select(TradingDecision)
                .order_by(TradingDecision.decided_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            rows = session.execute(stmt).scalars().all()
            items = [
                {
                    "decision_id": r.decision_id,
                    "version_id": r.version_id,
                    "decided_at": r.decided_at,
                    "ticker": r.ticker,
                    "raw_weight": float(r.raw_weight) if r.raw_weight is not None else None,
                    "discrete_action": r.discrete_action,
                    "vix_at_decision": float(r.vix_at_decision) if r.vix_at_decision is not None else None,
                    "failsafe_triggered": r.failsafe_triggered,
                }
                for r in rows
            ]
        return items, total

    # ----------------------------------------------------------- snapshots
    def record_snapshot(
        self,
        snapshot_date: date | str,
        equity_value: float,
        rolling_sharpe_30d: float | None = None,
        max_drawdown: float | None = None,
        cumulative_return: float | None = None,
    ) -> None:
        """FR-13, FR-18: idempotent like `persist()` — the CT orchestrator
        recomputes today's snapshot on every scheduled pass, so re-writing
        the same date must update, not duplicate."""
        snapshot_date = pd.Timestamp(snapshot_date).date()
        with self.session() as session:
            with session.begin():
                existing = session.get(PortfolioSnapshot, snapshot_date)
                if existing is None:
                    session.add(
                        PortfolioSnapshot(
                            snapshot_date=snapshot_date,
                            equity_value=equity_value,
                            rolling_sharpe_30d=rolling_sharpe_30d,
                            max_drawdown=max_drawdown,
                            cumulative_return=cumulative_return,
                        )
                    )
                else:
                    existing.equity_value = equity_value
                    existing.rolling_sharpe_30d = rolling_sharpe_30d
                    existing.max_drawdown = max_drawdown
                    existing.cumulative_return = cumulative_return

    def list_snapshots(self, start: date | str, end: date | str) -> pd.DataFrame:
        """FR-18: the equity curve, rolling Sharpe and max drawdown series
        the dashboard renders."""
        start = pd.Timestamp(start).date()
        end = pd.Timestamp(end).date()
        stmt = (
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.snapshot_date >= start, PortfolioSnapshot.snapshot_date <= end)
            .order_by(PortfolioSnapshot.snapshot_date)
        )
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
            data = [
                {
                    "date": r.snapshot_date,
                    "equity_value": float(r.equity_value),
                    "rolling_sharpe_30d": float(r.rolling_sharpe_30d) if r.rolling_sharpe_30d is not None else None,
                    "max_drawdown": float(r.max_drawdown) if r.max_drawdown is not None else None,
                    "cumulative_return": float(r.cumulative_return) if r.cumulative_return is not None else None,
                }
                for r in rows
            ]
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------ versions
    def get_active_version(self) -> dict | None:
        """I5/FR-16: the version currently being served, if any."""
        stmt = select(ModelVersion).where(ModelVersion.is_active.is_(True))
        with self.session() as session:
            row = session.execute(stmt).scalars().first()
            if row is None:
                return None
            return {
                "version_id": row.version_id,
                "run_id": row.run_id,
                "artifact_uri": row.artifact_uri,
                "promoted_at": row.promoted_at,
            }

    def promote_version(self, version_id: str) -> None:
        """FR-17: activate `version_id`, retiring whatever was active.

        Retiring rather than deleting the incumbent's record is deliberate
        — NFR-07 requires every past decision stay traceable to the version
        that produced it, including versions no longer serving.
        """
        now = datetime.now(timezone.utc)
        with self.session() as session:
            with session.begin():
                stmt = select(ModelVersion).where(ModelVersion.is_active.is_(True))
                for incumbent in session.execute(stmt).scalars().all():
                    incumbent.is_active = False
                    incumbent.retired_at = now

                candidate = session.get(ModelVersion, version_id)
                if candidate is None:
                    raise ValueError(f"no such model_version: {version_id}")
                candidate.is_active = True
                candidate.promoted_at = now
