"""Persistence layer (FR-05, DR-05, IR-03).

Writes go through a single transaction so that a failure mid-batch rolls
back rather than leaving the feature store partially updated (IR-02).
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
from sqlalchemy import create_engine, select, func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DATABASE
from src.dataops.models import Base, MarketObservation

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

        df = pd.DataFrame(data)
        if not df.empty:
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
