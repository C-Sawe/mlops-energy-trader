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
from sqlalchemy.orm import Session, joinedload, sessionmaker

from src.config import DATABASE
from src.dataops.models import (
    Base,
    MarketObservation,
    ModelRun,
    ModelVersion,
    PaperAccountSnapshot,
    PaperCycle,
    PaperOrder,
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
        """FR-19: a paginated decision log, most recent first.

        Joins through to `model_run` for `run_id` and the partition
        boundaries — NFR-07's traceability chain (decision -> version ->
        run -> data partition) is only actually *visible* if a caller can
        get all of it back in one call, not just the version_id FK.
        """
        with self.session() as session:
            total = session.execute(
                select(func.count()).select_from(TradingDecision)
            ).scalar_one()

            stmt = (
                select(TradingDecision)
                .options(joinedload(TradingDecision.version).joinedload(ModelVersion.run))
                .order_by(TradingDecision.decided_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            rows = session.execute(stmt).scalars().all()
            items = [
                {
                    "decision_id": r.decision_id,
                    "version_id": r.version_id,
                    "run_id": r.version.run_id,
                    "train_start": r.version.run.train_start,
                    "train_end": r.version.run.train_end,
                    "eval_start": r.version.run.eval_start,
                    "eval_end": r.version.run.eval_end,
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

    def get_promotion_sequence(self, promoted_at) -> int:
        """1-indexed position of `promoted_at` among all promotions in its
        calendar month — lets the dashboard label a model "September 2026
        #2" rather than just its version_id. `version_id` stays the real
        identifier everywhere NFR-07 traceability actually depends on
        (the decision log, the version->run->partition chain); this is a
        presentation label only, not a replacement identifier — two models
        promoted in the same month (which happened twice in one session,
        2026-09) would otherwise both read as an unqualified "September
        2026"."""
        ts = pd.Timestamp(promoted_at)
        month_start = ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_end = month_start + pd.DateOffset(months=1)
        with self.session() as session:
            return session.execute(
                select(func.count())
                .select_from(ModelVersion)
                .where(
                    ModelVersion.promoted_at.is_not(None),
                    ModelVersion.promoted_at >= month_start,
                    ModelVersion.promoted_at < month_end,
                    ModelVersion.promoted_at <= ts,
                )
            ).scalar_one()

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

    def get_cycle_stats(self, since: date | str) -> dict:
        """Training runs logged in the window, and how many were promoted
        vs rejected by the FR-17 acceptance gate.

        Counts every logged run regardless of trigger — there is no field
        distinguishing a `CTOrchestrator` retrain from a manual
        `scripts/train_agent.py` sweep, so this is honestly "training
        activity in the window", not specifically "autonomous retrains".
        The dashboard labels it accordingly rather than overclaiming.
        """
        since_ts = pd.Timestamp(since)
        with self.session() as session:
            total = session.execute(
                select(func.count()).select_from(ModelRun).where(ModelRun.created_at >= since_ts)
            ).scalar_one()
            promoted = session.execute(
                select(func.count())
                .select_from(ModelVersion)
                .join(ModelRun, ModelVersion.run_id == ModelRun.run_id)
                .where(ModelRun.created_at >= since_ts, ModelVersion.promoted_at.is_not(None))
            ).scalar_one()
            rejected = session.execute(
                select(func.count())
                .select_from(ModelRun)
                .where(ModelRun.created_at >= since_ts, ModelRun.status == "REJECTED")
            ).scalar_one()
        return {"total_runs": total, "promoted": promoted, "rejected": rejected}

    # ---------------------------------------------------------- ingest info
    def latest_ingest_info(self) -> dict | None:
        """Most recent observation across the whole universe — the
        dashboard's "Last ingest" and "Current VIX" status fields."""
        stmt = (
            select(MarketObservation)
            .order_by(MarketObservation.observation_date.desc(), MarketObservation.ingested_at.desc())
            .limit(1)
        )
        with self.session() as session:
            row = session.execute(stmt).scalars().first()
            if row is None:
                return None
            return {
                "date": row.observation_date,
                "vix": float(row.vix) if row.vix is not None else None,
                "ingested_at": row.ingested_at,
            }

    # -------------------------------------------------------------- candles
    def load_candles(self, ticker: str, start: date | str, end: date | str) -> list[dict]:
        """Daily OHLC for one ticker, for the dashboard's candlestick chart.
        These are the stored, split/dividend-adjusted prices (FR-02/DR-03),
        so bars before a recent ex-dividend date sit slightly below the raw
        prices a broker fill is quoted in."""
        start = pd.Timestamp(start).date()
        end = pd.Timestamp(end).date()
        stmt = (
            select(MarketObservation)
            .where(
                MarketObservation.ticker == ticker,
                MarketObservation.observation_date >= start,
                MarketObservation.observation_date <= end,
            )
            .order_by(MarketObservation.observation_date)
        )
        with self.session() as session:
            return [
                {
                    "date": r.observation_date,
                    "open": float(r.open_price),
                    "high": float(r.high_price),
                    "low": float(r.low_price),
                    "close": float(r.close_price),
                    "is_imputed": r.is_imputed,
                }
                for r in session.execute(stmt).scalars().all()
            ]

    # ------------------------------------------------------- paper trading
    def has_paper_cycle(self, signal_date: date) -> bool:
        with self.session() as session:
            return session.get(PaperCycle, signal_date) is not None

    def record_paper_cycle(
        self,
        signal_date: date,
        version_id: str | None,
        failsafe_triggered: bool,
        orders: list[dict],
        orders_failed: int,
    ) -> None:
        """One cycle and the orders it submitted, in a single transaction —
        a cycle row without its orders (or the reverse) would misreport
        what was actually sent to the broker."""
        with self.session() as session:
            with session.begin():
                session.add(
                    PaperCycle(
                        signal_date=signal_date,
                        version_id=version_id,
                        failsafe_triggered=failsafe_triggered,
                        orders_submitted=len(orders),
                        orders_failed=orders_failed,
                    )
                )
                session.flush()
                for o in orders:
                    session.add(PaperOrder(signal_date=signal_date, **o))

    def latest_paper_cycle(self) -> dict | None:
        stmt = select(PaperCycle).order_by(PaperCycle.signal_date.desc()).limit(1)
        with self.session() as session:
            c = session.execute(stmt).scalars().first()
            if c is None:
                return None
            return {
                "signal_date": c.signal_date,
                "ran_at": c.ran_at,
                "version_id": c.version_id,
                "failsafe_triggered": c.failsafe_triggered,
                "orders_submitted": c.orders_submitted,
                "orders_failed": c.orders_failed,
            }

    def unfilled_paper_order_ids(self) -> list[str]:
        """Orders still waiting on a terminal broker status."""
        open_states = ("new", "accepted", "pending_new", "partially_filled", "held", "accepted_for_bidding")
        stmt = select(PaperOrder.order_id).where(PaperOrder.status.in_(open_states))
        with self.session() as session:
            return list(session.execute(stmt).scalars().all())

    def update_paper_order(
        self,
        order_id: str,
        status: str,
        filled_qty: float | None,
        filled_avg_price: float | None,
        filled_at: datetime | None,
    ) -> None:
        with self.session() as session:
            with session.begin():
                order = session.get(PaperOrder, order_id)
                if order is None:
                    return
                order.status = status
                order.filled_qty = filled_qty
                order.filled_avg_price = filled_avg_price
                order.filled_at = filled_at

    def list_paper_fills(self, since: date | str, ticker: str | None = None) -> list[dict]:
        """Filled paper orders — the entry/exit markers on the candlestick chart."""
        stmt = select(PaperOrder).where(
            PaperOrder.filled_at.is_not(None),
            PaperOrder.signal_date >= pd.Timestamp(since).date(),
        )
        if ticker:
            stmt = stmt.where(PaperOrder.ticker == ticker)
        stmt = stmt.order_by(PaperOrder.filled_at)
        with self.session() as session:
            return [
                {
                    "order_id": o.order_id,
                    "ticker": o.ticker,
                    "side": o.side,
                    "signal_date": o.signal_date,
                    "decision_id": o.decision_id,
                    "filled_qty": float(o.filled_qty) if o.filled_qty is not None else None,
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price is not None else None,
                    "filled_at": o.filled_at,
                }
                for o in session.execute(stmt).scalars().all()
            ]

    def record_paper_account(
        self, snapshot_date: date, equity: float, cash: float, positions: list[dict], synced_at: datetime
    ) -> None:
        """Idempotent per date, like `record_snapshot`: the last sync of a
        day is that day's point on the forward equity curve."""
        with self.session() as session:
            with session.begin():
                existing = session.get(PaperAccountSnapshot, snapshot_date)
                if existing is None:
                    session.add(
                        PaperAccountSnapshot(
                            snapshot_date=snapshot_date, equity=equity, cash=cash,
                            positions=positions, synced_at=synced_at,
                        )
                    )
                else:
                    existing.equity = equity
                    existing.cash = cash
                    existing.positions = positions
                    existing.synced_at = synced_at

    def list_paper_account(self, start: date | str | None = None) -> list[dict]:
        stmt = select(PaperAccountSnapshot).order_by(PaperAccountSnapshot.snapshot_date)
        if start is not None:
            stmt = stmt.where(PaperAccountSnapshot.snapshot_date >= pd.Timestamp(start).date())
        with self.session() as session:
            return [
                {
                    "date": r.snapshot_date,
                    "equity": float(r.equity),
                    "cash": float(r.cash),
                    "positions": list(r.positions or []),
                    "synced_at": r.synced_at,
                }
                for r in session.execute(stmt).scalars().all()
            ]
