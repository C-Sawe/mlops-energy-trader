"""SQLAlchemy ORM models implementing the Figure 4.7 logical schema.

Entity-to-requirement mapping:
  market_observation  -> DR-02, DR-04, DR-05
  model_run           -> DR-08, NFR-07
  model_version       -> DR-09, NFR-07
  trading_decision    -> FR-19, NFR-07, NFR-11
  portfolio_snapshot  -> FR-13, FR-18
  paper_cycle / paper_order / paper_account_snapshot
                      -> CLAUDE.md §7's Alpaca deviation (forward paper
                         trading); paper_order.decision_id extends I6 to
                         the broker: every order traces to a decision.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    BigInteger,
    JSON,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class MarketObservation(Base):
    """One enriched daily observation for one ticker.

    DR-05 is enforced structurally: (observation_date, ticker) is the
    composite primary key, so a duplicate insert is rejected by the
    database rather than by application code that might be bypassed.
    """

    __tablename__ = "market_observation"

    observation_date: Mapped[date] = mapped_column(Date, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)

    open_price: Mapped[float] = mapped_column(Numeric(18, 6))
    high_price: Mapped[float] = mapped_column(Numeric(18, 6))
    low_price: Mapped[float] = mapped_column(Numeric(18, 6))
    close_price: Mapped[float] = mapped_column(Numeric(18, 6))
    volume: Mapped[int] = mapped_column(BigInteger)

    sma_20: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)
    rsi_14: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)
    vix: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)

    # DR-04: imputed rows stay distinguishable from observed rows.
    is_imputed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint("high_price >= low_price", name="ck_high_ge_low"),
        CheckConstraint("volume >= 0", name="ck_volume_non_negative"),
        Index("ix_market_observation_date", "observation_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<MarketObservation {self.ticker} {self.observation_date} "
            f"close={self.close_price}>"
        )


class ModelRun(Base):
    """A single training run and its resulting metrics.

    DR-08: the exact partition boundaries are stored, so any run can be
    reproduced from the record alone.
    """

    __tablename__ = "model_run"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    mlflow_run_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)

    train_start: Mapped[date] = mapped_column(Date, nullable=False)
    train_end: Mapped[date] = mapped_column(Date, nullable=False)
    eval_start: Mapped[date] = mapped_column(Date, nullable=False)
    eval_end: Mapped[date] = mapped_column(Date, nullable=False)

    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)

    sharpe_ratio: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)
    cumulative_return: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)

    status: Mapped[str] = mapped_column(String(24), default="PENDING", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    versions: Mapped[list["ModelVersion"]] = relationship(back_populates="run")

    __table_args__ = (
        # DR-06: no look-ahead leakage is representable in the record.
        CheckConstraint("eval_start > train_end", name="ck_eval_after_train"),
        CheckConstraint("train_end > train_start", name="ck_train_range_valid"),
        CheckConstraint("eval_end > eval_start", name="ck_eval_range_valid"),
    )


class ModelVersion(Base):
    """A registered, servable artifact produced by a run (DR-09)."""

    __tablename__ = "model_version"

    version_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("model_run.run_id"), nullable=False)

    artifact_uri: Mapped[str] = mapped_column(String(512), nullable=False)
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    retired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    run: Mapped[ModelRun] = relationship(back_populates="versions")
    decisions: Mapped[list["TradingDecision"]] = relationship(back_populates="version")


class TradingDecision(Base):
    """One autonomous action, attributable to the weights that produced it.

    The version_id foreign key is what makes NFR-07 satisfiable: every
    action traces to a model version, a run, and a data partition.
    """

    __tablename__ = "trading_decision"

    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("model_version.version_id"), nullable=False
    )

    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    raw_weight: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)
    discrete_action: Mapped[str] = mapped_column(String(16), nullable=False)
    vix_at_decision: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)

    # NFR-11: fail-safe activations are separable from model decisions,
    # so evaluation can exclude them rather than misattribute them.
    failsafe_triggered: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    version: Mapped[ModelVersion] = relationship(back_populates="decisions")

    __table_args__ = (
        CheckConstraint(
            "discrete_action IN ('BUY', 'HOLD', 'SELL', 'LIQUIDATE')",
            name="ck_action_enum",
        ),
        Index("ix_trading_decision_decided_at", "decided_at"),
    )


class PortfolioSnapshot(Base):
    """Daily portfolio telemetry driving FR-13 and FR-18."""

    __tablename__ = "portfolio_snapshot"

    snapshot_date: Mapped[date] = mapped_column(Date, primary_key=True)
    equity_value: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    rolling_sharpe_30d: Mapped[float | None] = mapped_column(
        Numeric(18, 6), nullable=True
    )
    max_drawdown: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)
    cumulative_return: Mapped[float | None] = mapped_column(
        Numeric(18, 6), nullable=True
    )


class PaperCycle(Base):
    """One daily forward paper-trading cycle, keyed by the market date whose
    close produced the signal. The primary key is what makes the job
    idempotent: a second cycle for the same signal date cannot be recorded,
    so a restart or a manual re-run never trades the same day twice."""

    __tablename__ = "paper_cycle"

    signal_date: Mapped[date] = mapped_column(Date, primary_key=True)
    ran_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    failsafe_triggered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    orders_submitted: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    orders_failed: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)


class PaperOrder(Base):
    """An order sent to the Alpaca *paper* account. Fill fields stay null
    until a later sync sees the fill — orders queued after the close
    execute at the next open, not when they are submitted."""

    __tablename__ = "paper_order"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    signal_date: Mapped[date] = mapped_column(
        ForeignKey("paper_cycle.signal_date"), nullable=False
    )
    decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("trading_decision.decision_id"), nullable=True
    )
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    requested_notional: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    filled_qty: Mapped[float | None] = mapped_column(Numeric(18, 9), nullable=True)
    filled_avg_price: Mapped[float | None] = mapped_column(Numeric(18, 6), nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint("side IN ('buy', 'sell')", name="ck_paper_side"),
        Index("ix_paper_order_ticker", "ticker"),
    )


class PaperAccountSnapshot(Base):
    """The paper account's equity, cash and positions as last synced on a
    given date — the forward equity curve. Last sync of the day wins."""

    __tablename__ = "paper_account_snapshot"

    snapshot_date: Mapped[date] = mapped_column(Date, primary_key=True)
    equity: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    cash: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    positions: Mapped[list] = mapped_column(JSON, default=list)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
