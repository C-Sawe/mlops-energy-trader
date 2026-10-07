"""Pydantic request/response contracts for the inference API (IR-04, IR-05).

NFR-08: malformed input must be rejected, not crash the service. Pydantic
validation at the boundary is what makes that true without every endpoint
hand-rolling its own checks.
"""
from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator

from src.config import DATA


class PredictRequest(BaseModel):
    """The caller's current portfolio state.

    This is the one piece of state `InferenceService` cannot know on its
    own: the market data needed for the rest of the observation is fetched
    from the repository, but only the caller (the actual portfolio holder)
    knows what it currently holds.
    """

    positions: dict[str, float] = Field(
        description="current weight per ticker, in [-1, 1]"
    )
    cash_weight: float = Field(ge=0.0, le=1.0)

    @field_validator("positions")
    @classmethod
    def _tickers_match_the_configured_universe(cls, value: dict[str, float]) -> dict[str, float]:
        expected = set(DATA.tickers)
        got = set(value.keys())
        if got != expected:
            raise ValueError(
                f"positions must cover exactly the configured universe {sorted(expected)}, got {sorted(got)}"
            )
        for ticker, weight in value.items():
            if not (-1.0 <= weight <= 1.0):
                raise ValueError(f"{ticker}: weight {weight} outside [-1, 1]")
        return value


class TickerDecision(BaseModel):
    ticker: str
    raw_weight: float | None
    discrete_action: str  # BUY | HOLD | SELL | LIQUIDATE


class PredictResponse(BaseModel):
    decisions: list[TickerDecision]
    failsafe_triggered: bool
    vix_at_decision: float | None
    version_id: str | None
    decided_at: datetime


class EquityPoint(BaseModel):
    date: date
    equity_value: float
    rolling_sharpe_30d: float | None = None
    max_drawdown: float | None = None
    cumulative_return: float | None = None
    # Buy-and-hold over the same replay window and costs (§10.1). None on
    # snapshots recorded before the benchmark was persisted.
    benchmark_equity: float | None = None


class TelemetryResponse(BaseModel):
    """FR-18: what the dashboard's equity curve, Sharpe and drawdown charts render."""

    points: list[EquityPoint]


class DecisionLogEntry(BaseModel):
    decision_id: str
    version_id: str
    run_id: str
    train_start: date
    train_end: date
    eval_start: date
    eval_end: date
    decided_at: datetime
    ticker: str
    raw_weight: float | None
    discrete_action: str
    vix_at_decision: float | None
    failsafe_triggered: bool


class DecisionLogResponse(BaseModel):
    """FR-19: a paginated decision log."""

    items: list[DecisionLogEntry]
    page: int
    page_size: int
    total: int


class CTStatusResponse(BaseModel):
    """FR-20: serving / evaluating / retraining, for the dashboard's status indicator."""

    status: str  # SERVING | EVALUATING | RETRAINING
    active_version_id: str | None
    # Presentation label only ("September 2026 #2") — active_version_id
    # above remains the real identifier for traceability (NFR-07).
    active_model_label: str | None
    rolling_sharpe: float | None
    target_sharpe_threshold: float
    last_evaluated_at: datetime | None
    # Set when a below-threshold Sharpe would have triggered FR-14 but a
    # volatile VIX deferred it instead — the CT loop's own fail-safe on
    # itself, distinct from I5/FR-12's inference-side one.
    last_retrain_deferred_at: datetime | None
    # NFR-05: set when a retrain thread genuinely crashed (not "candidate
    # lost the comparison", which is a normal, already-logged outcome).
    # Deliberately no error-message field — the raw exception text belongs
    # in server logs, not a public response.
    last_retrain_failed_at: datetime | None
    last_ingest_date: date | None
    current_vix: float | None
    vix_critical_threshold: float
    # Training activity in the reporting window — every logged run, not
    # specifically autonomous retrains (see MarketRepository.get_cycle_stats).
    cycle_window_days: int
    training_runs: int
    promoted_count: int
    rejected_count: int
    # Of training_runs, those the CT loop started itself (trigger_reason
    # ct_bootstrap / ct_decay), and how many of them were promoted. Runs
    # logged before trigger_reason existed are in neither count.
    autonomous_retrains: int = 0
    autonomous_promoted: int = 0
    # Not FR-20 itself (that's the CT loop) — shows FR-01's "without manual
    # intervention" ingestion actually ticking, the same way the fields
    # above show FR-13/14 actually ticking.
    ingestion_status: str  # IDLE | INGESTING
    last_ingestion_attempted_at: datetime | None
    last_ingestion_ok: bool | None


class HealthResponse(BaseModel):
    status: str
    active_version_id: str | None


class Candle(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    is_imputed: bool


class CandlesResponse(BaseModel):
    """Daily OHLC for one ticker — split/dividend adjusted (FR-02/DR-03)."""

    ticker: str
    candles: list[Candle]


class PaperPosition(BaseModel):
    ticker: str
    qty: float | None
    avg_entry_price: float | None
    current_price: float | None
    market_value: float | None
    unrealized_pl: float | None
    unrealized_plpc: float | None


class PaperFill(BaseModel):
    order_id: str
    ticker: str
    side: str  # buy | sell
    signal_date: date
    decision_id: str | None  # I6: the decision that produced this order
    filled_qty: float | None
    filled_avg_price: float | None
    filled_at: datetime | None


class PaperEquityPoint(BaseModel):
    date: date
    equity: float


class PaperStatusResponse(BaseModel):
    """The forward paper account (CLAUDE.md §7's Alpaca deviation). Simulated
    money on Alpaca's paper venue — never a profitability claim (§1)."""

    enabled: bool
    status: str  # IDLE | TRADING
    next_run_at: datetime | None
    last_run_at: datetime | None
    last_outcome: str | None
    last_signal_date: date | None
    last_cycle_failsafe: bool | None
    equity: float | None
    cash: float | None
    synced_at: datetime | None
    positions: list[PaperPosition]
    equity_curve: list[PaperEquityPoint]
    fills: list[PaperFill]
