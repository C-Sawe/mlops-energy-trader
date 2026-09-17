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


class TelemetryResponse(BaseModel):
    """FR-18: what the dashboard's equity curve, Sharpe and drawdown charts render."""

    points: list[EquityPoint]


class DecisionLogEntry(BaseModel):
    decision_id: str
    version_id: str
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
    rolling_sharpe: float | None
    target_sharpe_threshold: float
    last_evaluated_at: datetime | None


class HealthResponse(BaseModel):
    status: str
    active_version_id: str | None
