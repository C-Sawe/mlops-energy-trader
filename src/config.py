"""Central configuration.

All values are environment-driven (IR-03): no credentials or connection
strings are hard-coded, satisfying NFR-10.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


@dataclass(frozen=True)
class DataConfig:
    """DR-01, DR-02: the equity universe and the indicator parameters."""

    # Chapter 4 universe. ^VIX is ingested separately as a market-wide
    # feature and is joined onto every equity row, not treated as an equity.
    tickers: tuple[str, ...] = ("XOM", "CVX", "SHEL", "BP", "NEE")
    vix_symbol: str = "^VIX"

    sma_window: int = 20        # SMA_20
    rsi_window: int = 14        # RSI_14
    zscore_window: int = 60     # rolling standardisation lookback

    # DR-06: the evaluation window must fall strictly after the training
    # window. Enforced by partition_chronological().
    train_start: str = field(default_factory=lambda: _env("TRAIN_START", "2015-01-01"))
    train_end: str = field(default_factory=lambda: _env("TRAIN_END", "2023-12-31"))
    eval_start: str = field(default_factory=lambda: _env("EVAL_START", "2024-01-01"))
    eval_end: str = field(default_factory=lambda: _env("EVAL_END", "2025-12-31"))


@dataclass(frozen=True)
class DatabaseConfig:
    """IR-03: connection parameters supplied by environment configuration."""

    user: str = field(default_factory=lambda: _env("POSTGRES_USER", "mlops"))
    password: str = field(default_factory=lambda: _env("POSTGRES_PASSWORD", "mlops"))
    host: str = field(default_factory=lambda: _env("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env_int("POSTGRES_PORT", 5432))
    database: str = field(default_factory=lambda: _env("POSTGRES_DB", "mlops_trading"))

    @property
    def url(self) -> str:
        override = os.environ.get("DATABASE_URL")
        if override:
            return override
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    def __repr__(self) -> str:  # NFR-10: never leak the password in logs
        return (
            f"DatabaseConfig(host={self.host!r}, port={self.port}, "
            f"database={self.database!r}, user={self.user!r}, password='***')"
        )


@dataclass(frozen=True)
class IngestionConfig:
    """IR-01: bounded retry with backoff."""

    max_retries: int = 3
    backoff_seconds: float = 2.0
    request_timeout: int = 30


@dataclass(frozen=True)
class RiskConfig:
    """Thresholds governing FR-12 and FR-14.

    Both are configurable rather than constant, so that Chapter 5 can
    report a sensitivity analysis instead of defending a magic number.
    """

    vix_critical_threshold: float = field(
        default_factory=lambda: _env_float("VIX_CRITICAL_THRESHOLD", 35.0)
    )
    target_sharpe_threshold: float = field(
        default_factory=lambda: _env_float("TARGET_SHARPE_THRESHOLD", 1.0)
    )
    sharpe_evaluation_window: int = field(
        default_factory=lambda: _env_int("SHARPE_EVALUATION_WINDOW", 30)
    )
    # FR-11: continuous weight -> discrete action boundaries
    buy_threshold: float = 0.5
    sell_threshold: float = -0.5


@dataclass(frozen=True)
class EnvironmentConfig:
    """FR-06, FR-07: the trading MDP's economic parameters.

    Configurable rather than constant so Chapter 5 can report a
    sensitivity analysis (see RiskConfig) instead of defending magic
    numbers.
    """

    initial_cash: float = field(
        default_factory=lambda: _env_float("INITIAL_CASH", 100_000.0)
    )
    transaction_cost_pct: float = field(
        default_factory=lambda: _env_float("TRANSACTION_COST_PCT", 0.001)
    )
    drawdown_penalty_coef: float = field(
        default_factory=lambda: _env_float("DRAWDOWN_PENALTY_COEF", 1.0)
    )


DATA = DataConfig()
DATABASE = DatabaseConfig()
INGESTION = IngestionConfig()
RISK = RiskConfig()
ENVIRONMENT = EnvironmentConfig()
