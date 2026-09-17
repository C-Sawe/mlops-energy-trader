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


@dataclass(frozen=True)
class BrokerConfig:
    """Alpaca paper-trading credentials (NFR-10: never hard-coded, never
    committed). Deliberately paper-trading only — `base_url` is a fixed
    constant, not environment-configurable, so nothing in this codebase can
    be pointed at Alpaca's live-trading endpoint by an env var typo. See
    CLAUDE.md §7's Alpaca entry for why this exists and what it does and
    does not demonstrate.

    Properties, not `field(default_factory=...)`, for the same reason as
    `ServingConfig.bearer_token`: a frozen dataclass field's default factory
    runs once at module import, before a test (or a real run) has had a
    chance to set the environment variable.
    """

    base_url: str = "https://paper-api.alpaca.markets"

    @property
    def api_key(self) -> str | None:
        return os.environ.get("ALPACA_API_KEY")

    @property
    def secret_key(self) -> str | None:
        return os.environ.get("ALPACA_SECRET_KEY")


@dataclass(frozen=True)
class ServingConfig:
    """Sprint 4 note (CLAUDE.md §12): single-user local operation, no user
    table. A single bearer token is proportionate for the exposed surface
    (read telemetry + /predict); it is optional so local dev and the test
    suite need not set one — `None` disables the check entirely, which is
    the deliberate default, not an oversight.

    `bearer_token` is a property, not a `field(default_factory=...)`: a
    plain dataclass field's default factory runs once, at `ServingConfig()`
    construction — since `SERVING` is a module-level singleton and
    `src.config` is imported (and cached) well before any test gets to set
    `API_BEARER_TOKEN`, a frozen field would never see it. A property reads
    the environment fresh on every access, the same pattern
    `DatabaseConfig.url` already uses for `DATABASE_URL`.
    """

    @property
    def bearer_token(self) -> str | None:
        return os.environ.get("API_BEARER_TOKEN")


DATA = DataConfig()
DATABASE = DatabaseConfig()
INGESTION = IngestionConfig()
RISK = RiskConfig()
ENVIRONMENT = EnvironmentConfig()
BROKER = BrokerConfig()
SERVING = ServingConfig()
ENVIRONMENT = EnvironmentConfig()
