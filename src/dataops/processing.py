"""Feature engineering and normalisation (FR-02, FR-03, FR-04).

Every transformation here is strictly backward-looking. This is the single
most important property in the DataOps layer: a forward-looking window
leaks future information into a past observation, which inflates
backtest performance and produces an agent that cannot reproduce its
results live. DR-07 states the requirement; the tests enforce it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA

REQUIRED_COLUMNS = ("date", "ticker", "open", "high", "low", "close", "volume")


def _validate(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """FR-02 / DR-04: forward-fill gaps, flagging every imputed row.

    Reindexes each ticker onto the union of trading days observed across
    the universe, so that a gap in one ticker's supply does not silently
    shorten its series relative to the others.
    """
    _validate(df)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    calendar = pd.DatetimeIndex(sorted(df["date"].unique()))
    frames = []

    for ticker, group in df.groupby("ticker", sort=True):
        group = group.sort_values("date").set_index("date")
        group = group[~group.index.duplicated(keep="first")]

        reindexed = group.reindex(calendar)
        # A row is imputed if its close was absent before filling.
        reindexed["is_imputed"] = reindexed["close"].isna()
        reindexed["ticker"] = ticker

        price_cols = ["open", "high", "low", "close", "volume"]
        reindexed[price_cols] = reindexed[price_cols].ffill()

        # Leading rows before a ticker's first observation cannot be
        # forward-filled and are dropped rather than back-filled, since
        # back-filling would be a forward-looking operation.
        reindexed = reindexed[reindexed["close"].notna()]

        reindexed.index.name = "date"
        frames.append(reindexed.reset_index())

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def adjust_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    """FR-02 / DR-03: rescale OHLC by the split/dividend adjustment factor.

    yfinance supplies ``adj_close`` when ``auto_adjust=False``. The ratio
    adj_close/close is the cumulative adjustment factor for that row; the
    remaining OHLC fields are scaled by it so that price continuity holds
    across corporate actions. Where ``adj_close`` is absent the frame is
    assumed pre-adjusted and returned unchanged.
    """
    df = df.copy()
    if "adj_close" not in df.columns:
        return df

    factor = df["adj_close"] / df["close"]
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * factor
    # Share count moves inversely to price under a split.
    df["volume"] = (df["volume"] / factor).round()

    return df.drop(columns=["adj_close"])


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI.

    Uses exponential smoothing with alpha = 1/window, which is Wilder's
    original formulation, rather than a simple rolling mean of gains and
    losses. The two differ materially in the first ~3*window periods.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # A window with no losses is RSI 100 by definition; guard the 0/0 case.
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """FR-03: SMA_20 and RSI_14, computed per ticker."""
    _validate(df)
    df = df.copy().sort_values(["ticker", "date"])

    df["sma_20"] = df.groupby("ticker", sort=False)["close"].transform(
        lambda s: s.rolling(window=DATA.sma_window, min_periods=DATA.sma_window).mean()
    )
    df["rsi_14"] = df.groupby("ticker", sort=False)["close"].transform(
        lambda s: _rsi(s, DATA.rsi_window)
    )
    return df.reset_index(drop=True)


def attach_vix(df: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """FR-03: join the market-wide VIX level onto every equity row.

    VIX is a market-wide feature, not a per-ticker one, so it is merged
    on date alone and forward-filled across any date the index did not
    publish but the equities traded.
    """
    df = df.copy()
    vix = vix.copy()
    df["date"] = pd.to_datetime(df["date"])
    vix["date"] = pd.to_datetime(vix["date"])

    vix_series = (
        vix[["date", "close"]]
        .rename(columns={"close": "vix"})
        .drop_duplicates(subset="date")
        .sort_values("date")
    )

    merged = df.sort_values("date").merge(vix_series, on="date", how="left")
    merged["vix"] = merged["vix"].ffill()
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)


def normalize_rolling(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    window: int | None = None,
) -> pd.DataFrame:
    """FR-04 / DR-07: rolling Z-score using backward-looking windows only.

    For each column c, the standardised value at time t uses the mean and
    standard deviation of the window ending at t inclusive. pandas'
    ``rolling`` is backward-looking by default; this is asserted by the
    leakage test rather than assumed.

    A zero-variance window yields 0.0 rather than a division by zero,
    which is the correct reading: a constant series carries no deviation
    from its own mean.
    """
    _validate(df)
    window = window or DATA.zscore_window
    columns = columns or ["open", "high", "low", "close", "volume", "sma_20", "rsi_14", "vix"]
    columns = [c for c in columns if c in df.columns]

    df = df.copy().sort_values(["ticker", "date"])

    for col in columns:
        grouped = df.groupby("ticker", sort=False)[col]
        mean = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=window).mean()
        )
        std = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=window).std(ddof=0)
        )
        z = (df[col] - mean) / std
        df[f"{col}_z"] = z.where(std != 0, 0.0)

    return df.reset_index(drop=True)


def partition_chronological(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DR-06: split into train and out-of-sample sets with no overlap.

    Raises if the evaluation window does not begin strictly after the
    training window ends. Silently permitting an overlap here would
    invalidate every downstream performance figure, so it is a hard error.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    ts, te = pd.Timestamp(train_start), pd.Timestamp(train_end)
    es, ee = pd.Timestamp(eval_start), pd.Timestamp(eval_end)

    if es <= te:
        raise ValueError(
            f"look-ahead leakage: eval_start ({es.date()}) must be strictly "
            f"after train_end ({te.date()})"
        )
    if te <= ts or ee <= es:
        raise ValueError("each window must have positive duration")

    train = df[(df["date"] >= ts) & (df["date"] <= te)]
    evaluation = df[(df["date"] >= es) & (df["date"] <= ee)]
    return train.reset_index(drop=True), evaluation.reset_index(drop=True)


def build_feature_frame(
    equities: pd.DataFrame, vix: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Full DataOps transformation, in the order the requirements demand.

    Order matters: corporate-action adjustment precedes indicator
    computation (indicators on unadjusted prices are wrong across a
    split), and normalisation comes last so that it standardises the
    final feature values rather than intermediate ones.
    """
    df = adjust_corporate_actions(equities)
    df = impute_missing(df)
    df = compute_indicators(df)
    if vix is not None:
        df = attach_vix(df, vix)
    df = normalize_rolling(df)
    return df
