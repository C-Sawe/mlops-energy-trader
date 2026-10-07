"""Market data acquisition (FR-01, IR-01, IR-02).

The retry policy and the failure semantics are the substance of this
module. IR-02 requires that exhausted retries fail loudly and leave the
prior dataset intact rather than writing a partial batch, because a
half-written batch is indistinguishable downstream from a genuine gap in
the market.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd

from src.config import DATA, INGESTION

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """Raised when data cannot be retrieved after exhausting retries."""


@dataclass
class IngestionResult:
    equities: pd.DataFrame
    vix: pd.DataFrame
    requested_tickers: tuple[str, ...]
    failed_tickers: tuple[str, ...]


_PRICE_FIELDS = frozenset({"open", "high", "low", "close", "adj close", "volume"})


def _flatten_single_symbol(columns: pd.MultiIndex, ticker: str) -> pd.Index:
    """Drop the ticker level of a MultiIndex, refusing if it holds >1 symbol.

    yfinance 1.7 returns ``(field, ticker)`` columns even for a one-symbol
    download (CLAUDE.md §9). Keeping only the field level is safe solely
    because ``_fetch_one`` requests one symbol per call: on a batched
    download every ticker's ``Close`` would flatten to the same name and
    one would silently overwrite the rest. That is a corrupted dataset
    with no error, so it is refused explicitly instead (§11).
    """
    price_level = next(
        (
            i for i in range(columns.nlevels)
            if {str(v).strip().lower() for v in columns.get_level_values(i)} & _PRICE_FIELDS
        ),
        None,
    )
    if price_level is None:
        raise IngestionError(f"{ticker}: no OHLCV field level in MultiIndex columns")

    for level in range(columns.nlevels):
        if level == price_level:
            continue
        symbols = sorted({str(v) for v in columns.get_level_values(level)})
        if len(symbols) > 1:
            raise IngestionError(
                f"{ticker}: response holds {len(symbols)} symbols {symbols}; "
                "flattening would merge their columns. Fetch one symbol per "
                "call or split the frame by ticker first."
            )

    return columns.get_level_values(price_level)


def _normalise_yf_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten a yfinance frame into the canonical column contract.

    yfinance may return flat columns or ``(field, ticker)`` MultiIndex
    columns (the latter even for one symbol, in 1.7); both are handled so
    callers see one schema. A MultiIndex holding more than one symbol is
    rejected rather than flattened (see ``_flatten_single_symbol``).
    """
    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = _flatten_single_symbol(df.columns, ticker)

    df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    rename = {"adj_close": "adj_close", "datetime": "date", "index": "date"}
    df = df.rename(columns=rename)

    if "date" not in df.columns:
        raise IngestionError(f"{ticker}: no date column in response")

    keep = ["date", "open", "high", "low", "close", "volume"]
    if "adj_close" in df.columns:
        keep.append("adj_close")

    missing = set(keep) - set(df.columns)
    if missing:
        raise IngestionError(f"{ticker}: response missing {sorted(missing)}")

    df = df[keep].copy()
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.dropna(subset=["close"]).reset_index(drop=True)


def _fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download one symbol with bounded exponential backoff (IR-01)."""
    import yfinance as yf  # imported lazily so tests need no network

    last_error: Exception | None = None

    for attempt in range(1, INGESTION.max_retries + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,   # keep adj_close so DR-03 can be applied
                progress=False,
                timeout=INGESTION.request_timeout,
            )
            if raw is None or raw.empty:
                raise IngestionError(f"{ticker}: empty response")
            return _normalise_yf_frame(raw, ticker)

        except Exception as exc:  # noqa: BLE001 - retried then re-raised
            last_error = exc
            if attempt < INGESTION.max_retries:
                delay = INGESTION.backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "fetch failed for %s (attempt %d/%d): %s - retrying in %.1fs",
                    ticker, attempt, INGESTION.max_retries, exc, delay,
                )
                time.sleep(delay)

    raise IngestionError(
        f"{ticker}: failed after {INGESTION.max_retries} attempts"
    ) from last_error


def fetch_market_data(
    start: str,
    end: str,
    tickers: tuple[str, ...] | None = None,
    strict: bool = True,
) -> IngestionResult:
    """FR-01: ingest the equity universe plus the VIX index.

    With ``strict=True`` (the default, and the setting the scheduled job
    uses) any ticker failure aborts the whole ingestion, so the caller
    never receives a partial universe it might mistake for a complete
    one. ``strict=False`` exists for exploratory work only.
    """
    tickers = tickers or DATA.tickers
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for ticker in tickers:
        try:
            frames.append(_fetch_one(ticker, start, end))
            logger.info("fetched %s", ticker)
        except IngestionError as exc:
            if strict:
                raise
            logger.error("skipping %s: %s", ticker, exc)
            failed.append(ticker)

    if not frames:
        raise IngestionError("no equity data retrieved for any ticker")

    equities = pd.concat(frames, ignore_index=True)
    vix = _fetch_one(DATA.vix_symbol, start, end)

    return IngestionResult(
        equities=equities.sort_values(["ticker", "date"]).reset_index(drop=True),
        vix=vix,
        requested_tickers=tuple(tickers),
        failed_tickers=tuple(failed),
    )
