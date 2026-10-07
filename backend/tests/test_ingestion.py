"""Tests for yfinance response normalisation (FR-01, DR-02).

Synthetic frames shaped like real yfinance 1.7 responses (CLAUDE.md §9).
No network: ``_normalise_yf_frame`` is exercised directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataops.ingestion import IngestionError, _normalise_yf_frame

FIELDS = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]


def make_raw(tickers=("XOM",), n: int = 5, ticker_first: bool = False) -> pd.DataFrame:
    """A yfinance-style frame: DatetimeIndex, MultiIndex columns."""
    dates = pd.bdate_range("2024-01-02", periods=n, name="Date")
    data, columns = {}, []
    for i, ticker in enumerate(tickers):
        close = 100.0 + 10 * i + np.arange(n, dtype=float)
        values = {
            "Adj Close": close * 0.98, "Close": close, "High": close + 1,
            "Low": close - 1, "Open": close - 0.5, "Volume": np.full(n, 1_000_000),
        }
        for field in FIELDS:
            key = (ticker, field) if ticker_first else (field, ticker)
            columns.append(key)
            data[key] = values[field]
    names = ["Ticker", "Price"] if ticker_first else ["Price", "Ticker"]
    return pd.DataFrame(data, index=dates, columns=pd.MultiIndex.from_tuples(columns, names=names))


def test_single_symbol_multiindex_is_flattened():
    """The real yfinance 1.7 shape: (field, ticker) columns for one symbol."""
    out = _normalise_yf_frame(make_raw(("XOM",)), "XOM")

    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume", "adj_close", "ticker"]
    assert (out["ticker"] == "XOM").all()
    assert out["close"].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]


def test_ticker_first_multiindex_is_flattened_by_field_level():
    """group_by='ticker' puts the symbol level first; the field level is found by name."""
    out = _normalise_yf_frame(make_raw(("CVX",), ticker_first=True), "CVX")

    assert out["close"].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert out["adj_close"].iloc[0] == pytest.approx(98.0)


def test_flat_frame_still_normalises():
    raw = make_raw(("NEE",))
    raw.columns = raw.columns.get_level_values("Price")

    out = _normalise_yf_frame(raw, "NEE")

    assert len(out) == 5
    assert out["high"].iloc[0] == pytest.approx(101.0)


@pytest.mark.parametrize("ticker_first", [False, True])
def test_batched_multiindex_is_rejected_not_merged(ticker_first):
    """Two symbols in one frame would flatten to duplicate 'Close' columns,
    one silently overwriting the other. That must fail loudly instead."""
    raw = make_raw(("XOM", "CVX"), ticker_first=ticker_first)

    with pytest.raises(IngestionError, match=r"2 symbols \['CVX', 'XOM'\]"):
        _normalise_yf_frame(raw, "XOM")
