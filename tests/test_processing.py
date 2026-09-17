"""Tests for the DataOps transformation layer.

Traceability: TC-02 (DR-04), TC-03 (DR-05), plus DR-06 and DR-07, which
have no numbered test case in Chapter 4 but are the properties whose
violation would silently invalidate every downstream result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataops import processing as P


def make_series(n: int = 200, tickers=("XOM", "CVX"), seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic OHLCV data. No network required."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    frames = []
    for i, ticker in enumerate(tickers):
        close = 100.0 + np.cumsum(rng.normal(0.05, 1.0, n)) + i * 10
        close = np.maximum(close, 1.0)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": rng.integers(1e6, 2e7, n),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------- DR-04
def test_missing_rows_are_forward_filled_and_flagged():
    """TC-02: an injected gap is filled and marked is_imputed."""
    df = make_series(60)
    gap_date = df["date"].iloc[30]
    df = df[~((df["ticker"] == "XOM") & (df["date"] == gap_date))]

    out = P.impute_missing(df)
    row = out[(out["ticker"] == "XOM") & (out["date"] == gap_date)]

    assert len(row) == 1, "gap row was not reinstated"
    assert bool(row["is_imputed"].iloc[0]) is True
    assert row["close"].notna().all()

    # The filled value must equal the prior observation, not the next one.
    prev = out[(out["ticker"] == "XOM") & (out["date"] < gap_date)].iloc[-1]
    assert row["close"].iloc[0] == pytest.approx(prev["close"])


def test_observed_rows_are_not_flagged_as_imputed():
    out = P.impute_missing(make_series(50))
    assert not out["is_imputed"].any()


def test_leading_gap_is_dropped_not_backfilled():
    """Back-filling would be forward-looking, so leading rows are dropped."""
    df = make_series(40)
    first_two = df[df["ticker"] == "XOM"]["date"].iloc[:2]
    df = df[~((df["ticker"] == "XOM") & (df["date"].isin(first_two)))]

    out = P.impute_missing(df)
    xom = out[out["ticker"] == "XOM"]
    assert xom["date"].min() > first_two.iloc[-1]


# --------------------------------------------------------------- FR-03
def test_sma_matches_manual_calculation():
    df = make_series(60, tickers=("XOM",))
    out = P.compute_indicators(df)
    expected = df["close"].iloc[:20].mean()
    assert out["sma_20"].iloc[19] == pytest.approx(expected)
    assert pd.isna(out["sma_20"].iloc[18]), "SMA must not emit before 20 periods"


def test_rsi_bounded_and_correct_on_monotonic_series():
    """A strictly rising series has no losses, so RSI is 100."""
    dates = pd.bdate_range("2020-01-01", periods=40)
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "TEST",
            "open": range(100, 140),
            "high": range(101, 141),
            "low": range(99, 139),
            "close": [float(x) for x in range(100, 140)],
            "volume": [1_000_000] * 40,
        }
    )
    out = P.compute_indicators(df)
    rsi = out["rsi_14"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()
    assert rsi.iloc[-1] == pytest.approx(100.0)


# --------------------------------------------------------------- DR-07
def test_zscore_uses_only_backward_looking_window():
    """The leakage test.

    Mutating a future observation must not change any past standardised
    value. If rolling() were centred or forward-shifted, this fails.
    """
    df = P.compute_indicators(P.impute_missing(make_series(150, tickers=("XOM",))))

    base = P.normalize_rolling(df, columns=["close"], window=20)

    tampered = df.copy()
    tampered.loc[tampered.index[-1], "close"] *= 10.0  # extreme future shock
    after = P.normalize_rolling(tampered, columns=["close"], window=20)

    # Every value except the final one must be unchanged.
    pd.testing.assert_series_equal(
        base["close_z"].iloc[:-1],
        after["close_z"].iloc[:-1],
        check_names=False,
    )


def test_zscore_of_constant_series_is_zero_not_nan():
    dates = pd.bdate_range("2020-01-01", periods=40)
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "FLAT",
            "open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0,
            "volume": 1_000_000,
        }
    )
    out = P.normalize_rolling(df, columns=["close"], window=10)
    tail = out["close_z"].dropna()
    assert len(tail) > 0
    assert (tail == 0.0).all()


# --------------------------------------------------------------- DR-06
def test_partition_rejects_overlapping_windows():
    df = make_series(300)
    with pytest.raises(ValueError, match="look-ahead leakage"):
        P.partition_chronological(
            df, "2020-01-01", "2020-06-30", "2020-06-01", "2020-12-31"
        )


def test_partition_produces_disjoint_ordered_sets():
    df = make_series(400)
    train, evaluation = P.partition_chronological(
        df, "2020-01-01", "2020-12-31", "2021-01-01", "2021-06-30"
    )
    assert not train.empty and not evaluation.empty
    assert train["date"].max() < evaluation["date"].min()
    assert set(train["date"]).isdisjoint(set(evaluation["date"]))


# --------------------------------------------------------------- walk-forward
def test_walk_forward_splits_are_chronological_and_disjoint():
    splits = P.walk_forward_splits(
        "2015-01-01", "2020-01-01", train_days=365, eval_days=90
    )
    assert len(splits) > 1  # more than one regime, not a single split

    for split in splits:
        assert split["eval_start"] > split["train_end"]  # DR-06, by construction
        assert split["train_end"] > split["train_start"]
        assert split["eval_end"] > split["eval_start"]

    for a, b in zip(splits, splits[1:]):
        assert b["train_start"] > a["train_start"]  # each window rolls forward


def test_walk_forward_default_step_gives_non_overlapping_eval_windows():
    splits = P.walk_forward_splits(
        "2015-01-01", "2018-01-01", train_days=365, eval_days=180
    )
    for a, b in zip(splits, splits[1:]):
        assert b["eval_start"] > a["eval_end"]  # no evaluation day scored twice


def test_walk_forward_splits_never_exceed_the_end_date():
    end = pd.Timestamp("2016-06-30")
    splits = P.walk_forward_splits("2015-01-01", "2016-06-30", train_days=365, eval_days=90)
    for split in splits:
        assert split["eval_end"] <= end


def test_walk_forward_splits_empty_when_range_too_short():
    splits = P.walk_forward_splits("2015-01-01", "2015-06-01", train_days=365, eval_days=90)
    assert splits == []


# --------------------------------------------------------------- DR-03
def test_corporate_action_adjustment_preserves_ratios():
    """A 2:1 split halves price and doubles volume, leaving value intact."""
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=2),
            "ticker": "SPLIT",
            "open": [100.0, 100.0],
            "high": [100.0, 100.0],
            "low": [100.0, 100.0],
            "close": [100.0, 100.0],
            "adj_close": [50.0, 100.0],
            "volume": [1_000_000, 1_000_000],
        }
    )
    out = P.adjust_corporate_actions(df)
    assert out["close"].iloc[0] == pytest.approx(50.0)
    assert out["volume"].iloc[0] == pytest.approx(2_000_000)
    assert out["close"].iloc[1] == pytest.approx(100.0)
    assert "adj_close" not in out.columns


# --------------------------------------------------------------- VIX
def test_vix_is_joined_on_date_across_all_tickers():
    df = P.impute_missing(make_series(30))
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    vix = pd.DataFrame({"date": dates, "close": np.linspace(15, 30, len(dates))})

    out = P.attach_vix(df, vix)
    assert out["vix"].notna().all()

    # The same date carries the same VIX for every ticker.
    per_date = out.groupby("date")["vix"].nunique()
    assert (per_date == 1).all()


# --------------------------------------------------------------- pipeline
def test_full_pipeline_produces_expected_columns():
    equities = make_series(200)
    dates = pd.DatetimeIndex(sorted(equities["date"].unique()))
    vix = pd.DataFrame({"date": dates, "close": np.linspace(12, 40, len(dates))})

    out = P.build_feature_frame(equities, vix)

    for col in ("sma_20", "rsi_14", "vix", "is_imputed", "close_z", "vix_z"):
        assert col in out.columns, f"missing {col}"
    assert out["close_z"].notna().any()


def test_validation_rejects_malformed_frame():
    with pytest.raises(ValueError, match="missing required columns"):
        P.compute_indicators(pd.DataFrame({"date": [], "ticker": []}))
