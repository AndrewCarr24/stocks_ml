import numpy as np
import pandas as pd

from stocks_ml.data.shortint import extract_shortint, ingest_shortint
from stocks_ml.data.store import DataStore
from stocks_ml.features.insiders import short_features


def _raw(rows):
    """rows: list of (symbolCode, settlementDate 'YYYY-MM-DD', currentShortPositionQuantity)."""
    return pd.DataFrame(rows, columns=["symbolCode", "settlementDate",
                                       "currentShortPositionQuantity"])


# ---- extract_shortint -------------------------------------------------------

def test_extract_shortint_shape_and_publication_lag():
    raw = _raw([("aaa", "2023-06-15", 50000)])  # lowercase symbol, normalize_symbol upper-cases
    out = extract_shortint(raw)
    assert list(out.columns) == ["ticker", "settlement_date", "publication_date", "short_interest"]
    row = out.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["settlement_date"] == pd.Timestamp("2023-06-15")
    assert row["publication_date"] == pd.Timestamp("2023-06-15") + pd.Timedelta(days=14)
    assert row["short_interest"] == 50000.0


def test_extract_shortint_empty_input():
    out = extract_shortint(pd.DataFrame(columns=["symbolCode", "settlementDate",
                                                 "currentShortPositionQuantity"]))
    assert out.empty
    assert list(out.columns) == ["ticker", "settlement_date", "publication_date", "short_interest"]


# ---- ingest_shortint ---------------------------------------------------------

def test_ingest_shortint_writes_and_records_manifest(tmp_path):
    store = DataStore(tmp_path)

    def fetch(ua, start, end):
        return _raw([("AAA", "2023-06-15", 50000), ("BBB", "2023-06-15", 20000)])

    summary = ingest_shortint(store, "ua", fetch_fn=fetch)
    assert summary["n_rows"] == 2
    assert summary["coverage_start"] == "2023-06-15"
    df = store.read("shortint")
    assert set(df["ticker"]) == {"AAA", "BBB"}


def test_ingest_shortint_incremental_only_fetches_new_window(tmp_path):
    store = DataStore(tmp_path)
    calls = []

    def fetch(ua, start, end):
        calls.append((start, end))
        return _raw([("AAA", "2023-06-15", 50000)])

    ingest_shortint(store, "ua", fetch_fn=fetch)
    first_start = calls[0][0]

    calls.clear()
    ingest_shortint(store, "ua", fetch_fn=fetch)
    second_start = calls[0][0]
    assert second_start > first_start
    assert second_start == pd.Timestamp("2023-06-16")


def test_ingest_shortint_records_failure_without_corrupting_store(tmp_path):
    store = DataStore(tmp_path)

    def good_fetch(ua, start, end):
        return _raw([("AAA", "2023-06-15", 50000)])

    ingest_shortint(store, "ua", fetch_fn=good_fetch)

    def bad_fetch(ua, start, end):
        raise RuntimeError("network down")

    summary = ingest_shortint(store, "ua", fetch_fn=bad_fetch)
    assert summary["fetch_failed"] is True
    df = store.read("shortint")
    assert set(df["ticker"]) == {"AAA"}  # untouched, not wiped


# ---- short_features: ratio/dtc arithmetic, publication-lag PIT ------------

def _shares_outstanding(rows):
    """rows: list of (date, ticker, shares)."""
    df = pd.DataFrame(rows, columns=["date", "ticker", "shares"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _volume(tickers=("AAA",), start="2023-01-02", periods=260, value=200_000.0):
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(value, index=dates, columns=list(tickers))


def _shortint_rows(rows):
    """rows: list of (ticker, settlement_date, publication_date, short_interest)."""
    df = pd.DataFrame(rows, columns=["ticker", "settlement_date", "publication_date",
                                     "short_interest"])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df["publication_date"] = pd.to_datetime(df["publication_date"])
    return df


def test_short_ratio_and_days_to_cover_arithmetic():
    t = pd.Timestamp("2023-06-01")
    shares_out = _shares_outstanding([(t, "AAA", 1_000_000.0)])
    volume = _volume(value=200_000.0)
    shortint = _shortint_rows([
        ("AAA", t - pd.Timedelta(days=20), t - pd.Timedelta(days=1), 300_000.0),
    ])
    feats = short_features(shortint, shares_out, volume)
    row = feats.iloc[0]
    assert np.isclose(row["f_short_ratio"], 300_000.0 / 1_000_000.0)
    assert np.isclose(row["f_short_dtc"], 300_000.0 / 200_000.0)


def test_short_features_publication_lag_honored():
    """settlement before t but publication AFTER t -> NaN at t."""
    t = pd.Timestamp("2023-06-01")
    shares_out = _shares_outstanding([(t, "AAA", 1_000_000.0)])
    volume = _volume()
    shortint = _shortint_rows([
        ("AAA", t - pd.Timedelta(days=20), t + pd.Timedelta(days=1), 300_000.0),  # published after t
    ])
    feats = short_features(shortint, shares_out, volume)
    row = feats.iloc[0]
    assert np.isnan(row["f_short_ratio"])
    assert np.isnan(row["f_short_dtc"])


def test_short_features_publication_on_t_is_visible():
    t = pd.Timestamp("2023-06-01")
    shares_out = _shares_outstanding([(t, "AAA", 1_000_000.0)])
    volume = _volume()
    shortint = _shortint_rows([
        ("AAA", t - pd.Timedelta(days=20), t, 300_000.0),  # published exactly at t
    ])
    feats = short_features(shortint, shares_out, volume)
    row = feats.iloc[0]
    assert np.isclose(row["f_short_ratio"], 0.3)


def test_short_features_latest_publication_used():
    t = pd.Timestamp("2023-06-01")
    shares_out = _shares_outstanding([(t, "AAA", 1_000_000.0)])
    volume = _volume()
    shortint = _shortint_rows([
        ("AAA", t - pd.Timedelta(days=40), t - pd.Timedelta(days=20), 100_000.0),
        ("AAA", t - pd.Timedelta(days=20), t - pd.Timedelta(days=2), 400_000.0),  # most recent visible
    ])
    feats = short_features(shortint, shares_out, volume)
    row = feats.iloc[0]
    assert np.isclose(row["f_short_ratio"], 0.4)


def test_short_features_empty_shortint_is_nan():
    t = pd.Timestamp("2023-06-01")
    shares_out = _shares_outstanding([(t, "AAA", 1_000_000.0)])
    volume = _volume()
    empty = _shortint_rows([])
    feats = short_features(empty, shares_out, volume)
    assert np.isnan(feats.iloc[0]["f_short_ratio"])
    assert np.isnan(feats.iloc[0]["f_short_dtc"])
