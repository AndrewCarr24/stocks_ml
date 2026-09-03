import numpy as np
import pandas as pd

from stocks_ml.features.insiders import _windowed_asof_sum, insider_features


# ---- insider_features: PIT, net-buy arithmetic, buyers, evt flag ----------

def _dollar_volume(tickers=("AAA",), start="2023-01-02", periods=260, value=1_000_000.0):
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(value, index=dates, columns=list(tickers))


def _form4_rows(rows):
    """rows: list of (ticker, filed, trans_date, code, shares, value)."""
    df = pd.DataFrame(rows, columns=["ticker", "filed", "trans_date", "code", "shares", "value"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["trans_date"] = pd.to_datetime(df["trans_date"])
    return df


def test_insider_features_date_only_filing_is_available_next_day():
    dv = _dollar_volume()
    t = pd.Timestamp("2023-06-01")
    form4 = _form4_rows([
        ("AAA", t, t, "P", 100.0, 1000.0),                       # filed == t: counts
        ("AAA", t + pd.Timedelta(days=1), t, "P", 500.0, 5000.0),  # filed == t+1: must NOT count
    ])
    feats = insider_features(form4, pd.DatetimeIndex([t]), dv)
    row = feats[(feats.date == t) & (feats.ticker == "AAA")].iloc[0]
    assert row["f_insider_net_13w"] == 0.0
    next_day = t + pd.Timedelta(days=1)
    visible = insider_features(form4, pd.DatetimeIndex([next_day]), dv)
    visible_row = visible[(visible.date == next_day) & (visible.ticker == "AAA")].iloc[0]
    assert np.isclose(visible_row["f_insider_net_13w"], 1000.0 / dv.loc[next_day, "AAA"])


def test_insider_net_13w_hand_built_buys_and_sells():
    dv = _dollar_volume()
    t = pd.Timestamp("2023-06-01")
    form4 = _form4_rows([
        ("AAA", t - pd.Timedelta(days=5), t - pd.Timedelta(days=5), "P", 100.0, 2000.0),
        ("AAA", t - pd.Timedelta(days=3), t - pd.Timedelta(days=3), "S", 50.0, 500.0),
        ("AAA", t - pd.Timedelta(days=100), t - pd.Timedelta(days=100), "P", 999.0, 99999.0),  # outside 91d window
    ])
    feats = insider_features(form4, pd.DatetimeIndex([t]), dv)
    row = feats[(feats.date == t) & (feats.ticker == "AAA")].iloc[0]
    expected = (2000.0 - 500.0) / dv.loc[t, "AAA"]
    assert np.isclose(row["f_insider_net_13w"], expected)


def test_insider_buyers_13w_counts_distinct_p_filings():
    dv = _dollar_volume()
    t = pd.Timestamp("2023-06-01")
    form4 = _form4_rows([
        ("AAA", t - pd.Timedelta(days=10), t, "P", 100.0, 1000.0),
        ("AAA", t - pd.Timedelta(days=5), t, "P", 200.0, 2000.0),   # 2nd distinct filed date
        ("AAA", t - pd.Timedelta(days=2), t, "S", 50.0, 500.0),      # sale: not a buyer filing
        ("AAA", t - pd.Timedelta(days=200), t, "P", 300.0, 3000.0),  # too old
    ])
    feats = insider_features(form4, pd.DatetimeIndex([t]), dv)
    row = feats[(feats.date == t) & (feats.ticker == "AAA")].iloc[0]
    assert row["f_insider_buyers_13w"] == 2.0


def test_evt_insider_buy_2w_on_at_10_trading_days_off_at_11():
    dv = _dollar_volume()
    cal = dv.index
    F = cal[50]
    form4 = _form4_rows([("AAA", F, F, "P", 100.0, 1000.0)])
    t_on, t_off = cal[61], cal[62]  # conservative next-day availability shifts the window
    feats = insider_features(form4, pd.DatetimeIndex([t_on, t_off]), dv)
    row_on = feats[(feats.date == t_on) & (feats.ticker == "AAA")].iloc[0]
    row_off = feats[(feats.date == t_off) & (feats.ticker == "AAA")].iloc[0]
    assert row_on["f_evt_insider_buy_2w"] == 1.0
    assert row_off["f_evt_insider_buy_2w"] == 0.0


def test_windowed_asof_sum_respects_a_non_default_grid_index():
    """Regression: the helper must key off `grid.index` itself (not assume a
    fresh 0..n-1 RangeIndex), since callers may hand it an arbitrary index."""
    t = pd.Timestamp("2023-06-01")
    events = pd.DataFrame({"ticker": ["AAA"], "filed": [t - pd.Timedelta(days=5)], "val": [10.0]})
    grid = pd.DataFrame({"date": [t, t], "ticker": ["AAA", "BBB"]}, index=[5, 9])
    out = _windowed_asof_sum(events, "val", grid, 91)
    assert out.loc[5] == 10.0
    assert out.loc[9] == 0.0


def test_insider_features_survives_unsorted_multi_ticker_input():
    """Regression companion to the short_features fix: real form4 data also
    arrives as many unsorted tickers. Use tickers whose filed dates run
    OPPOSITE their alphabetical order and shuffle the row order, matching
    real ingestion -- insider_features already sorts by the "on" key alone
    at each merge_asof site, so this should already pass; pinned here so a
    future regression in that convention is caught immediately.
    """
    tickers = ["ZZZ", "YYY", "XXX", "AAA", "BBB", "CCC"]
    dv = _dollar_volume(tickers=tickers, periods=260)
    cal = dv.index
    t_eval = cal[220]

    rows = []
    for i, tkr in enumerate(tickers):
        F = cal[200 - i * 30]  # later alphabetical ticker -> EARLIER filed date
        rows.append((tkr, F, F, "P", 100.0, 1000.0 + i))
        rows.append((tkr, F - pd.Timedelta(days=10), F - pd.Timedelta(days=10), "S", 50.0, 500.0))
    form4 = _form4_rows(rows).sample(frac=1, random_state=0).reset_index(drop=True)

    feats = insider_features(form4, pd.DatetimeIndex([t_eval]), dv)
    assert len(feats) == len(tickers)
    assert feats["f_insider_net_13w"].notna().all()


def test_insider_features_empty_form4_matches_no_activity_defaults():
    dv = _dollar_volume()
    t = pd.Timestamp("2023-06-01")
    empty = _form4_rows([])
    feats = insider_features(empty, pd.DatetimeIndex([t]), dv)
    row = feats[(feats.date == t) & (feats.ticker == "AAA")].iloc[0]
    assert row["f_insider_net_13w"] == 0.0
    assert row["f_insider_buyers_13w"] == 0.0
    assert row["f_evt_insider_buy_2w"] == 0.0
