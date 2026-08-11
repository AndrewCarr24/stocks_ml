import numpy as np
import pandas as pd

from stocks_ml.data.insiders import extract_transactions, ingest_form4
from stocks_ml.data.store import DataStore
from stocks_ml.features.insiders import _windowed_asof_sum, insider_features


def _submission(rows):
    """rows: list of (accession, filing_date 'DD-MON-YYYY', issuer_cik str)."""
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK"])


def _trans(rows):
    """rows: list of (accession, trans_date 'DD-MON-YYYY', code, shares, price)."""
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
                                       "TRANS_SHARES", "TRANS_PRICEPERSHARE"])


# ---- extract_transactions -------------------------------------------------

def test_extract_transactions_keeps_only_open_market_codes():
    sub = _submission([
        ("A1", "01-MAR-2023", "0000000001"),
        ("A2", "02-MAR-2023", "0000000001"),
        ("A3", "03-MAR-2023", "0000000001"),
        ("A4", "04-MAR-2023", "0000000001"),
        ("A5", "05-MAR-2023", "0000000001"),
    ])
    trans = _trans([
        ("A1", "01-MAR-2023", "P", "100", "10.0"),   # keep: open-market purchase
        ("A2", "02-MAR-2023", "S", "50", "12.0"),    # keep: open-market sale
        ("A3", "03-MAR-2023", "M", "200", "1.0"),    # drop: option exercise
        ("A4", "04-MAR-2023", "A", "300", "0.0"),    # drop: grant/award
        ("A5", "05-MAR-2023", "G", "10", "0.0"),     # drop: gift
    ])
    out = extract_transactions(sub, trans, {1: "AAA"})
    assert sorted(out["code"].unique()) == ["P", "S"]
    assert len(out) == 2


def test_extract_transactions_output_contract_and_value_arithmetic():
    sub = _submission([("A1", "15-JUN-2023", "0000000042")])
    trans = _trans([("A1", "10-JUN-2023", "P", "1000", "25.5")])
    out = extract_transactions(sub, trans, {42: "ZZZ"})
    assert list(out.columns) == ["ticker", "filed", "trans_date", "code", "shares", "value"]
    row = out.iloc[0]
    assert row["ticker"] == "ZZZ"
    assert row["filed"] == pd.Timestamp("2023-06-15")
    assert row["trans_date"] == pd.Timestamp("2023-06-10")
    assert row["shares"] == 1000.0
    assert np.isclose(row["value"], 1000.0 * 25.5)


def test_extract_transactions_value_is_nan_safe_when_price_missing():
    sub = _submission([("A1", "15-JUN-2023", "0000000042")])
    trans = _trans([("A1", "10-JUN-2023", "P", "1000", "")])  # blank price
    out = extract_transactions(sub, trans, {42: "ZZZ"})
    assert out.iloc[0]["shares"] == 1000.0
    assert np.isnan(out.iloc[0]["value"])


def test_extract_transactions_drops_unmappable_ciks():
    sub = _submission([
        ("A1", "01-MAR-2023", "0000000001"),  # mappable
        ("A2", "02-MAR-2023", "0000099999"),  # NOT in cik_to_ticker
    ])
    trans = _trans([
        ("A1", "01-MAR-2023", "P", "100", "10.0"),
        ("A2", "02-MAR-2023", "P", "100", "10.0"),
    ])
    out = extract_transactions(sub, trans, {1: "AAA"})
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "AAA"


def test_extract_transactions_empty_inputs():
    out = extract_transactions(_submission([]), _trans([]), {})
    assert out.empty
    assert list(out.columns) == ["ticker", "filed", "trans_date", "code", "shares", "value"]


# ---- ingest_form4 -----------------------------------------------------------

def _fake_quarter(ticker="AAA", cik="0000000001", filed="15-MAY-2023", code="P"):
    sub = _submission([("ACC", filed, cik)])
    trans = _trans([("ACC", filed, code, "100", "10.0")])
    return {"SUBMISSION": sub, "NONDERIV_TRANS": trans}


def test_ingest_form4_records_manifest_and_failed_quarters(tmp_path):
    store = DataStore(tmp_path)
    calls = []

    def fetch(year, q, ua):
        calls.append((year, q))
        if (year, q) == (2006, 1):
            return _fake_quarter()
        raise RuntimeError("404 not published")

    summary = ingest_form4(store, "ua", fetch_fn=fetch, cik_to_ticker={1: "AAA"})
    assert [2006, 1] in summary["quarters"]
    assert len(summary["failed_quarters"]) > 0
    assert summary["n_rows"] == 1
    df = store.read("form4")
    assert df.iloc[0]["ticker"] == "AAA"


def test_ingest_form4_skips_done_quarters_but_always_refetches_latest(tmp_path):
    store = DataStore(tmp_path)
    calls = []

    def fetch(year, q, ua):
        calls.append((year, q))
        return _fake_quarter(filed=f"15-{['JAN','APR','JUL','OCT'][q-1]}-{year}")

    s1 = ingest_form4(store, "ua", fetch_fn=fetch, cik_to_ticker={1: "AAA"})
    n_first_pass = len(calls)
    assert n_first_pass > 1  # spans many quarters from 2006Q1 onward

    calls.clear()
    s2 = ingest_form4(store, "ua", fetch_fn=fetch, cik_to_ticker={1: "AAA"})
    # only the latest quarter should be refetched on the second pass
    assert len(calls) == 1
    latest = tuple(s1["quarters"][-1])
    assert calls[0] == latest


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
