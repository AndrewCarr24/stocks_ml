import numpy as np
import pandas as pd
import pytest

from stocks_ml.config import Config
from stocks_ml.data.store import DataStore

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
START, PERIODS = "2020-01-02", 720  # ~2.9 years of business days


def make_prices(seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, periods=PERIODS)
    rows = []
    for i, t in enumerate([*TICKERS, "SPY"]):
        drift = 0.0002 + 0.0001 * i
        rets = rng.normal(drift, 0.015, len(dates))
        close = 50.0 * np.cumprod(1 + rets)
        open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
        for d, o, c in zip(dates, open_, close):
            rows.append({"date": d, "ticker": t, "open": o, "high": max(o, c),
                         "low": min(o, c), "close": c, "volume": 1e6 + rng.integers(0, 1e5)})
    return pd.DataFrame(rows)


def make_membership():
    return pd.DataFrame({
        "ticker": TICKERS,
        "start_date": [pd.Timestamp("2015-01-01")] * 7 + [pd.Timestamp("2021-06-01")],  # HHH joins late
        "end_date": [pd.NaT] * 6 + [pd.Timestamp("2021-06-01"), pd.NaT],                # GGG leaves
        "sector": ["Tech", "Tech", "Energy", "Energy", "Health", "Health", "Retail", "Retail"],
    })


def make_fred():
    dates = pd.date_range("2019-12-01", "2023-01-31", freq="D")
    return pd.concat([
        pd.DataFrame({"date": dates, "series": "VIXCLS", "value": 20.0 + 5 * np.sin(np.arange(len(dates)) / 30)}),
        pd.DataFrame({"date": dates, "series": "DTB3", "value": 2.0}),
    ], ignore_index=True)


def make_edgar():
    rows = []
    for t in TICKERS[:4]:  # fundamentals for half the universe (tests NaN handling)
        for fy, filed in [("2019", "2020-02-15"), ("2020", "2021-02-15"), ("2021", "2022-02-15")]:
            y = int(fy)
            rows += [
                (t, "net_income", f"{fy}-01-01", f"{fy}-12-31", filed, 100.0 + 10 * (y - 2019), "10-K"),
                (t, "revenues", f"{fy}-01-01", f"{fy}-12-31", filed, 500.0, "10-K"),
                (t, "gross_profit", f"{fy}-01-01", f"{fy}-12-31", filed, 250.0, "10-K"),
                (t, "ocf", f"{fy}-01-01", f"{fy}-12-31", filed, 120.0, "10-K"),
                (t, "assets", None, f"{fy}-12-31", filed, 1000.0 * (1.1 ** (y - 2019)), "10-K"),
                (t, "equity", None, f"{fy}-12-31", filed, 400.0, "10-K"),
                (t, "liabilities", None, f"{fy}-12-31", filed, 600.0, "10-K"),
                (t, "shares", None, f"{fy}-12-31", filed, 100.0, "10-K"),
            ]
    df = pd.DataFrame(rows, columns=["ticker", "concept", "start", "end", "filed", "val", "form"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


def make_form4():
    # Dated well after test_no_lookahead.py's CUTOFF (2022-06-30, with price
    # history in this fixture ending 2022-10-05) so the no-lookahead suite's
    # before-cutoff equality checks see the same all-zero "no activity"
    # defaults from both the real and future-corrupted stores (that helper
    # doesn't forward form4/shortint, so pre-cutoff parity depends on no real
    # transaction being visible that early).
    rows = [
        ("AAA", "2022-08-05", "2022-08-03", "P", 1000.0, 25000.0),
        ("AAA", "2022-08-19", "2022-08-17", "P", 500.0, 13000.0),
        ("BBB", "2022-09-02", "2022-08-30", "S", 2000.0, 60000.0),
    ]
    df = pd.DataFrame(rows, columns=["ticker", "filed", "trans_date", "code", "shares", "value"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["trans_date"] = pd.to_datetime(df["trans_date"])
    return df


def make_shortint():
    # publication_date also kept after CUTOFF for the same reason as above.
    rows = [
        ("AAA", "2022-08-01", "2022-08-15", 500_000.0),
        ("BBB", "2022-08-17", "2022-08-31", 300_000.0),
    ]
    df = pd.DataFrame(rows, columns=["ticker", "settlement_date", "publication_date",
                                     "short_interest"])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df["publication_date"] = pd.to_datetime(df["publication_date"])
    return df


@pytest.fixture
def synthetic_store(tmp_path):
    store = DataStore(tmp_path / "data")
    store.write("prices", make_prices())
    store.write("membership", make_membership())
    store.write("fred", make_fred())
    store.write("edgar", make_edgar())
    store.write("form4", make_form4())
    store.write("shortint", make_shortint())
    return store


@pytest.fixture
def tiny_cfg(tmp_path):
    return Config(
        data_dir=tmp_path / "data", user_agent="test", horizon_days=5, purge_days=10,
        rebalance_weekday=4, retrain_weeks=4, backtest_start=pd.Timestamp("2021-01-04"),
        cv_train_years=2, train_sample_rows=None,
        fred_series={"VIXCLS": 1, "DTB3": 1},
        edgar_concepts={},
    )
