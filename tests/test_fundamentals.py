import numpy as np
import pandas as pd

from stocks_ml.features.fundamentals import fundamental_features


def _edgar():
    rows = [
        # ticker, concept, start, end, filed, val, form
        ("AAA", "net_income", "2022-01-01", "2022-12-31", "2023-02-15", 100.0, "10-K"),
        ("AAA", "net_income", "2021-01-01", "2021-12-31", "2022-02-15", 80.0, "10-K"),
        ("AAA", "net_income", "2023-01-01", "2023-03-31", "2023-05-01", 30.0, "10-Q"),  # quarterly: ignored
        ("AAA", "assets", None, "2022-12-31", "2023-02-15", 1000.0, "10-K"),
        ("AAA", "assets", None, "2021-12-31", "2022-02-15", 800.0, "10-K"),
        ("AAA", "assets", None, "2023-03-31", "2023-05-01", 1100.0, "10-Q"),
        ("AAA", "equity", None, "2022-12-31", "2023-02-15", 500.0, "10-K"),
        ("AAA", "liabilities", None, "2022-12-31", "2023-02-15", 500.0, "10-K"),
        ("AAA", "shares", None, "2022-12-31", "2023-02-15", 50.0, "10-K"),
        ("AAA", "revenues", "2022-01-01", "2022-12-31", "2023-02-15", 400.0, "10-K"),
        ("AAA", "gross_profit", "2022-01-01", "2022-12-31", "2023-02-15", 200.0, "10-K"),
        ("AAA", "ocf", "2022-01-01", "2022-12-31", "2023-02-15", 150.0, "10-K"),
    ]
    df = pd.DataFrame(rows, columns=["ticker", "concept", "start", "end", "filed", "val", "form"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


def _base(date, close=20.0):
    return pd.DataFrame({"date": [pd.Timestamp(date)], "ticker": ["AAA"], "close": [close]})


def test_point_in_time_uses_only_filed_facts():
    out = fundamental_features(_edgar(), _base("2023-01-31"))
    # 2022 10-K filed 2023-02-15 is NOT yet visible on 2023-01-31; the fixture's
    # equity/shares facts only exist in that filing -> ratios must be NaN
    assert out.iloc[0][["f_earnings_yield", "f_roe", "f_book_to_market"]].isna().all()
    # after filing, 2022 values apply
    out2 = fundamental_features(_edgar(), _base("2023-03-01"))
    row = out2.iloc[0]
    mktcap = 50.0 * 20.0  # shares * close = 1000
    assert np.isclose(row["f_earnings_yield"], 100.0 / mktcap)
    assert np.isclose(row["f_book_to_market"], 500.0 / mktcap)
    assert np.isclose(row["f_sales_to_price"], 400.0 / mktcap)
    assert np.isclose(row["f_cf_to_price"], 150.0 / mktcap)
    assert np.isclose(row["f_roe"], 100.0 / 500.0)
    assert np.isclose(row["f_gross_profitability"], 200.0 / 1000.0)
    assert np.isclose(row["f_ocf_to_assets"], 150.0 / 1000.0)
    assert np.isclose(row["f_leverage"], 500.0 / 1000.0)
    assert np.isclose(row["f_log_mktcap"], np.log(mktcap))


def test_quarterly_balance_sheet_updates_instants():
    out = fundamental_features(_edgar(), _base("2023-06-01"))
    # 10-Q filed 2023-05-01 updates assets to 1100
    assert np.isclose(out.iloc[0]["f_gross_profitability"], 200.0 / 1100.0)


def test_asset_growth_from_consecutive_annuals():
    out = fundamental_features(_edgar(), _base("2023-03-01"))
    assert np.isclose(out.iloc[0]["f_asset_growth"], 1000.0 / 800.0 - 1.0)


def test_missing_ticker_yields_nans():
    base = pd.DataFrame({"date": [pd.Timestamp("2023-03-01")], "ticker": ["ZZZ"], "close": [5.0]})
    out = fundamental_features(_edgar(), base)
    assert out.iloc[0][["f_roe", "f_earnings_yield", "f_asset_growth"]].isna().all()
