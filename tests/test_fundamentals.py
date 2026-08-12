import numpy as np
import pandas as pd
import pytest

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


def test_asset_growth_waits_until_both_periods_are_filed():
    edgar = _edgar()
    prior = (edgar["concept"].eq("assets") & edgar["end"].eq(pd.Timestamp("2021-12-31")))
    edgar.loc[prior, "filed"] = pd.Timestamp("2023-04-01")
    assert np.isnan(fundamental_features(edgar, _base("2023-03-01")).iloc[0]["f_asset_growth"])
    assert np.isclose(fundamental_features(edgar, _base("2023-04-03")).iloc[0]["f_asset_growth"],
                      1000.0 / 800.0 - 1.0)


def test_date_only_filing_is_not_visible_at_same_day_close():
    assert fundamental_features(_edgar(), _base("2023-02-15")).iloc[0][
        ["f_earnings_yield", "f_roe"]].isna().all()


def test_missing_ticker_yields_nans():
    base = pd.DataFrame({"date": [pd.Timestamp("2023-03-01")], "ticker": ["ZZZ"], "close": [5.0]})
    out = fundamental_features(_edgar(), base)
    assert out.iloc[0][["f_roe", "f_earnings_yield", "f_asset_growth"]].isna().all()


def test_sue_seasonal_surprise_and_nincr():
    from stocks_ml.features.fundamentals import earnings_quality_features

    rows = []
    # 10 quarters of AAA net income: rising, with known seasonal diffs
    for i in range(10):
        end = pd.Timestamp("2020-03-31") + pd.DateOffset(months=3 * i)
        rows.append(["AAA", "net_income", end - pd.DateOffset(days=89), end,
                     end + pd.Timedelta(days=30), 100.0 + 10 * i, "10-Q"])
    edgar = pd.DataFrame(rows, columns=["ticker", "concept", "start", "end",
                                        "filed", "val", "form"])
    base = pd.DataFrame({"date": [pd.Timestamp("2023-01-06")],
                         "ticker": ["AAA"], "close": [50.0]})
    out = earnings_quality_features(edgar, base)
    # seasonal diff is constant (+40): sigma -> 0 -> SUE undefined there; but
    # nincr must count the run of positive seasonal surprises (capped at 8)
    assert out["f_nincr"].iloc[0] >= 4
    # constant surprises give zero sigma -> f_sue stays NaN, never inf
    assert not np.isinf(out["f_sue"]).any()


def test_sue_positive_for_accelerating_earnings():
    from stocks_ml.features.fundamentals import earnings_quality_features

    rng = np.random.default_rng(0)
    rows = []
    vals = []
    for i in range(16):
        # seasonal growth with noise so sigma > 0; last surprise strongly positive
        v = 100 + 12 * i + rng.normal(0, 3)
        vals.append(v)
        end = pd.Timestamp("2019-03-31") + pd.DateOffset(months=3 * i)
        rows.append(["AAA", "net_income", end - pd.DateOffset(days=89), end,
                     end + pd.Timedelta(days=30), v, "10-Q"])
    edgar = pd.DataFrame(rows, columns=["ticker", "concept", "start", "end",
                                        "filed", "val", "form"])
    base = pd.DataFrame({"date": [pd.Timestamp("2023-06-02")],
                         "ticker": ["AAA"], "close": [50.0]})
    out = earnings_quality_features(edgar, base)
    assert np.isfinite(out["f_sue"].iloc[0])
    assert out["f_sue"].iloc[0] > 0          # growing seasonal earnings


def test_net_issuance_yoy_share_change():
    from stocks_ml.features.fundamentals import earnings_quality_features

    rows = []
    for i, (end, val) in enumerate([("2021-12-31", 1000.0), ("2022-12-31", 1100.0)]):
        e = pd.Timestamp(end)
        rows.append(["AAA", "shares", e, e, e + pd.Timedelta(days=45), val, "10-K"])
    edgar = pd.DataFrame(rows, columns=["ticker", "concept", "start", "end",
                                        "filed", "val", "form"])
    base = pd.DataFrame({"date": [pd.Timestamp("2023-06-02")],
                         "ticker": ["AAA"], "close": [50.0]})
    out = earnings_quality_features(edgar, base)
    assert out["f_net_issuance"].iloc[0] == pytest.approx(0.10)   # +10% shares
