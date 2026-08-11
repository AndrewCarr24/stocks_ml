import numpy as np
import pandas as pd

from stocks_ml.features.panel import (
    POINT_IN_TIME_MACRO_SERIES, calendar_features, feature_cols, make_labels, market_macro_features,
    price_features, rebalance_dates, sector_relative_momentum, trading_calendar,
)


def _prices(growth_by_ticker, start="2022-01-03", periods=300):
    """Deterministic prices: each ticker compounds at a fixed daily rate."""
    dates = pd.bdate_range(start, periods=periods)
    rows = []
    for ticker, g in growth_by_ticker.items():
        px = 100.0 * np.cumprod(np.full(len(dates), 1 + g))
        for d, p in zip(dates, px):
            rows.append({"date": d, "ticker": ticker, "open": p, "high": p,
                         "low": p, "close": p, "volume": 1e6})
    return pd.DataFrame(rows)


def test_rebalance_dates_are_fridays_or_prior_trading_day():
    prices = _prices({"AAA": 0.001})
    cal = trading_calendar(prices)
    # remove Friday 2022-01-21 to simulate a holiday
    cal = cal[cal != pd.Timestamp("2022-01-21")]
    rdates = rebalance_dates(cal, "2022-01-03", "2022-02-28")
    assert pd.Timestamp("2022-01-14") in rdates          # normal Friday
    assert pd.Timestamp("2022-01-20") in rdates          # Thursday before holiday Friday
    assert all(d in set(cal) for d in rdates)


def test_momentum_matches_known_growth():
    prices = _prices({"AAA": 0.01})
    cal = trading_calendar(prices)
    rdates = rebalance_dates(cal, "2022-06-01", "2023-01-31")
    feats = price_features(prices, rdates)
    row = feats[feats.date == rdates[-1]].iloc[0]
    assert np.isclose(row["f_mom_4w"], 1.01 ** 20 - 1, rtol=1e-6)
    assert row["aux_vol"] >= 0  # deterministic growth -> ~0 vol


def test_missing_momentum_does_not_drop_stock_week():
    prices = _prices({"AAA": 0.01, "SPY": 0.001})
    cal = trading_calendar(prices)
    t = cal[100]
    prices = prices[~((prices["ticker"] == "AAA") & (prices["date"] == cal[95]))]
    feats = price_features(prices, pd.DatetimeIndex([t]))
    row = feats[feats["ticker"] == "AAA"].iloc[0]
    assert np.isnan(row["f_mom_1w"])


def test_labels_are_open_to_open_excess():
    prices = _prices({"AAA": 0.01, "BBB": 0.0, "CCC": -0.01})
    cal = trading_calendar(prices)
    rdates = rebalance_dates(cal, "2022-03-01", "2022-06-30")
    labels = make_labels(prices, rdates, horizon=5)
    t = rdates[0]
    grp = labels[labels.date == t].set_index("ticker")
    assert np.isclose(grp.loc["AAA", "fwd_ret"], 1.01 ** 5 - 1, rtol=1e-6)
    assert np.isclose(grp.loc["BBB", "fwd_ret"], 0.0, atol=1e-12)
    # label = fwd_ret - median; median ticker's label is 0
    assert np.isclose(grp.loc["BBB", "label"], 0.0, atol=1e-12)
    assert grp.loc["AAA", "label"] > 0 > grp.loc["CCC", "label"]
    assert grp["label_end_date"].nunique() == 1
    assert grp["label_end_date"].iloc[0] > t


def test_labels_nan_at_end_of_data():
    prices = _prices({"AAA": 0.01})
    cal = trading_calendar(prices)
    rdates = rebalance_dates(cal, "2022-03-01", str(cal.max().date()))
    labels = make_labels(prices, rdates, horizon=5)
    assert labels[labels.date == rdates[-1]]["fwd_ret"].isna().all()


def test_market_macro_and_calendar_features():
    prices = _prices({"SPY": 0.001})
    cal = trading_calendar(prices)
    rdates = rebalance_dates(cal, "2022-06-01", "2022-12-30")
    fred = pd.DataFrame({"VIXCLS": 20.0},
                        index=pd.date_range("2022-01-01", "2023-01-01", freq="D"))
    mm = market_macro_features(prices, fred, rdates)
    assert {"f_mkt_mom_4w", "f_macro_VIXCLS", "f_macro_VIXCLS_chg"} <= set(mm.columns)
    assert np.isclose(mm["f_macro_VIXCLS"].iloc[-1], 20.0)
    cf = calendar_features(rdates)
    assert set(cf.columns) == {"date", "f_month", "f_woq"}
    assert feature_cols(cf) == ["f_month", "f_woq"]


def test_only_audited_macro_series_enter_production_matrix():
    frame = pd.DataFrame(columns=[
        "f_macro_T10Y2Y", "f_macro_T10Y2Y_chg",
        "f_macro_FEDFUNDS", "f_macro_FEDFUNDS_chg",
        "f_macro_UNRATE", "f_macro_VIXCLS",
    ])
    assert POINT_IN_TIME_MACRO_SERIES == {"T10Y2Y", "FEDFUNDS"}
    assert feature_cols(frame) == [
        "f_macro_T10Y2Y", "f_macro_T10Y2Y_chg",
        "f_macro_FEDFUNDS", "f_macro_FEDFUNDS_chg",
    ]


def _ov_ia_prices(overnight, intraday, start="2022-01-03", ticker="AAA"):
    """Deterministic prices from explicit per-day overnight/intraday return paths."""
    dates = pd.bdate_range(start, periods=len(overnight))
    opens, closes = [], []
    prev_close = 100.0
    for i in range(len(dates)):
        o = prev_close * (1 + overnight[i]) if i > 0 else prev_close
        c = o * (1 + intraday[i])
        opens.append(o)
        closes.append(c)
        prev_close = c
    return pd.DataFrame({"date": dates, "ticker": ticker, "open": opens,
                         "high": np.maximum(opens, closes), "low": np.minimum(opens, closes),
                         "close": closes, "volume": 1e6}), dates, closes


def test_overnight_intraday_decomposition_matches_close_to_close():
    rng = np.random.default_rng(5)
    n = 120
    overnight = rng.normal(0.0005, 0.01, n)
    intraday = rng.normal(0.0003, 0.008, n)
    prices, dates, closes = _ov_ia_prices(overnight, intraday)
    cal = trading_calendar(prices)
    rdates = rebalance_dates(cal, "2022-04-01", "2022-06-30")
    feats = price_features(prices, rdates)
    t = rdates[-1]
    row = feats[feats.date == t].iloc[0]
    pos = dates.get_loc(t)
    lhs = (1 + row["f_overnight_4w"]) * (1 + row["f_intraday_4w"])
    rhs = closes[pos] / closes[pos - 20]
    assert np.isclose(lhs, rhs, rtol=1e-8)


def _beta_prices(spy_ret, stock_ret, ticker):
    dates = pd.bdate_range("2022-01-03", periods=len(spy_ret))
    spy_close = 100.0 * np.cumprod(1 + spy_ret)
    stock_close = 50.0 * np.cumprod(1 + stock_ret)
    rows = []
    for d, c in zip(dates, spy_close):
        rows.append({"date": d, "ticker": "SPY", "open": c, "high": c, "low": c,
                    "close": c, "volume": 1e6})
    for d, c in zip(dates, stock_close):
        rows.append({"date": d, "ticker": ticker, "open": c, "high": c, "low": c,
                    "close": c, "volume": 1e6})
    return pd.DataFrame(rows), dates


def test_beta_and_idio_vol_for_two_times_spy_stock():
    rng = np.random.default_rng(11)
    n = 300
    spy_ret = rng.normal(0.0003, 0.01, n)
    stock_ret = 2 * spy_ret
    prices, dates = _beta_prices(spy_ret, stock_ret, "LEV")
    cal = trading_calendar(prices)
    rdates = rebalance_dates(cal, "2022-06-01", "2022-12-30")
    feats = price_features(prices, rdates)
    row = feats[(feats.date == rdates[-1]) & (feats.ticker == "LEV")].iloc[0]
    assert np.isclose(row["f_beta_60d"], 2.0, atol=1e-6)
    assert np.isclose(row["f_idio_vol_60d"], 0.0, atol=1e-6)


def test_market_residual_reversal_removes_known_beta_component():
    rng = np.random.default_rng(41)
    n = 300
    spy_ret = rng.normal(0.0002, 0.008, n)
    stock_ret = 1.5 * spy_ret
    prices, _ = _beta_prices(spy_ret, stock_ret, "BETA")
    dates = trading_calendar(prices)
    rdates = rebalance_dates(dates, "2022-06-01", "2022-12-30")
    feats = price_features(prices, rdates)
    row = feats[(feats.date == rdates[-1]) & (feats.ticker == "BETA")].iloc[0]
    assert abs(row["f_rev_resid_mkt_1w"]) < 5e-4


def test_weekly_residual_return_lags_remove_known_beta_component():
    rng = np.random.default_rng(47)
    spy_ret = rng.normal(0.0002, 0.008, 320)
    prices, _ = _beta_prices(spy_ret, 1.5 * spy_ret, "BETA")
    rdates = rebalance_dates(trading_calendar(prices), "2022-06-01", "2023-02-28")
    feats = price_features(prices, rdates)
    row = feats[(feats.date == rdates[-1]) & (feats.ticker == "BETA")].iloc[0]
    for lag in range(1, 5):
        assert abs(row[f"f_resid_ret_lag{lag}w"]) < 5e-4


def test_amihud_is_higher_for_same_return_with_lower_volume():
    rng = np.random.default_rng(43)
    n = 180
    returns = rng.normal(0.0002, 0.01, n)
    dates = pd.bdate_range("2022-01-03", periods=n)
    rows = []
    for ticker, volume in (("LIQ", 10_000_000), ("ILLIQ", 100_000)):
        close = 100 * np.cumprod(1 + returns)
        for d, c in zip(dates, close):
            rows.append({"date": d, "ticker": ticker, "open": c, "high": c,
                         "low": c, "close": c, "volume": volume})
    prices = pd.DataFrame(rows)
    rdates = rebalance_dates(trading_calendar(prices), "2022-06-01", "2022-09-30")
    feats = price_features(prices, rdates)
    latest = feats[feats.date == rdates[-1]].set_index("ticker")
    assert latest.loc["ILLIQ", "f_amihud_4w"] > latest.loc["LIQ", "f_amihud_4w"]


def test_beta_zero_and_idio_matches_own_vol_when_independent_of_spy():
    # beta = corr * (sigma_stock / sigma_spy): with unequal vols, even weak
    # sample correlation gets amplified, so an unconstrained random draw is
    # flaky. Instead force the stock's return window to be exactly orthogonal
    # (zero sample covariance) to SPY's over the trailing-60 window the
    # feature actually reads -- beta is then 0 by construction, deterministically.
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    rdates = rebalance_dates(dates, "2022-08-01", "2022-12-30")
    t = rdates[-1]
    pos = dates.get_loc(t)
    lo, hi = pos - 59, pos + 1

    rng = np.random.default_rng(23)
    spy_ret = rng.normal(0.0, 0.01, n)
    stock_ret = rng.normal(0.0, 0.01, n)
    spy_win = spy_ret[lo:hi]
    raw = stock_ret[lo:hi]
    spy_dm, raw_dm = spy_win - spy_win.mean(), raw - raw.mean()
    proj = (raw_dm @ spy_dm) / (spy_dm @ spy_dm)
    stock_ret[lo:hi] = raw - proj * spy_dm  # exactly zero sample covariance with spy_win

    prices, _ = _beta_prices(spy_ret, stock_ret, "IND")
    feats = price_features(prices, rdates)
    row = feats[(feats.date == t) & (feats.ticker == "IND")].iloc[0]
    expected_vol = np.std(stock_ret[lo:hi], ddof=1) * np.sqrt(252)
    assert abs(row["f_beta_60d"]) < 1e-8
    assert np.isclose(row["f_idio_vol_60d"], expected_vol, rtol=1e-6)


def test_dispersion_matches_hand_computed_std_of_5d_returns():
    prices = _prices({"AAA": 0.01, "BBB": 0.0, "CCC": -0.01, "SPY": 0.002})
    cal = trading_calendar(prices)
    rdates = rebalance_dates(cal, "2022-06-01", "2022-12-30")
    fred = pd.DataFrame({"VIXCLS": 20.0},
                        index=pd.date_range("2022-01-01", "2023-01-01", freq="D"))
    mm = market_macro_features(prices, fred, rdates)
    t = rdates[-1]
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    ret5 = close.pct_change(5)
    expected = ret5.loc[t].std(ddof=1)
    got = mm.loc[mm.date == t, "f_mkt_dispersion"].iloc[0]
    assert np.isclose(got, expected)


def test_sector_relative_momentum_known_medians():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2022-01-07")] * 3,
        "ticker": ["AAA", "BBB", "CCC"],
        "sector": ["Tech", "Tech", "Tech"],
        "f_mom_4w": [0.10, 0.05, 0.00],
        "f_mom_12w": [0.20, 0.15, 0.10],
    })
    out = sector_relative_momentum(df)
    b = out[df.ticker == "BBB"].iloc[0]
    a = out[df.ticker == "AAA"].iloc[0]
    c = out[df.ticker == "CCC"].iloc[0]
    assert np.isclose(b["f_mom_4w_sect"], 0.0)   # median stock -> 0
    assert np.isclose(a["f_mom_4w_sect"], 0.05)
    assert np.isclose(c["f_mom_4w_sect"], -0.05)
    assert np.isclose(b["f_mom_12w_sect"], 0.0)


def test_sector_relative_momentum_unknown_sector_is_nan():
    df = pd.DataFrame({
        "date": [pd.Timestamp("2022-01-07")] * 2,
        "ticker": ["AAA", "BBB"],
        "sector": ["Tech", np.nan],
        "f_mom_4w": [0.10, 0.05],
        "f_mom_12w": [0.20, 0.15],
    })
    out = sector_relative_momentum(df)
    assert np.isnan(out[df.ticker == "BBB"]["f_mom_4w_sect"].iloc[0])
