import numpy as np
import pandas as pd

from stocks_ml.features.panel import (
    calendar_features, feature_cols, make_labels, market_macro_features,
    price_features, rebalance_dates, trading_calendar,
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
