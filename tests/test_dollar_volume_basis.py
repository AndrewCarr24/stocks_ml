"""Dollar volume on a split-consistent basis (2026-09-19): the vendor's volume is split-adjusted."""
import numpy as np
import pandas as pd


def test_price_features_dollar_volume_ignores_the_level_basis():
    from stocks_ml.features.panel import price_features
    days = pd.bdate_range("2010-01-04", periods=80)
    rows = []
    for t in days:
        # a name that will split 2:1 after the window: closeunadj = 2 x close_split, volume split-adjusted (x2)
        rows.append({"date": t, "ticker": "SPL", "open": 50.0, "close": 50.0, "closeadj": 50.0, "close_split": 50.0,
                     "closeunadj": 100.0, "volume": 2e6})
        rows.append({"date": t, "ticker": "FLAT", "open": 50.0, "close": 50.0, "closeadj": 50.0, "close_split": 50.0,
                     "closeunadj": 50.0, "volume": 1e6})
        rows.append({"date": t, "ticker": "SPY", "open": 100.0, "close": 100.0, "closeadj": 100.0, "close_split": 100.0,
                     "closeunadj": 100.0, "volume": 1e6})
    prices = pd.DataFrame(rows)
    out = price_features(prices, pd.DatetimeIndex([days[-1]]), level_field="closeunadj")
    dv = out.set_index(["date", "ticker"])["f_dollar_vol"]
    # true dollar volume: SPL trades 2e6 x $50 = $100M, FLAT 1e6 x $50 = $50M; the mixed basis would say $200M for SPL
    assert np.isclose(np.exp(dv.loc[(days[-1], "SPL")]), 100e6) and np.isclose(np.exp(dv.loc[(days[-1], "FLAT")]), 50e6)
