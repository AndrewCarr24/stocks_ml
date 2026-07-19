import numpy as np
import pandas as pd

from stocks_ml.features.events import filing_features
from stocks_ml.features.panel import trading_calendar


def _prices(tickers, start="2022-01-03", periods=140, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    rows = []
    for t in tickers:
        close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(dates)))
        for d, c in zip(dates, close):
            rows.append({"date": d, "ticker": t, "open": c, "high": c, "low": c,
                        "close": c, "volume": 1e6})
    return pd.DataFrame(rows)


def _edgar(ticker, filed_dates):
    return pd.DataFrame({
        "ticker": [ticker] * len(filed_dates),
        "concept": ["net_income"] * len(filed_dates),
        "filed": pd.to_datetime(filed_dates),
        "form": ["10-K"] * len(filed_dates),
    })


def _empty_edgar():
    return pd.DataFrame({
        "ticker": pd.Series([], dtype=str),
        "concept": pd.Series([], dtype=str),
        "filed": pd.Series([], dtype="datetime64[ns]"),
        "form": pd.Series([], dtype=str),
    })


def test_filed_5d_flag_on_within_window_off_at_six():
    prices = _prices(["AAA"])
    cal = trading_calendar(prices)
    F = cal[30]
    edgar = _edgar("AAA", [F])
    t_on = cal[35]   # F + 5 trading days
    t_off = cal[36]  # F + 6 trading days
    feats = filing_features(edgar, prices, pd.DatetimeIndex([t_on, t_off]))
    row_on = feats[(feats.date == t_on) & (feats.ticker == "AAA")].iloc[0]
    row_off = feats[(feats.date == t_off) & (feats.ticker == "AAA")].iloc[0]
    assert row_on["f_evt_filed_5d"] == 1.0
    assert row_off["f_evt_filed_5d"] == 0.0


def test_days_since_filing_counts_trading_days_and_caps_at_63():
    prices = _prices(["AAA"])
    cal = trading_calendar(prices)
    F = cal[10]
    edgar = _edgar("AAA", [F])
    t_mid = cal[40]   # 30 trading days after F
    t_far = cal[130]  # well over 63 trading days after F
    feats = filing_features(edgar, prices, pd.DatetimeIndex([t_mid, t_far]))
    row_mid = feats[feats.date == t_mid].iloc[0]
    row_far = feats[feats.date == t_far].iloc[0]
    assert row_mid["f_days_since_filing"] == 30
    assert row_far["f_days_since_filing"] == 63


def test_pead_matches_hand_built_reaction_window_and_nan_when_unrealized():
    prices = _prices(["AAA"])
    cal = trading_calendar(prices)
    F = cal[20]
    edgar = _edgar("AAA", [F])
    t = cal[25]  # F + 5 trading days, so F+1 <= t holds
    feats = filing_features(edgar, prices, pd.DatetimeIndex([t]))
    row = feats[feats.date == t].iloc[0]

    close = prices.set_index(["date", "ticker"])["close"]
    expected = close[(cal[21], "AAA")] / close[(cal[19], "AAA")] - 1.0
    assert np.isclose(row["f_pead"], expected)

    # NaN when the reaction window isn't fully realized yet (t == F)
    feats2 = filing_features(edgar, prices, pd.DatetimeIndex([F]))
    row2 = feats2[feats2.date == F].iloc[0]
    assert np.isnan(row2["f_pead"])


def test_no_filing_ticker_all_nan_or_zero():
    prices = _prices(["AAA"])
    cal = trading_calendar(prices)
    edgar = _empty_edgar()
    t = cal[50]
    feats = filing_features(edgar, prices, pd.DatetimeIndex([t]))
    row = feats[feats.date == t].iloc[0]
    assert row["f_evt_filed_5d"] == 0.0
    assert np.isnan(row["f_days_since_filing"])
    assert np.isnan(row["f_pead"])


def test_most_recent_filing_selected_and_dates_deduped():
    prices = _prices(["AAA"])
    cal = trading_calendar(prices)
    F1, F2 = cal[10], cal[40]
    # duplicate filed rows (multiple concepts on the same filed date) must not
    # break "most recent filed <= t" selection.
    edgar = pd.concat([_edgar("AAA", [F1, F1, F2, F2])], ignore_index=True)
    t = cal[45]  # 5 trading days after F2
    feats = filing_features(edgar, prices, pd.DatetimeIndex([t]))
    row = feats[feats.date == t].iloc[0]
    assert row["f_days_since_filing"] == 5
    assert row["f_evt_filed_5d"] == 1.0
