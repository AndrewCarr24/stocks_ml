import pandas as pd

from stocks_ml.data.prices import drop_corrupt_series


def test_drop_corrupt_series_kills_repeat_jumpers_spares_single_bounce():
    dates = pd.bdate_range("2020-01-01", periods=10)
    def series(t, closes):
        return pd.DataFrame({"date": dates[:len(closes)], "ticker": t,
                             "open": closes, "high": closes, "low": closes,
                             "close": closes, "volume": 1e6})
    corrupt = series("BAD", [10, 20, 20, 20, 40, 40, 40, 40, 40, 40])   # two doublings
    bounce = series("AIG", [10, 4, 4, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5])  # one crash -60%, one +62% -> neither trips 1.9x twice
    clean = series("OK", [10 + 0.1 * i for i in range(10)])
    prices = pd.concat([corrupt, bounce, clean], ignore_index=True)
    out, dropped = drop_corrupt_series(prices)
    assert dropped == ["BAD"]
    assert set(out.ticker.unique()) == {"AIG", "OK"}
