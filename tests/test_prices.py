import pandas as pd

from stocks_ml.data.prices import drop_corrupt_series


def series(t, closes, raw=None, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    df = pd.DataFrame({"date": dates, "ticker": t, "open": closes, "high": closes, "low": closes,
                       "close": closes, "volume": 1e6})
    if raw is not None:
        df["closeunadj"] = raw
    return df


def test_without_a_raw_reference_two_jumps_cut_the_series_from_the_second_point_in_time():
    corrupt = series("BAD", [10, 20, 20, 20, 40, 40, 40, 40, 40, 40])          # two doublings: cut from the second
    bounce = series("AIG", [10, 4, 4, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5])       # one crash, one recovery: kept
    clean = series("OK", [10 + 0.1 * i for i in range(10)])
    out, cut = drop_corrupt_series(pd.concat([corrupt, bounce, clean], ignore_index=True))
    assert list(cut) == ["BAD"] and cut["BAD"] == pd.Timestamp("2020-01-07")   # the 5th session: the second doubling
    bad = out[out.ticker == "BAD"]
    assert len(bad) == 4 and bad.date.max() < cut["BAD"]                          # the history before it stays
    assert set(out.ticker.unique()) == {"BAD", "AIG", "OK"}


def test_a_real_move_is_not_corruption_and_a_broken_adjustment_is_cut_from_its_first_day():
    # Lehman: the raw print collapses with the adjusted close -> a real move, the whole series stays
    lehman = series("LEHMQ", [10, 10, 0.2, 0.05, 0.05, 0.1, 0.02, 0.02], raw=[10, 10, 0.2, 0.05, 0.05, 0.1, 0.02, 0.02])
    # a squeeze: both series jump -> kept
    gme = series("GME", [10, 10, 25, 60, 30, 30, 30, 30], raw=[10, 10, 25, 60, 30, 30, 30, 30])
    # a broken adjustment: the adjusted close doubles on a day the raw print does not move -> cut from that day
    broken = series("BAD", [10, 10, 20, 20, 20, 20, 20, 20], raw=[10, 10, 10, 10, 10, 10, 10, 10])
    # a split: the raw print halves, the adjusted close does not -> kept
    split = series("SPL", [10, 10, 10, 10, 10, 10, 10, 10], raw=[20, 20, 20, 10, 10, 10, 10, 10])
    out, cut = drop_corrupt_series(pd.concat([lehman, gme, broken, split], ignore_index=True))
    assert list(cut) == ["BAD"] and cut["BAD"] == pd.Timestamp("2020-01-03")
    assert (out[out.ticker == "BAD"].date < cut["BAD"]).all() and len(out[out.ticker == "BAD"]) == 2
    for t in ("LEHMQ", "GME", "SPL"):
        assert len(out[out.ticker == t]) == 8


def test_the_split_adjusted_close_is_the_reference_when_present():
    # IAC 2008-08-21: a 1-for-2 reverse split and a spin-off on one day. The raw print barely moves
    # (the two cancel), the split-adjusted and total-return closes both halve: a real event, kept.
    iac = series("IAC", [10, 10, 4.7, 4.7, 4.7, 4.7], raw=[17.6, 17.6, 16.6, 16.6, 16.6, 16.6])
    iac["close_split"] = [35.2, 35.2, 16.6, 16.6, 16.6, 16.6]
    # a broken dividend adjustment: the total-return close doubles, the split-adjusted close does not
    bad = series("BAD", [10, 10, 20, 20, 20, 20], raw=[10, 10, 10, 10, 10, 10])
    bad["close_split"] = [10, 10, 10, 10, 10, 10]
    out, cut = drop_corrupt_series(pd.concat([iac, bad], ignore_index=True))
    assert list(cut) == ["BAD"] and len(out[out.ticker == "IAC"]) == 6
