from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stocks_ml.selection import (COST, decide_book, decide_horizon, decide_window,
                                 pick_capped, simulate, slice_row, week_slot)


def _frame(weeks, top6, spy=0.0, rand=0.0, top3=None, top10=None):
    return pd.DataFrame({
        "week": weeks, "top3": top3 if top3 is not None else top6,
        "top6": top6, "top10": top10 if top10 is not None else top6,
        "spy": spy, "rand_mean": rand, "top15": "A,B,C"})


def test_decide_horizon_prefers_cost_adjusted_compounding():
    weeks = pd.date_range("2010-01-01", periods=104, freq="W-FRI")
    # 1w: +30bp/week gross but pays weekly cost; 4w: +200bp/4w
    g1 = _frame(weeks, 0.003)
    g4 = _frame(weeks, 0.020)
    h, res = decide_horizon({"1w": g1, "4w": g4}, weeks[0], weeks[-1])
    assert h == "4w" and res["4w"] > res["1w"]


def test_decide_horizon_flips_when_weekly_dominates():
    weeks = pd.date_range("2010-01-01", periods=104, freq="W-FRI")
    h, _ = decide_horizon({"1w": _frame(weeks, 0.02),
                           "4w": _frame(weeks, 0.005)}, weeks[0], weeks[-1])
    assert h == "1w"


def test_decide_window_pairs_on_common_weeks():
    weeks = pd.date_range("2010-01-01", periods=50, freq="4W-FRI")
    sweeps = {2: _frame(weeks, 0.01, rand=0.005),
              5: _frame(weeks, 0.02, rand=0.005),
              3: _frame(weeks[:10], 0.09, rand=0.0)}  # partial overlap only
    w, res = decide_window(sweeps, weeks[0], weeks[-1])
    # 3y graded only on the common (first-10) weeks like everyone else
    assert set(res) == {2, 3, 5}
    assert w == 3  # its common-week edge is largest


def test_decide_book_uses_named_columns():
    weeks = pd.date_range("2010-01-01", periods=104, freq="W-FRI")
    df = _frame(weeks, top6=0.01, top3=0.02, top10=0.005)
    b, res = decide_book(df, "4w", weeks[0], weeks[-1])
    assert b == 3 and res[3] > res[10]


def test_pick_capped_spills_to_next_sector():
    smap = {"A": "tech", "B": "tech", "C": "tech", "D": "oil", "E": "oil", "F": "bank"}
    got = pick_capped(["A", "B", "C", "D", "E", "F"], cap=2, k=4, smap=smap)
    assert got == ["A", "B", "D", "E"]  # C blocked by tech cap


def test_pick_capped_fills_when_short():
    smap = {"A": "tech", "B": "tech", "C": "tech"}
    got = pick_capped(["A", "B", "C"], cap=2, k=3, smap=smap)
    assert len(got) == 3  # falls back rather than returning a short book


# --- rank-date alignment ---------------------------------------------------
# Rank dates are the last trading day of the week: a Friday, or a Thursday
# when the Friday is a holiday. A pick dated t is paid for the week that
# starts at t's close; snapping t back to the previous Friday label would
# pay it for the week that had already happened.

def _weekly(closes, start="2015-12-04"):
    n = len(next(iter(closes.values())))
    idx = pd.date_range(start, periods=n, freq="W-FRI")
    cw = pd.DataFrame(closes, index=idx, dtype=float)
    return SimpleNamespace(
        cw=cw, wret=cw.pct_change(fill_method=None), spy_w=cw["SPY"], smap={},
        fwd={"1w": cw.pct_change(1, fill_method=None).shift(-1)}, members={})


def test_week_slot_is_the_friday_label_of_the_pick_week():
    idx = pd.date_range("2015-12-04", periods=5, freq="W-FRI")
    assert week_slot(idx, pd.Timestamp("2015-12-24")) == pd.Timestamp("2015-12-25")
    assert week_slot(idx, pd.Timestamp("2015-12-25")) == pd.Timestamp("2015-12-25")
    assert week_slot(idx, pd.Timestamp("2016-01-08")) is None


def test_simulate_pays_a_thursday_pick_for_the_following_week():
    # labels: 12-04, 12-11, 12-18, 12-25, 01-01. A: +10% into the holiday
    # week ending Thu 12-24, then -5% the week after.
    ctx = _weekly({"A": [100, 100, 100, 110, 104.5], "SPY": [100] * 5})
    holdings = pd.DataFrame({"week": [pd.Timestamp("2015-12-24")], "top15": ["A"]})
    rets = simulate(ctx, holdings, "1w", book=1, cap=None, stop=None, floor="none")
    assert list(rets.index) == [pd.Timestamp("2016-01-01")]
    assert rets.iloc[0] == pytest.approx(-0.05 - COST)


def test_simulate_holds_the_book_through_a_week_without_a_pick():
    # picks on 12-04 (A) and 12-18 (B); no pick dated 12-11.
    ctx = _weekly({"A": [100, 102, 105.06, 105.06, 105.06],
                   "B": [100, 100, 100, 103, 103], "SPY": [100] * 5})
    holdings = pd.DataFrame({"week": pd.to_datetime(["2015-12-04", "2015-12-18"]),
                             "top15": ["A", "B"]})
    rets = simulate(ctx, holdings, "1w", book=1, cap=None, stop=None, floor="none")
    assert list(rets.index) == list(pd.to_datetime(["2015-12-11", "2015-12-18", "2015-12-25"]))
    assert rets.iloc[0] == pytest.approx(0.02 - COST)   # A, rotation paid
    assert rets.iloc[1] == pytest.approx(0.03)          # A held, no rotation
    assert rets.iloc[2] == pytest.approx(0.03 - COST)   # B


def test_slice_row_reads_the_forward_return_from_the_pick_week():
    names = [f"N{i}" for i in range(101)]
    closes = {n: [100, 100, 100, 100, 100] for n in names}
    closes["N0"] = [100, 100, 100, 110, 104.5]      # +10% then -5%
    closes["SPY"] = [100, 100, 100, 101, 102.01]     # +1% then +1%
    ctx = _weekly(closes)
    t = pd.Timestamp("2015-12-24")
    ctx.members[t] = names + ["SPY"]
    preds = pd.Series(np.linspace(1, 0, len(names)), index=names)  # N0 ranked first
    row = slice_row(ctx, t, "1w", preds)
    assert row["spy"] == pytest.approx(0.01)
    assert row["top3"] == pytest.approx(-0.05 / 3)


def test_simulate_trace_reassembles_each_credited_week():
    ctx = _weekly({"A": [100, 102, 105.06, 105.06, 105.06],
                   "B": [100, 100, 100, 103, 103], "SPY": [100, 101, 102, 103, 104]})
    holdings = pd.DataFrame({"week": pd.to_datetime(["2015-12-04", "2015-12-18"]),
                             "top15": ["A", "B"]})
    trace = []
    rets = simulate(ctx, holdings, "1w", 1, None, None, "60/40", trace=trace)
    assert [x["nxt"] for x in trace] == list(rets.index)
    assert [x["rotated"] for x in trace] == [[0], [], [0]]
    assert [x["sleeves"] for x in trace] == [[["A"]], [["A"]], [["B"]]]
    assert [x["t"] for x in trace] == list(pd.to_datetime(["2015-12-04", "2015-12-04", "2015-12-18"]))
    for x in trace:
        book_r = np.mean([np.mean(v) for v in x["vals"]]) - x["cost"]
        assert x["r"] == pytest.approx(0.6 * book_r + 0.4 * x["fr"])
