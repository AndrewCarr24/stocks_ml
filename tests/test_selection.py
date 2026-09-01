import numpy as np
import pandas as pd

from stocks_ml.selection import (decide_book, decide_horizon, decide_window,
                                 pick_capped)


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
