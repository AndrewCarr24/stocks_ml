"""ops/fill_basis_regrade.py: the campaign re-graded on the live job's fill rules."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stocks_ml.selection as sel

spec = importlib.util.spec_from_file_location(
    "fill_basis_regrade", Path(__file__).resolve().parents[1] / "ops/fill_basis_regrade.py")
fbr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fbr)


def _world():
    """Ten weeks of daily prices for 110 names; each name's open sits 1% above
    the previous close, so the fill basis and the close basis disagree."""
    days = pd.bdate_range("2024-01-05", periods=50)
    names = [f"T{i:03d}" for i in range(110)] + ["SPY"]
    rng = np.random.default_rng(0)
    closes = pd.DataFrame(np.cumprod(1 + rng.normal(0, 0.01, (len(days), len(names))), axis=0) * 100,
                          index=days, columns=names)
    opens = closes.shift(1).fillna(closes) * 1.01
    labels = pd.date_range("2024-01-05", periods=10, freq="W-FRI")
    members = {t: set(names[:-1]) for t in labels}
    return SimpleNamespace(smap={}, members=members, **sel.price_frames(closes, opens)), names


def test_fills_at_close_grades_a_pick_close_to_close():
    ctx, names = _world()
    t = pd.Timestamp("2024-01-12")
    preds = pd.Series(np.arange(110, 0, -1.0), index=names[:-1])
    fill = sel.slice_row(ctx, t, "4w", preds)
    close = sel.slice_row(fbr.fills_at_close(ctx), t, "4w", preds)
    c = ctx.closes
    n = names[0]
    assert close["top3"] == pytest.approx(
        np.mean([c.loc["2024-02-09", k] / c.loc[t, k] - 1 for k in names[:3]]))
    assert fill["top3"] == pytest.approx(
        np.mean([ctx.opens.loc["2024-02-12", k] / ctx.opens.loc["2024-01-15", k] - 1 for k in names[:3]]))
    assert fill["top15"] == close["top15"] == ",".join(names[:15])
    assert ctx.opens.loc["2024-01-15", n] != c.loc[t, n]


def test_top6_agreement_counts_weeks_with_the_same_six_names():
    a = pd.DataFrame({"week": pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"]),
                      "top15": ["A,B,C,D,E,F,G", "A,B,C,D,E,F,G", "A,B,C,D,E,F,G"]})
    b = pd.DataFrame({"week": pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-26"]),
                      "top15": ["F,E,D,C,B,A,G", "A,B,C,D,E,G,F", "A,B,C,D,E,F,G"]})
    assert fbr.top6_agreement(a, b) == (1, 2)


def test_pre_fix_reads_the_rank_date_regrade_note():
    note = "2021-01 -> 2024-06 (pre-holdout): $174 (+17.2%/yr) | before the rank-date fix (dccca9a): $176 (+18.1%/yr)"
    assert fbr._pre_fix(note) == "$176 (+18.1%/yr)"
    assert fbr._pre_fix("2021: $174") is None


def test_layer_table_flags_a_changed_argmax_between_record_and_deployed():
    ev = {"record": {"cap": (None, {"None": 0.70, "2": 0.69})},
          "close": {"cap": (None, {"None": 0.70, "2": 0.69})},
          "fill": {"cap": (2, {"None": 0.68, "2": 0.71})}}
    table = fbr.layer_table(ev)
    assert "None -> 2 **CHANGED**" in table and "as deployed" in table
    ev["fill"]["cap"] = (None, {"None": 0.72, "2": 0.71})
    assert "CHANGED" not in fbr.layer_table(ev)


def _checks():
    return {"horizon": {"n_weeks": 200, "res": {
                "1w/close": {"net": 7.0, "vs_rand": 3.0}, "1w/fill": {"net": 6.0, "vs_rand": 2.5},
                "4w/close": {"net": 10.0, "vs_rand": 4.0}, "4w/fill": {"net": 9.5, "vs_rand": 3.8}},
                        "paired": {"1w": {"diff": -1.0, "t": -0.4, "level_se": 25.0, "gap_in_bp": 33.0, "gap_out_bp": 17.0},
                                   "4w": {"diff": -0.5, "t": -0.3, "level_se": 9.0, "gap_in_bp": 6.0, "gap_out_bp": 11.0}},
                        "population": {"1w": {"n": 963, "net": 17.0, "net_sampled": 28.0},
                                       "4w": {"n": 963, "net": 15.7, "net_sampled": 14.6}}},
            "window": {"close": (5, {3: 10.3, 5: 10.5}), "fill": (3, {3: 10.4, 5: 10.2})}}


def test_check_table_shows_both_horizons_paired_and_the_window_argmax():
    table = fbr.check_table(_checks())
    assert "200 sampled weeks" in table
    assert ("| 4w top-6, 2y window: net %/yr (vs random) | +10.0 (+4.0) | +9.5 (+3.8) "
            "| -0.5%/yr, t -0.3 (SE of a sampled level 9) |") in table
    assert "| 1w population grid, close basis, 963 weeks: net %/yr all weeks / sampled weeks | +17.0 / +28.0 | | |" in table
    assert "3: +10.3 / 5: +10.5 (5y)" in table and "(3y)" in table


def test_horizon_note_reads_the_saved_numbers():
    ev = {"record": {"horizon": ("4w", {"1w": 7.1, "4w": 10.7})}}
    note = fbr.horizon_note(_checks(), ev)
    assert note.startswith("- The sampled horizon check does not resolve")
    assert "-1.0%/yr paired, t -0.4" in note and "SE of 25%/yr" in note
    assert "+17.0%/yr over 963 weeks, +28.0 at the sampled weeks" in note
    assert "4w leads 1w by +3.6%/yr compounded" in note
    assert "entry (+33 bp)" in note and "exit (+17 bp)" in note
