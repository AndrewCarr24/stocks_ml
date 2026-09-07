import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stocks_ml.selection as sel

spec = importlib.util.spec_from_file_location(
    "regrade_campaign", Path(__file__).resolve().parents[1] / "ops/regrade_campaign.py")
regrade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regrade)


def _ctx():
    weeks = pd.date_range("2024-06-07", periods=6, freq="W-FRI")
    names = [f"T{i:03d}" for i in range(120)]
    rng = np.random.default_rng(0)
    fwd = pd.DataFrame(rng.normal(0, 0.05, (len(weeks), len(names))), index=weeks, columns=names)
    fwd["SPY"] = 0.01
    return SimpleNamespace(fwd={"4w": fwd}, members={t: set(names) for t in
                                                     list(weeks) + [pd.Timestamp("2024-07-03")]})


def test_graded_rows_keep_cached_order_and_join_thursday_to_its_own_week():
    ctx = _ctx()
    thu = pd.Timestamp("2024-07-03")  # Independence Day Friday: rank date is Thursday
    names = [f"T{i:03d}" for i in range(15)]
    hold = pd.DataFrame({"week": [thu], "tickers": [",".join(names)]})
    rows = regrade.graded_rows(sel, ctx, hold, "4w")
    r = ctx.fwd["4w"].loc[pd.Timestamp("2024-07-05")]  # first W-FRI label >= Thursday
    assert rows.loc[0, "top15"] == ",".join(names)
    assert rows.loc[0, "top3"] == pytest.approx(r[names[:3]].mean())
    assert rows.loc[0, "top6"] == pytest.approx(r[names[:6]].mean())
    assert rows.loc[0, "week"] == thu
    # the pre-fix engine snapped the same Thursday to the Friday before it
    assert sel.week_slot(ctx.fwd["4w"].index, thu) == pd.Timestamp("2024-07-05")
    assert ctx.fwd["4w"].index.asof(thu) == pd.Timestamp("2024-06-28")


def test_pop_table_matches_the_campaign_formulas():
    weeks = pd.date_range("2013-01-04", periods=16, freq="W-FRI")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"week": weeks, "spy": rng.normal(0.002, 0.02, 16),
                       "rand_mean": rng.normal(0.001, 0.02, 16)})
    for k in (3, 6, 10):
        df[f"top{k}"] = df["rand_mean"] + rng.normal(0.005, 0.03, 16)
    out = regrade.pop_table({"4w": df})
    s = out[("2013-2024", "4w", "top6")]
    d = df["top6"] - df["rand_mean"]
    blocks = d.groupby(np.arange(16) // 4).mean()
    assert s["vsrand"] == pytest.approx(d.mean() * 13 * 100)
    assert s["t"] == pytest.approx(blocks.mean() / (blocks.std(ddof=1) / 2))
    sub = df.iloc[::4]
    nav = np.cumprod(1 + sub["top6"].values)
    assert s["cmp"] == pytest.approx((nav[-1] ** (1 / (4 / 13)) - 1) * 100)
    assert s["n"] == 16 and ("2013-2024", "4w", "sp500") in out


def test_layer_table_flags_a_changed_argmax():
    ev = {"old": {"floor": ("70/30", {"none": 0.5, "70/30": 0.6})},
          "new": {"floor": ("none", {"none": 0.7, "70/30": 0.6})}}
    table = regrade.layer_table(ev)
    assert "70/30 -> none **CHANGED**" in table
    ev["new"]["floor"] = ("70/30", {"none": 0.5, "70/30": 0.61})
    assert "CHANGED" not in regrade.layer_table(ev)
