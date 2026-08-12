import numpy as np
import pandas as pd
import pytest

from stocks_ml.backtest.blend import (MAX_MOVE, MetaAllocator, blended_weights,
                                      run_weights_backtest)


def _nav(weekly_rets, start="2018-01-05", freq="W-FRI"):
    idx = pd.date_range(start, periods=len(weekly_rets) + 1, freq=freq)
    return pd.Series(100.0 * np.cumprod([1.0] + list(1 + np.asarray(weekly_rets))),
                     index=idx)


def test_equal_weights_for_indistinguishable_sleeves():
    flat = _nav([0.001] * 200)
    alloc = MetaAllocator(sleeve_names=["a", "b"])
    w = alloc.review({"a": flat, "b": flat.copy()})
    assert w["a"] == pytest.approx(w["b"]) == pytest.approx(0.5)


def test_tilt_is_bounded_not_winner_take_all():
    good = _nav([0.01] * 200)
    bad = _nav([-0.001] * 200)
    alloc = MetaAllocator(sleeve_names=["good", "bad"])
    w = alloc.review({"good": good, "bad": bad})
    assert w["good"] > w["bad"]                     # tilt toward the leader...
    assert w["good"] / w["bad"] <= (1.25 / 0.75) * 2 + 1e-9  # ...bounded (incl. dead-money halving)
    assert w["bad"] > 0.15                           # loser is shrunk, never fired


def test_drawdown_suspension_and_hysteresis():
    # crash to -50% drawdown -> suspended; partial recovery to -30% stays
    # suspended; recovery to -10% reinstates
    path = [0.002] * 100 + [-0.05] * 14
    crashed = _nav(path)
    steady = _nav([0.002] * len(path))
    alloc = MetaAllocator(sleeve_names=["c", "s"])
    w = alloc.review({"c": crashed, "s": steady})
    assert w["c"] == 0.0 and "c" in alloc.suspended
    cont = _nav([0.03] * 10, start=str(crashed.index[-1].date()))
    partial = pd.concat([crashed, crashed.iloc[-1] * cont / 100.0])
    dd = 1 - partial.iloc[-1] / partial.cummax().iloc[-1]
    assert dd > 0.20
    w = alloc.review({"c": partial, "s": steady})
    assert w["c"] == 0.0                             # hysteresis: still out
    cont2 = _nav([0.05] * 25, start=str(crashed.index[-1].date()))
    recovered = pd.concat([crashed, crashed.iloc[-1] * cont2 / 100.0])
    assert 1 - recovered.iloc[-1] / recovered.cummax().iloc[-1] < 0.20
    w = alloc.review({"c": recovered, "s": steady})
    assert w["c"] > 0.0 and "c" not in alloc.suspended


def test_probation_halves_young_sleeves():
    old = _nav([0.002] * 200)
    young = _nav([0.002] * 20)
    alloc = MetaAllocator(sleeve_names=["old", "young"])
    w = alloc.review({"old": old, "young": young})
    assert w["young"] < w["old"]
    assert w["young"] > 0.2                          # halved, not excluded


def test_change_budget_caps_weight_moves():
    a, b = _nav([0.002] * 200), _nav([0.002] * 200)
    alloc = MetaAllocator(sleeve_names=["a", "b"])
    first = alloc.review({"a": a, "b": b})            # 50/50 adopted outright
    # b crashes: target would drop it to 0, but the move is capped
    crashed_b = pd.concat([b, _nav([-0.06] * 12, start=str(b.index[-1].date()))])
    a2 = pd.concat([a, _nav([0.002] * 12, start=str(a.index[-1].date()))])
    second = alloc.review({"a": a2, "b": crashed_b})
    gap = first["b"] - 0.0
    assert second["b"] == pytest.approx(first["b"] - MAX_MOVE * gap)


def test_blended_weights_and_trading_invariants(tiny_cfg):
    rdates = pd.date_range("2021-02-05", periods=30, freq="W-FRI")
    nav_a, nav_b = _nav([0.002] * 60, "2020-01-03"), _nav([0.001] * 60, "2020-01-03")
    wa = pd.DataFrame(0.5, index=rdates, columns=["AAA", "BBB"])
    wb = pd.DataFrame(1.0, index=rdates[::4], columns=["CCC"])   # slower cadence
    targets, meta = blended_weights({"a": nav_a, "b": nav_b},
                                    {"a": wa, "b": wb}, rdates)
    assert (targets.sum(axis=1) <= 1 + 1e-9).all()
    assert set(targets.columns) <= {"AAA", "BBB", "CCC"}
    assert (meta.sum(axis=1) <= 1 + 1e-9).all()
    # meta weights only change on review dates (every 13 weeks)
    changes = (meta.diff().abs().sum(axis=1) > 1e-12).to_numpy().nonzero()[0]
    assert all(i % 13 == 0 for i in changes)

    dates = pd.bdate_range("2021-01-04", periods=260)
    px = 100.0 * np.cumprod(np.full(len(dates), 1.001))
    prices = pd.concat([
        pd.DataFrame({"date": dates, "ticker": t, "open": px, "high": px,
                      "low": px, "close": px, "volume": 1e6})
        for t in ["AAA", "BBB", "CCC"]], ignore_index=True)
    nav, costs = run_weights_backtest(prices, targets, tiny_cfg)
    assert len(nav) > 50 and costs > 0
    # fully-ish invested in assets compounding 0.1%/day -> NAV grows
    assert nav.iloc[-1] > nav.iloc[0]
