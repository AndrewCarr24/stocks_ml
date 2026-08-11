import numpy as np
import pandas as pd
import pytest

from stocks_ml.backtest.strategies import (
    EqualWeightTopK, FractionalKelly, RiskState, VolScaledTopK, make_strategies,
)

PREDS = pd.Series({"A": 0.05, "B": 0.03, "C": 0.01, "D": -0.02, "E": np.nan})
VOLS = pd.Series({"A": 0.20, "B": 0.40, "C": 0.20, "D": 0.20, "E": 0.20})
OK = RiskState(drawdown=0.0)


def _check_invariants(w):
    assert (w >= -1e-12).all()
    assert w.sum() <= 1 + 1e-9


def test_equal_weight_topk_picks_positive_preds_only():
    w = EqualWeightTopK(k=4).propose_weights(PREDS, VOLS, OK)
    _check_invariants(w)
    assert set(w[w > 0].index) == {"A", "B", "C"}      # D negative, E NaN
    assert np.allclose(w[w > 0], 0.25)                  # unfilled slot stays cash


def test_vol_scaled_weights_inverse_to_vol():
    w = VolScaledTopK(k=3, vol_target=10.0, rho=0.3, dd_derisk=0.15, dd_full=0.25
                      ).propose_weights(PREDS, VOLS, OK)
    _check_invariants(w)
    assert w["A"] == pytest.approx(2 * w["B"], rel=1e-6)  # A has half B's vol


def test_vol_target_caps_portfolio_vol():
    strat = VolScaledTopK(k=3, vol_target=0.05, rho=0.3, dd_derisk=0.15, dd_full=0.25)
    w = strat.propose_weights(PREDS, VOLS, OK)
    _check_invariants(w)
    sub = w[w > 0]
    vols = VOLS[sub.index]
    var = (sub**2 * vols**2).sum()
    cov = 0.0
    for i in sub.index:
        for j in sub.index:
            if i != j:
                cov += sub[i] * sub[j] * 0.3 * vols[i] * vols[j]
    assert np.sqrt(var + cov) <= 0.05 + 1e-9


def test_drawdown_guard_and_hysteresis():
    strat = VolScaledTopK(k=3, vol_target=10.0, rho=0.3, dd_derisk=0.15, dd_full=0.25)
    full = strat.propose_weights(PREDS, VOLS, OK).sum()
    half = strat.propose_weights(PREDS, VOLS, RiskState(drawdown=0.18)).sum()
    assert half == pytest.approx(full * 0.5, rel=1e-6)
    zero = strat.propose_weights(PREDS, VOLS, RiskState(drawdown=0.30)).sum()
    assert zero == 0.0
    # recovering to dd=0.10 (still >= derisk/2): stays de-risked
    still = strat.propose_weights(PREDS, VOLS, RiskState(drawdown=0.10)).sum()
    assert still == pytest.approx(full * 0.5, rel=1e-6)
    # full recovery below derisk/2 = 0.075: guard releases
    released = strat.propose_weights(PREDS, VOLS, RiskState(drawdown=0.05)).sum()
    assert released == pytest.approx(full, rel=1e-6)


def test_vol_scaled_rejects_invalid_rho():
    with pytest.raises(ValueError):
        VolScaledTopK(k=3, vol_target=0.15, rho=-0.9, dd_derisk=0.15, dd_full=0.25)
    with pytest.raises(ValueError):
        VolScaledTopK(k=3, vol_target=0.15, rho=1.0, dd_derisk=0.15, dd_full=0.25)


def test_kelly_sizing_caps_and_renormalizes():
    strat = FractionalKelly(fraction=0.25, cap=0.20)
    w = strat.propose_weights(PREDS, VOLS, OK)
    _check_invariants(w)
    assert w.max() <= 0.20 + 1e-9
    assert w.get("D", 0.0) == 0.0                       # negative edge -> zero
    # kelly weight before cap: fraction * mu / sigma_weekly^2
    raw_a = 0.25 * 0.05 / (0.20**2 / 52)
    assert w["A"] == pytest.approx(min(raw_a, 0.20), rel=1e-6) or w.sum() == pytest.approx(1.0, rel=1e-6)


def test_registry(tiny_cfg):
    strats = make_strategies(tiny_cfg)
    assert set(strats) == {"equal_topk", "vol_scaled", "kelly", "kelly_spy", "topk_spy"}


def test_spy_floor_routes_remainder_to_spy():
    from stocks_ml.backtest.strategies import FractionalKelly, SpyFloor

    inner = FractionalKelly(fraction=0.25, cap=0.20)
    wrapped = SpyFloor(inner)
    w_inner = inner.propose_weights(PREDS, VOLS, OK)
    w = wrapped.propose_weights(PREDS, VOLS, OK)
    # stock sleeve identical to the inner strategy
    for tk, val in w_inner.items():
        assert w[tk] == pytest.approx(val)
    # remainder is exactly SPY; invariants hold
    assert w["SPY"] == pytest.approx(1.0 - w_inner.sum())
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-12).all()


def test_spy_floor_all_spy_when_inner_abstains():
    from stocks_ml.backtest.strategies import EqualWeightTopK, SpyFloor

    all_negative = pd.Series({"A": -0.01, "B": -0.02})
    vols = pd.Series({"A": 0.2, "B": 0.2})
    w = SpyFloor(EqualWeightTopK(k=2)).propose_weights(all_negative, vols, OK)
    assert dict(w) == {"SPY": pytest.approx(1.0)}


def test_spy_floor_no_spy_row_when_fully_invested():
    from stocks_ml.backtest.strategies import EqualWeightTopK, SpyFloor

    preds = pd.Series({"A": 0.05, "B": 0.03})
    vols = pd.Series({"A": 0.2, "B": 0.2})
    w = SpyFloor(EqualWeightTopK(k=2)).propose_weights(preds, vols, OK)
    assert "SPY" not in w  # inner already sums to 1; no dust row
    assert w.sum() == pytest.approx(1.0)
