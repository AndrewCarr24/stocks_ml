import numpy as np
import pandas as pd
import pytest

from stocks_ml.backtest.strategies import (
    EqualWeightTopK, FractionalKelly, RiskState, SpyFloor, VolScaledTopK,
    make_strategies, select_top_k,
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


def test_full_stop_is_not_absorbing():
    # The cash-lock bug: at zero exposure the NAV freezes, so the drawdown
    # (measured vs the all-time peak) never improves and the release
    # threshold is unreachable. The guard must re-enter on its own.
    strat = VolScaledTopK(k=3, vol_target=10.0, rho=0.3, dd_derisk=0.15, dd_full=0.25)
    fresh = VolScaledTopK(k=3, vol_target=10.0, rho=0.3, dd_derisk=0.15, dd_full=0.25)
    full = fresh.propose_weights(PREDS, VOLS, OK).sum()
    stuck = RiskState(drawdown=0.30)  # frozen NAV: drawdown stays constant
    for _ in range(VolScaledTopK.REENTRY_WEEKS):
        assert strat.propose_weights(PREDS, VOLS, stuck).sum() == 0.0
    # cool-off spent: resumes at half exposure despite the unchanged drawdown
    half = strat.propose_weights(PREDS, VOLS, stuck).sum()
    assert half == pytest.approx(full * 0.5, rel=1e-6)
    # full recovery releases the guard and refills the cool-off budget
    released = strat.propose_weights(PREDS, VOLS, RiskState(drawdown=0.05)).sum()
    assert released == pytest.approx(full, rel=1e-6)
    assert strat.propose_weights(PREDS, VOLS, stuck).sum() == 0.0


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


def test_select_top_k_takes_small_standout_tie():
    # 4 standouts tied at the top: the whole group fits, so all are taken,
    # then the next levels keep filling.
    preds = pd.Series({"W": 0.05, "X": 0.05, "Y": 0.05, "Z": 0.05,
                       "M": 0.03, "N": 0.02, "P": 0.01, "Q": 0.005, "R": 0.001})
    picks = select_top_k(preds, 8)
    assert set(picks) == {"W", "X", "Y", "Z", "M", "N", "P", "Q"}


def test_select_top_k_refuses_group_bigger_than_remaining_slots():
    # 3 genuine picks, then a 6-way tie that would need 5 more slots: the tie
    # is refused whole, never sampled.
    tied = {f"T{i}": 0.02 for i in range(6)}
    preds = pd.Series({"A": 0.09, "B": 0.07, "C": 0.05, **tied})
    picks = select_top_k(preds, 8)
    assert set(picks) == {"A", "B", "C"}


def test_select_top_k_degenerate_two_level_predictions_select_nothing():
    # The failure that motivated the guard: ~150 stocks tied at the top value.
    preds = pd.Series(0.000257, index=[f"S{i:03d}" for i in range(500)])
    preds.iloc[:151] = 0.000265
    assert len(select_top_k(preds, 8)) == 0


def test_select_top_k_ignores_row_order():
    # Selection must be a function of values only — never of ticker order.
    preds = pd.Series({"AAA": 0.02, "BBB": 0.02, "CCC": 0.02, "DDD": 0.05})
    for order in (["AAA", "BBB", "CCC", "DDD"], ["DDD", "CCC", "BBB", "AAA"]):
        assert set(select_top_k(preds.reindex(order), 2)) == {"DDD"}


def test_equal_topk_unfilled_tie_slots_stay_cash():
    tied = {f"T{i}": 0.02 for i in range(7)}   # 7-way tie > 6 remaining slots
    preds = pd.Series({"A": 0.09, "B": 0.07, **tied})
    vols = pd.Series(0.2, index=preds.index)
    w = EqualWeightTopK(k=8).propose_weights(preds, vols, OK)
    _check_invariants(w)
    assert set(w.index) == {"A", "B"}
    assert w.sum() == pytest.approx(2 / 8)


def test_topk_spy_holds_spy_on_degenerate_predictions():
    preds = pd.Series(0.000257, index=[f"S{i:03d}" for i in range(500)])
    preds.iloc[:151] = 0.000265
    vols = pd.Series(0.2, index=preds.index)
    w = SpyFloor(EqualWeightTopK(k=8)).propose_weights(preds, vols, OK)
    assert dict(w) == {"SPY": pytest.approx(1.0)}


def test_vol_scaled_refuses_degenerate_tie():
    preds = pd.Series(0.000257, index=[f"S{i:03d}" for i in range(500)])
    preds.iloc[:151] = 0.000265
    vols = pd.Series(0.2, index=preds.index)
    strat = VolScaledTopK(k=8, vol_target=0.15, rho=0.3, dd_derisk=0.15, dd_full=0.25)
    assert strat.propose_weights(preds, vols, OK).empty


def test_confidence_topk_floor_gates_conviction():
    from stocks_ml.backtest.strategies import ConfidenceTopK, SpyFloor

    probs = pd.Series({"A": 0.72, "B": 0.61, "C": 0.55, "D": 0.49, "E": 0.30})
    vols = pd.Series(0.2, index=probs.index)
    w = ConfidenceTopK(k=4, floor=0.5).propose_weights(probs, vols, OK)
    assert set(w.index) == {"A", "B", "C"}          # D/E below the 50% floor
    assert np.allclose(w, 0.25)                      # slots stay 1/k
    # under SpyFloor, the low-confidence shortfall buys the index
    w_spy = SpyFloor(ConfidenceTopK(k=4, floor=0.5)).propose_weights(probs, vols, OK)
    assert w_spy["SPY"] == pytest.approx(0.25)
    assert w_spy.sum() == pytest.approx(1.0)


def test_confidence_topk_abstains_when_nothing_clears_floor():
    from stocks_ml.backtest.strategies import ConfidenceTopK, SpyFloor

    probs = pd.Series({"A": 0.45, "B": 0.40})
    vols = pd.Series(0.2, index=probs.index)
    assert ConfidenceTopK(k=4, floor=0.5).propose_weights(probs, vols, OK).empty
    w = SpyFloor(ConfidenceTopK(k=4, floor=0.5)).propose_weights(probs, vols, OK)
    assert dict(w) == {"SPY": pytest.approx(1.0)}


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


def test_banded_topk_holds_until_rank_decays_past_band():
    from stocks_ml.backtest.strategies import BandedTopK

    strat = BandedTopK(k=2, exit_rank=4)
    vols = pd.Series(0.2, index=list("ABCDEF"))
    # week 1: A,B best -> held
    w1 = strat.propose_weights(pd.Series({"A": .06, "B": .05, "C": .04, "D": .03,
                                          "E": .02, "F": .01}), vols, OK)
    assert set(w1.index) == {"A", "B"}
    # week 2: B slips to rank 4 (inside band) -> still held despite C,D ranking higher
    w2 = strat.propose_weights(pd.Series({"A": .06, "C": .05, "D": .045, "B": .04,
                                          "E": .02, "F": .01}), vols, OK)
    assert set(w2.index) == {"A", "B"}
    # week 3: B falls to rank 5 (past band) -> evicted, top non-held enters
    w3 = strat.propose_weights(pd.Series({"A": .06, "C": .05, "D": .045, "E": .042,
                                          "B": .04, "F": .01}), vols, OK)
    assert "B" not in w3.index and "C" in w3.index
    # negative prediction evicts regardless of rank
    w4 = strat.propose_weights(pd.Series({"A": -.01, "C": .05, "D": .045, "E": .042,
                                          "B": .04, "F": .01}), vols, OK)
    assert "A" not in w4.index


def test_banded_topk_rejects_inverted_band():
    from stocks_ml.backtest.strategies import BandedTopK

    with pytest.raises(ValueError):
        BandedTopK(k=16, exit_rank=8)


def test_elite_entry_only_fills_from_top_ranks():
    from stocks_ml.backtest.strategies import EliteEntryBanded

    strat = EliteEntryBanded(k=3, entry_rank=2, exit_rank=5)
    vols = pd.Series(0.2, index=list("ABCDEF"))
    # only ranks 1-2 may enter: the third slot stays empty (-> floor)
    w1 = strat.propose_weights(pd.Series({"A": .06, "B": .05, "C": .04, "D": .03,
                                          "E": .02, "F": .01}), vols, OK)
    assert set(w1.index) == {"A", "B"}
    assert w1.sum() == pytest.approx(2 / 3)
    # held names keep slots inside the exit band even when out of the entry pool
    w2 = strat.propose_weights(pd.Series({"C": .06, "D": .05, "A": .04, "B": .03,
                                          "E": .02, "F": .01}), vols, OK)
    assert {"A", "B", "C"} == set(w2.index)   # A,B grandfathered (ranks 3,4), C enters
    with pytest.raises(ValueError):
        EliteEntryBanded(k=3, entry_rank=6, exit_rank=5)


def test_sector_cap_blocks_crowded_entries():
    from stocks_ml.backtest.strategies import SectorCapElite

    smap = {"A": "tech", "B": "tech", "C": "tech", "D": "energy", "E": "tech"}
    strat = SectorCapElite(k=4, entry_rank=4, exit_rank=6, cap=2, sector_map=smap)
    vols = pd.Series(0.2, index=list("ABCDE"))
    w = strat.propose_weights(pd.Series({"A": .06, "B": .05, "C": .04, "D": .03,
                                         "E": .02}), vols, OK)
    # A,B take the two tech slots; C (tech, rank 3) is refused; D (energy) enters;
    # the fourth slot has no eligible candidate left inside entry_rank -> empty
    assert set(w.index) == {"A", "B", "D"}
    assert w.sum() == pytest.approx(3 / 4)
    # grandfathering: held tech names keep slots even if the cap would refuse them now
    w2 = strat.propose_weights(pd.Series({"E": .06, "A": .05, "B": .04, "C": .03,
                                          "D": .02}), vols, OK)
    assert {"A", "B", "D"}.issubset(set(w2.index))
    assert "E" not in w2.index                # E is tech: cap already spent on A,B


def test_eased_weights_moves_half_the_gap():
    from stocks_ml.backtest.strategies import EasedWeights, EqualWeightTopK

    strat = EasedWeights(EqualWeightTopK(k=2), lam=0.5)
    vols = pd.Series(0.2, index=["A", "B", "C"])
    w1 = strat.propose_weights(pd.Series({"A": .05, "B": .04, "C": .01}), vols, OK)
    assert w1["A"] == pytest.approx(0.25)     # half of the 0.5 target, from zero
    # target flips to B,C: A decays toward zero, entrants build up gradually
    w2 = strat.propose_weights(pd.Series({"B": .05, "C": .04, "A": -.01}), vols, OK)
    assert w2["A"] == pytest.approx(0.125)
    assert w2["B"] == pytest.approx(0.375)    # 0.5*0.25 + 0.5*0.5
    _check_invariants(w2)
    with pytest.raises(ValueError):
        EasedWeights(EqualWeightTopK(k=2), lam=1.0)


def test_book_average_partial_where_members_disagree():
    from stocks_ml.backtest.strategies import BookAverage, EqualWeightTopK

    strat = BookAverage([EqualWeightTopK(k=2), EqualWeightTopK(k=4)])
    preds = pd.Series({"A": .05, "B": .04, "C": .03, "D": .02, "E": -.01})
    vols = pd.Series(0.2, index=preds.index)
    w = strat.propose_weights(preds, vols, OK)
    _check_invariants(w)
    assert w["A"] == pytest.approx((0.5 + 0.25) / 2)   # both members hold A
    assert w["C"] == pytest.approx(0.25 / 2)            # only the k=4 member holds C
    with pytest.raises(ValueError):
        BookAverage([])
