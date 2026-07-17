from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from stocks_ml.backtest.simulator import BacktestResult, run_backtest
from stocks_ml.backtest.strategies import Strategy

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


class AlwaysFirst(Strategy):
    """100% in the alphabetically first ticker available."""
    name = "always_first"

    def propose_weights(self, preds, vols, risk):
        t = sorted(preds.dropna().index)[0]
        return pd.Series({t: 1.0})


class Flat(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X))


def _world(daily=0.01, periods=280):
    """4 identical tickers compounding at `daily`; panel rows for all of them."""
    dates = pd.bdate_range("2021-01-04", periods=periods)
    px = 100.0 * np.cumprod(np.full(periods, 1 + daily))
    prices = pd.concat([
        pd.DataFrame({"date": dates, "ticker": t, "open": px, "high": px,
                      "low": px, "close": px, "volume": 1e6})
        for t in [*TICKERS, "SPY"]
    ], ignore_index=True)
    rdates = pd.DatetimeIndex([d for d in dates if d.weekday() == 4][4:-2])
    panel = pd.DataFrame({
        "date": np.repeat(rdates, len(TICKERS)),
        "ticker": TICKERS * len(rdates),
        "f_x": 0.5, "aux_vol": 0.2,
        "fwd_ret": 1.01**5 - 1, "label": 0.0,
    })
    return panel, prices, rdates


class Overleveraged(Strategy):
    """Violates the long-only/no-leverage invariant by returning weights summing to 1.5."""
    name = "overleveraged"

    def propose_weights(self, preds, vols, risk):
        t = sorted(preds.dropna().index)[0]
        return pd.Series({t: 1.5})


def test_simulator_rejects_leveraged_weights(tiny_cfg):
    panel, prices, _ = _world()
    with pytest.raises(ValueError):
        run_backtest(panel, prices, Overleveraged(), Flat(), tiny_cfg)


def test_zero_cost_nav_tracks_asset(tiny_cfg):
    panel, prices, rdates = _world()
    cfg = replace(tiny_cfg, cost_bps=0.0)
    res = run_backtest(panel, prices, AlwaysFirst(), Flat(), cfg)
    assert isinstance(res, BacktestResult)
    # fully invested in an asset compounding 1%/bday -> NAV compounds likewise
    realized = res.nav.iloc[-1] / res.nav.iloc[0]
    ideal = 1.01 ** np.busday_count(res.nav.index[0].date(), res.nav.index[-1].date())
    assert realized == pytest.approx(ideal, rel=0.01)


def test_costs_reduce_nav_and_cash_never_negative(tiny_cfg):
    panel, prices, _ = _world()
    nav0 = run_backtest(panel, prices, AlwaysFirst(), Flat(), replace(tiny_cfg, cost_bps=0.0)).nav
    res100 = run_backtest(panel, prices, AlwaysFirst(), Flat(), replace(tiny_cfg, cost_bps=100.0))
    # one initial 100% buy at 100bps -> ~1% drag, no further trades (weights constant)
    assert res100.nav.iloc[-1] / nav0.iloc[-1] == pytest.approx(0.99, rel=0.005)
    assert res100.total_costs == pytest.approx(1.0, rel=0.05)   # ~$1 on $100


def test_retrain_cadence(tiny_cfg):
    panel, prices, _ = _world()
    res = run_backtest(panel, prices, AlwaysFirst(), Flat(), tiny_cfg)
    # weekly rebalances, refit every retrain_weeks -> fits ~= rebalances / retrain_weeks
    expected = int(np.ceil(len(res.weights) / tiny_cfg.retrain_weeks))
    assert abs(res.n_fits - expected) <= 1


def test_weights_recorded(tiny_cfg):
    panel, prices, _ = _world()
    res = run_backtest(panel, prices, AlwaysFirst(), Flat(), tiny_cfg)
    assert (res.weights["AAA"].dropna() == 1.0).all()


class RecordingFirst(AlwaysFirst):
    def __init__(self):
        self.seen = []

    def propose_weights(self, preds, vols, risk):
        self.seen.append(risk.drawdown)
        return super().propose_weights(preds, vols, risk)


def test_removal_haircuts_none_is_byte_identical_to_default(tiny_cfg):
    panel, prices, _ = _world()
    baseline = run_backtest(panel, prices, AlwaysFirst(), Flat(), tiny_cfg)
    explicit_none = run_backtest(panel, prices, AlwaysFirst(), Flat(), tiny_cfg,
                                 removal_haircuts=None)
    pd.testing.assert_series_equal(baseline.nav, explicit_none.nav)
    pd.testing.assert_frame_equal(baseline.weights, explicit_none.weights)
    assert baseline.total_costs == explicit_none.total_costs
    assert baseline.n_fits == explicit_none.n_fits


def test_removal_haircut_reduces_nav_of_liquidated_position(tiny_cfg):
    """AAA is dropped from the panel (simulating index removal) at a known
    rebalance date, so AlwaysFirst switches entirely into BBB from that point.
    Since the portfolio is 100% in AAA right up to the liquidation (cost_bps=0,
    no cash), a haircut=0.5 on that removal event must exactly halve the
    liquidation proceeds -- and therefore exactly halve NAV from that exec day
    onward -- versus the same run without haircuts.
    """
    panel, prices, rdates = _world()
    # index 25 (of 50 weekly rebalances) is comfortably after the model's first
    # fit -- AAA must actually be held for a while before it is "removed", or
    # there is nothing to liquidate
    drop_date = rdates[25]
    panel = panel[~((panel["date"] >= drop_date) & (panel["ticker"] == "AAA"))].reset_index(drop=True)
    cfg = replace(tiny_cfg, cost_bps=0.0)

    close_w = prices.pivot(index="date", columns="ticker", values="close").sort_index().ffill()
    cal = close_w.index
    exec_day = cal[cal.searchsorted(drop_date, side="right")]

    res_plain = run_backtest(panel, prices, AlwaysFirst(), Flat(), cfg)
    haircuts = pd.DataFrame({"ticker": ["AAA"], "date": [exec_day], "haircut": [0.5]})
    res_haircut = run_backtest(panel, prices, AlwaysFirst(), Flat(), cfg,
                               removal_haircuts=haircuts)

    before = res_plain.nav.index < exec_day
    pd.testing.assert_series_equal(res_haircut.nav[before], res_plain.nav[before])
    on_or_after = res_plain.nav.index >= exec_day
    assert on_or_after.sum() > 0
    pd.testing.assert_series_equal(res_haircut.nav[on_or_after], 0.5 * res_plain.nav[on_or_after],
                                   check_names=False)


def test_removal_haircut_applies_to_first_of_two_removal_events_for_same_ticker(tiny_cfg):
    """AAA is removed, re-added, then removed again -- two distinct removal
    events for the same ticker, far enough apart (112 days) that a 35-day
    tolerance window never spans both. removal_haircuts carries a row for
    EACH event with a different haircut. A dict keyed only by ticker (keeping
    "last one wins") would silently drop the first event's haircut, since the
    second event's date is >35 days from the first liquidation's exec day.
    Both haircuts must independently apply.
    """
    panel, prices, rdates = _world()
    first_drop, readd, second_drop = rdates[20], rdates[28], rdates[36]
    mask_gone = ((panel["ticker"] == "AAA") &
                (((panel["date"] >= first_drop) & (panel["date"] < readd)) |
                 (panel["date"] >= second_drop)))
    panel = panel[~mask_gone].reset_index(drop=True)
    cfg = replace(tiny_cfg, cost_bps=0.0)

    close_w = prices.pivot(index="date", columns="ticker", values="close").sort_index().ffill()
    cal = close_w.index
    exec_day1 = cal[cal.searchsorted(first_drop, side="right")]
    exec_day2 = cal[cal.searchsorted(second_drop, side="right")]
    assert (exec_day2 - exec_day1).days > 35  # the two events must not overlap the tolerance window

    res_plain = run_backtest(panel, prices, AlwaysFirst(), Flat(), cfg)
    haircuts = pd.DataFrame({"ticker": ["AAA", "AAA"], "date": [exec_day1, exec_day2],
                             "haircut": [0.3, 0.6]})
    res_haircut = run_backtest(panel, prices, AlwaysFirst(), Flat(), cfg,
                               removal_haircuts=haircuts)

    before1 = res_plain.nav.index < exec_day1
    mid = (res_plain.nav.index >= exec_day1) & (res_plain.nav.index < exec_day2)
    after2 = res_plain.nav.index >= exec_day2
    assert before1.sum() > 0 and mid.sum() > 0 and after2.sum() > 0

    pd.testing.assert_series_equal(res_haircut.nav[before1], res_plain.nav[before1])
    # the FIRST event's 0.3 haircut must already show up here, well before the
    # second event's date -- this is what a "last one wins" dict would miss
    pd.testing.assert_series_equal(res_haircut.nav[mid], 0.7 * res_plain.nav[mid],
                                   check_names=False)
    # both haircuts compound: (1-0.3) * (1-0.6) = 0.28
    pd.testing.assert_series_equal(res_haircut.nav[after2], 0.28 * res_plain.nav[after2],
                                   check_names=False)


def test_drawdown_reflects_intraweek_peak(tiny_cfg):
    dates = pd.bdate_range("2021-01-04", periods=280)
    spike_day = dates[(dates.weekday == 2)][40]        # a mid-span Wednesday, well after first fit
    px = np.where(dates < spike_day, 100.0, 106.0)
    px[dates == spike_day] = 120.0
    prices = pd.concat([
        pd.DataFrame({"date": dates, "ticker": t, "open": px, "high": px,
                      "low": px, "close": px, "volume": 1e6})
        for t in [*TICKERS, "SPY"]
    ], ignore_index=True)
    rdates = pd.DatetimeIndex([d for d in dates if d.weekday() == 4][4:-2])
    panel = pd.DataFrame({
        "date": np.repeat(rdates, len(TICKERS)),
        "ticker": TICKERS * len(rdates),
        "f_x": 0.5, "aux_vol": 0.2, "fwd_ret": 0.0, "label": 0.0,
    })
    strat = RecordingFirst()
    run_backtest(panel, prices, strat, Flat(), replace(tiny_cfg, cost_bps=0.0))
    # NAV peaks at the Wednesday spike (120) then settles at 106: the next signal
    # must see dd = 1 - 106/120, not 0
    assert max(strat.seen) == pytest.approx(1 - 106.0 / 120.0, abs=1e-6)
