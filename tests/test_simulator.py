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
