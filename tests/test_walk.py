import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from stocks_ml.models.walk import rebalance_calendar, walk_forward_predictions

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


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


def test_one_fit_per_rebalance_date(tiny_cfg):
    panel, _, _ = _world()
    wf = walk_forward_predictions(panel, Flat(), tiny_cfg)
    # staggered ensemble: one member refreshed per rebalance week
    assert wf.n_fits == len(wf.preds)
    assert all(set(p.index) == set(TICKERS) for p in wf.preds.values())


class CutoffEcho(BaseEstimator, RegressorMixin):
    """Predicts a constant derived from its training cutoff — makes ensemble
    membership observable in the predictions."""

    def fit(self, X, y):
        self.cutoff_ = pd.Timestamp(X.attrs["dates"].max()).toordinal()
        return self

    def predict(self, X):
        return np.full(len(X), float(self.cutoff_))


def test_predictions_average_staggered_members(tiny_cfg):
    panel, prices, rdates = _world()
    wf = walk_forward_predictions(panel, CutoffEcho(), tiny_cfg)
    scored = sorted(wf.preds)
    t = scored[-1]  # well past warmup: ensemble holds retrain_weeks members
    # members were refreshed at t, t-1w, ..., each trained on labeled data up
    # to its own cutoff; CutoffEcho echoes that cutoff, so the ensemble mean
    # must equal the mean of the last retrain_weeks echoes
    idx = scored.index(t)
    expected_members = scored[idx - tiny_cfg.retrain_weeks + 1: idx + 1]
    labeled = panel[panel["label"].notna()]
    echoes = []
    for m in expected_members:
        cut = m - pd.Timedelta(days=tiny_cfg.purge_days)
        echoes.append(float(pd.Timestamp(labeled[labeled["date"] <= cut]["date"].max()).toordinal()))
    assert wf.preds[t].iloc[0] == pytest.approx(np.mean(echoes))


def test_predictions_independent_of_walk_start(tiny_cfg):
    """The refit-anchor bug: a shifted start must not change predictions once
    the ensemble is warmed up (retrain_weeks members present)."""
    panel, prices, rdates = _world()
    full = walk_forward_predictions(panel, CutoffEcho(), tiny_cfg)
    shifted_start = sorted(full.preds)[len(full.preds) // 2 + 1]  # mid-walk, offset anchor
    late = walk_forward_predictions(panel, CutoffEcho(), tiny_cfg, start=shifted_start)
    warm = sorted(late.preds)[tiny_cfg.retrain_weeks - 1:]
    assert len(warm) >= 5
    for t in warm:
        pd.testing.assert_series_equal(full.preds[t], late.preds[t])


def test_single_date_walk_is_the_champion_call(tiny_cfg):
    """selection.ensemble_preds walks one date at a time (start == end):
    exactly one fit, scoring that date's cross-section, and the member's
    training cutoff honours the purge."""
    panel, _, _ = _world()
    t = sorted(panel["date"].unique())[-1]
    wf = walk_forward_predictions(panel, CutoffEcho(), tiny_cfg, start=t, end=t,
                                  purge_days=35)
    assert wf.n_fits == 1 and list(wf.preds) == [t]
    cutoff = pd.Timestamp.fromordinal(int(wf.preds[t].iloc[0]))
    assert t - cutoff >= pd.Timedelta(days=35)


def test_rebalance_every_slows_the_cadence(tiny_cfg):
    panel, prices, rdates = _world()
    weekly = walk_forward_predictions(panel, Flat(), tiny_cfg)
    monthly = walk_forward_predictions(panel, Flat(), tiny_cfg, rebalance_every=4)
    # a quarter of the rebalances (and fits: one member per rebalance)
    assert abs(len(monthly.preds) * 4 - len(weekly.preds)) <= 4
    assert monthly.n_fits == len(monthly.preds)
    # rebalance dates are a subset of the panel's every-4th date grid
    grid = set(pd.DatetimeIndex(sorted(panel["date"].unique()))[::4])
    assert set(monthly.preds) <= grid
    assert set(rebalance_calendar(panel, rebalance_every=4)) <= grid


def test_walk_forward_respects_label_col_and_purge(tiny_cfg):
    panel, prices, rdates = _world()
    panel = panel.rename(columns={"label": "label_4w"})
    panel["label"] = np.nan          # weekly label unusable: must not be touched
    wf = walk_forward_predictions(panel, Flat(), tiny_cfg, label_col="label_4w",
                                  purge_days=42, rebalance_every=4)
    assert len(wf.preds) > 0 and wf.n_fits > 0


def test_end_bound_limits_the_walk(tiny_cfg):
    panel, prices, rdates = _world()
    end = rdates[len(rdates) // 2]
    wf = walk_forward_predictions(panel, Flat(), tiny_cfg, end=end)
    assert max(wf.preds) <= end


def test_cache_roundtrip_reproduces_predictions(tiny_cfg, tmp_path):
    panel, _, _ = _world()
    cache = tmp_path / "walk" / "preds.parquet"
    first = walk_forward_predictions(panel, CutoffEcho(), tiny_cfg, cache_path=cache)
    assert cache.exists()
    second = walk_forward_predictions(panel, Flat(), tiny_cfg, cache_path=cache)
    assert sorted(second.preds) == sorted(first.preds)
    for t in first.preds:   # served from the cache: Flat never ran
        pd.testing.assert_series_equal(first.preds[t], second.preds[t], check_names=False)
