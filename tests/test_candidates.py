import numpy as np
import pandas as pd
from sklearn.base import clone

from stocks_ml.models.candidates import (
    BASELINE_NAMES, AutoMLRegressor, MomentumRank, TopQuintileClassifier,
    WeekGroupedXGBRanker, ZeroForecast, get_candidates, make_xgb,
)


def _xy(n=200, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f_mom_26w": rng.normal(size=n), "f_other": rng.normal(size=n)})
    y = pd.Series(2.0 * X["f_mom_26w"] + rng.normal(0, 0.1, n), name="label")
    return X, y


def _dated_xy(n_weeks=60, per_week=50, seed=3):
    """Date-major panel-like frame: signal in f_mom_26w, noise elsewhere."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2020-01-03", periods=n_weeks, freq="W-FRI")
    n = n_weeks * per_week
    X = pd.DataFrame({"f_mom_26w": rng.normal(size=n), "f_other": rng.normal(size=n)})
    X.attrs["dates"] = np.repeat(weeks.to_numpy(), per_week)
    y = pd.Series(0.02 * X["f_mom_26w"] + rng.normal(0, 0.01, n), name="label")
    return X, y


def test_week_grouped_ranker_learns_within_week_order():
    X, y = _dated_xy()
    model = WeekGroupedXGBRanker(n_estimators=300, early_stopping_rounds=100,
                                 min_child_weight=5).fit(X, y)
    preds = model.predict(X)
    # scores must recover the within-week ordering driven by f_mom_26w
    r = np.corrcoef(preds, X["f_mom_26w"])[0, 1]
    assert r > 0.5
    # predict() centers on the median: above-median conviction is positive
    assert abs(np.median(preds)) < 1e-9


def test_week_grouped_ranker_is_cloneable():
    est = WeekGroupedXGBRanker(n_estimators=50)
    c = clone(est)
    assert c.get_params()["n_estimators"] == 50
    assert c.get_params()["eval_fraction"] == est.eval_fraction


def test_top_quintile_classifier_probabilities_track_signal():
    X, y = _dated_xy()
    model = TopQuintileClassifier(n_estimators=300, early_stopping_rounds=100,
                                  min_child_weight=5).fit(X, y)
    p = model.predict(X)
    assert ((p >= 0) & (p <= 1)).all()
    # higher signal -> higher probability of the top quintile. Magnitudes stay
    # compressed when early stopping keeps few trees (the synthetic signal
    # saturates validation AUC immediately), so assert on ordering, not scale.
    hi = p[X["f_mom_26w"] > 1.0].mean()
    lo = p[X["f_mom_26w"] < -1.0].mean()
    assert hi > lo
    assert np.corrcoef(p, X["f_mom_26w"])[0, 1] > 0.3


def test_top_quintile_classifier_trains_on_extremes_only():
    X, y = _dated_xy()
    model = TopQuintileClassifier(quantile=0.2)
    cls = model._extreme_classes(y, pd.DatetimeIndex(X.attrs["dates"]))
    frac = cls.notna().mean()
    assert 0.35 < frac < 0.45          # ~2 * quantile of rows kept
    assert set(cls.dropna().unique()) == {0.0, 1.0}


def test_zero_forecast():
    X, y = _xy()
    preds = ZeroForecast().fit(X, y).predict(X)
    assert (preds == 0).all() and len(preds) == len(X)


def test_momentum_rank_predicts_momentum_column():
    X, y = _xy()
    preds = MomentumRank().fit(X, y).predict(X)
    np.testing.assert_allclose(preds, X["f_mom_26w"].to_numpy())


def test_xgb_learns_signal_and_handles_nan():
    X, y = _xy(400)
    X.loc[::7, "f_other"] = np.nan
    model = make_xgb().fit(X, y)
    r = np.corrcoef(model.predict(X), y)[0, 1]
    assert r > 0.8


def test_candidates_registry_and_cloneability(tiny_cfg, tmp_path):
    # explicit empty models_dir: must not depend on whether the real repo's
    # models/xgb_tuned.json happens to exist on disk (it does once `stocks-ml
    # tune` has been run for real — see test_tuning.py for xgb_tuned coverage)
    cands = get_candidates(tiny_cfg, models_dir=tmp_path / "models")
    assert set(cands) == {"zero", "momentum", "xgb", "automl"}
    assert set(BASELINE_NAMES) <= set(cands)
    for est in cands.values():
        clone(est)  # must not raise


def test_automl_adapter_interface():
    est = AutoMLRegressor()
    assert hasattr(est, "fit") and hasattr(est, "predict")


def test_automl_adapter_coerces_inputs_and_does_not_mutate_caller_X(monkeypatch):
    import automl_tool.automl as automl_mod

    received = {}

    class FakeAutoML:
        def __init__(self, X, y, outcome):
            received["X"], received["y"], received["outcome"] = X, y, outcome

        def fit_pipeline(self):
            received["X"]["flag"] = received["X"]["flag"].astype(int)  # simulate automl_tool's in-place mutation

    monkeypatch.setattr(automl_mod, "AutoML", FakeAutoML)

    X = pd.DataFrame({"f_mom_26w": [0.1, 0.2, 0.3], "flag": [True, False, True]})
    y = np.array([0.01, -0.02, 0.03])
    AutoMLRegressor().fit(X, y)
    assert isinstance(received["y"], pd.Series) and received["y"].name == "label"
    assert received["outcome"] == "label"
    assert X["flag"].dtype == bool  # caller's frame untouched despite FakeAutoML's mutation
