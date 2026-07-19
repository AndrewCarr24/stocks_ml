"""Phase B model zoo: LightGBM, CatBoost, IC-scored ElasticNet, and the ensemble."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from stocks_ml.models.candidates import (
    EnsembleCandidate, ICElasticNet, TimeTailEarlyStopCatBoost,
    TimeTailEarlyStopLGBM, get_candidates, make_tuned,
)
from stocks_ml.models.tuning import SEARCH_SPACES, sample_configs, tune_model


def _signal_xy(n=400, seed=0, with_nans=False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"sig": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = pd.Series(2.0 * X["sig"] + rng.normal(0, 0.3, n), name="label")
    if with_nans:
        X.iloc[::7, 1] = np.nan  # sparse NaNs like the real panel
    return X, y


# ---- wrappers: clone, fit/predict finite, nonconstant, silent ----

@pytest.mark.parametrize("factory", [
    lambda: TimeTailEarlyStopLGBM(n_estimators=100),
    lambda: TimeTailEarlyStopCatBoost(iterations=100),
    lambda: ICElasticNet(),
])
def test_wrapper_clone_and_fit(factory):
    est = factory()
    clone(est)  # must not raise
    X, y = _signal_xy(with_nans=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any Python warning fails the test
        preds = clone(est).fit(X, y).predict(X)
    assert np.all(np.isfinite(preds))
    assert preds.std() > 0  # learned the signal, not a constant


def test_icelasticnet_survives_nans():
    """Its reason to exist: plain ElasticNet raises on NaN input."""
    X, y = _signal_xy(with_nans=True)
    assert X.isna().any().any()
    preds = ICElasticNet().fit(X, y).predict(X)
    assert np.all(np.isfinite(preds))
    assert np.corrcoef(preds, X["sig"])[0, 1] > 0.8


def test_lgbm_catboost_learn_signal():
    X, y = _signal_xy()
    for est in (TimeTailEarlyStopLGBM(n_estimators=200), TimeTailEarlyStopCatBoost(iterations=200)):
        p = est.fit(X, y).predict(X)
        assert np.corrcoef(p, X["sig"])[0, 1] > 0.8


# ---- ensemble arithmetic ----

class _Const:
    """Minimal cloneable estimator returning a fixed prediction vector."""
    def __init__(self, values):
        self.values = values

    def get_params(self, deep=True):
        return {"values": self.values}

    def set_params(self, **p):
        self.values = p["values"]
        return self

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(self.values, dtype=float)


def test_ensemble_averages_zscored_predictions():
    a, b = [1.0, 2, 3, 4], [10.0, 20, 30, 40]
    ens = EnsembleCandidate(estimators=[_Const(a), _Const(b)])
    X = pd.DataFrame({"x": range(4)})
    ens.fit(X, pd.Series(range(4)))

    def z(v):
        v = np.asarray(v, float)
        return (v - v.mean()) / v.std()

    expected = (z(a) + z(b)) / 2
    np.testing.assert_allclose(ens.predict(X), expected)


def test_ensemble_is_cloneable():
    ens = EnsembleCandidate(estimators=[_Const([1.0, 2]), _Const([3.0, 4])])
    clone(ens)  # must not raise


# ---- tune_model family dispatch ----

def test_tune_model_dispatches_by_family(synthetic_store, tiny_cfg, tmp_path):
    from stocks_ml.features.panel import build_panel

    build_panel(synthetic_store, tiny_cfg)
    out = tmp_path / "models"
    for family in ("lgbm", "enet"):
        tune_model(synthetic_store, tiny_cfg, family=family, n_samples=2, out_dir=out)
        assert (out / f"{family}_tuned.json").exists()
        assert (out / f"tuning_{family}.md").exists()
        params = json.loads((out / f"{family}_tuned.json").read_text())
        make_tuned(family, out).fit(*_signal_xy()[:2])  # params reconstruct a working est


def test_tune_xgb_preserves_legacy_filenames(synthetic_store, tiny_cfg, tmp_path):
    from stocks_ml.features.panel import build_panel

    build_panel(synthetic_store, tiny_cfg)
    out = tmp_path / "models"
    tune_model(synthetic_store, tiny_cfg, family="xgb", n_samples=2, out_dir=out)
    assert (out / "xgb_tuned.json").exists()   # legacy name
    assert (out / "tuning.md").exists()        # legacy name
    assert (out / "tuning_xgb.md").exists()    # new name too


def test_sample_configs_enet_caps_at_grid_size():
    # enet grid is 4 alphas x 3 l1_ratios = 12; asking for more returns 12
    configs = sample_configs(999, family="enet")
    assert len(configs) == 12
    assert all(set(c) == set(SEARCH_SPACES["enet"]) for c in configs)


# ---- registry file-gating ----

def test_registry_gates_tuned_and_ensemble_on_files(tiny_cfg, tmp_path):
    md = tmp_path / "models"
    md.mkdir()
    base = set(get_candidates(tiny_cfg, models_dir=md))
    assert base == {"zero", "momentum", "xgb", "automl"}  # no tuned files yet

    (md / "lgbm_tuned.json").write_text(json.dumps(
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 100,
         "min_child_samples": 50, "eval_fraction": 0.1, "early_stopping_rounds": 20}))
    one = get_candidates(tiny_cfg, models_dir=md)
    assert "lgbm_tuned" in one and "ensemble" not in one  # 1 tuned -> no ensemble

    (md / "enet_tuned.json").write_text(json.dumps({"alpha": 1e-4, "l1_ratio": 0.5}))
    two = get_candidates(tiny_cfg, models_dir=md)
    assert {"lgbm_tuned", "enet_tuned", "ensemble"} <= set(two)  # 2 tuned -> ensemble appears
