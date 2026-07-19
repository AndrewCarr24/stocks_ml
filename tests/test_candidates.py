import numpy as np
import pandas as pd
from sklearn.base import clone

from stocks_ml.models.candidates import (
    BASELINE_NAMES, AutoMLRegressor, MomentumRank, ZeroForecast, get_candidates, make_xgb,
)


def _xy(n=200, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f_mom_26w": rng.normal(size=n), "f_other": rng.normal(size=n)})
    y = pd.Series(2.0 * X["f_mom_26w"] + rng.normal(0, 0.1, n), name="label")
    return X, y


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
