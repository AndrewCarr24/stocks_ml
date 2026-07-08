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


def test_candidates_registry_and_cloneability(tiny_cfg):
    cands = get_candidates(tiny_cfg)
    assert set(cands) == {"zero", "momentum", "xgb", "automl"}
    assert set(BASELINE_NAMES) <= set(cands)
    for est in cands.values():
        clone(est)  # must not raise


def test_automl_adapter_interface():
    est = AutoMLRegressor()
    assert hasattr(est, "fit") and hasattr(est, "predict")
