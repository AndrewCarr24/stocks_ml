import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from stocks_ml.models.cv import CandidateResult
from stocks_ml.models.champion import (
    extract_recipe, load_champion, run_training, select_champion,
)


def _res(name, ic):
    return CandidateResult(name=name, mean_ic=ic, fold_ics=[ic], n_test_weeks=10)


def test_select_champion_ml_wins():
    results = {"zero": _res("zero", 0.0), "momentum": _res("momentum", 0.03),
               "xgb": _res("xgb", 0.08)}
    assert select_champion(results) == "xgb"


def test_select_champion_falls_back_to_momentum():
    results = {"zero": _res("zero", 0.0), "momentum": _res("momentum", 0.05),
               "xgb": _res("xgb", 0.02)}
    assert select_champion(results) == "momentum"


class Signal(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["f_mom_4w"].to_numpy()


def test_run_training_writes_artifacts(synthetic_store, tiny_cfg, tmp_path):
    from stocks_ml.features.panel import build_panel
    from stocks_ml.models.candidates import MomentumRank, ZeroForecast

    build_panel(synthetic_store, tiny_cfg)
    out = tmp_path / "models"
    candidates = {"zero": ZeroForecast(), "momentum": MomentumRank(), "sig": Signal()}
    results = run_training(synthetic_store, tiny_cfg, candidates=candidates, out_dir=out)
    assert set(results) == {"zero", "momentum", "sig"}
    assert (out / "champion.joblib").exists()
    assert (out / "champion.json").exists()
    assert "mean rank IC" in (out / "selection.md").read_text()
    name, est = load_champion(out)
    assert name in {"sig", "momentum"}
    assert not hasattr(est, "automl_")  # unfitted recipe
