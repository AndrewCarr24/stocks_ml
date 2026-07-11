import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from stocks_ml.models.cv import CandidateResult
from stocks_ml.models.champion import (
    extract_recipe, holdout_start_date, load_champion, run_training, select_champion,
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


def test_select_champion_nan_baseline_does_not_poison_gate():
    results = {"zero": _res("zero", float("nan")), "momentum": _res("momentum", 0.03),
               "xgb": _res("xgb", 0.08)}
    assert select_champion(results) == "xgb"          # NaN zero must not block a real winner
    results["xgb"] = _res("xgb", 0.01)
    assert select_champion(results) == "momentum"      # still must beat momentum
    results["xgb"] = _res("xgb", float("nan"))
    assert select_champion(results) == "momentum"      # NaN contender never wins


def test_holdout_start_date_edges():
    dates = pd.DatetimeIndex(pd.date_range("2020-01-03", periods=100, freq="W-FRI"))
    assert holdout_start_date(dates, 0) is None
    assert holdout_start_date(dates, 1) == dates[48]   # 100 - 52
    assert holdout_start_date(dates, 5) is None        # 260 >= 100 -> no usable holdout


def test_degenerate_fold_candidate_ineligible():
    results = {"zero": _res("zero", float("nan")), "momentum": _res("momentum", 0.001),
               "auto": CandidateResult(name="auto", mean_ic=0.05,
                                       fold_ics=[float("nan"), 0.05, 0.05], n_test_weeks=30),
               "xgb": _res("xgb", 0.01)}
    assert select_champion(results) == "xgb"   # auto's higher IC can't win with a NaN fold


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


class Constant(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_final_fit_constant_predictor_is_excluded(synthetic_store, tiny_cfg, tmp_path):
    from stocks_ml.features.panel import build_panel
    from stocks_ml.models.candidates import MomentumRank, ZeroForecast

    build_panel(synthetic_store, tiny_cfg)
    out = tmp_path / "models"
    candidates = {"zero": ZeroForecast(), "momentum": MomentumRank(), "sig": Signal(),
                  "const": Constant()}
    run_training(synthetic_store, tiny_cfg, candidates=candidates, out_dir=out)
    name, est = load_champion(out)
    assert name in {"sig", "momentum"}          # const can never be persisted as champion
    text = (out / "selection.md").read_text()
    assert "selection" in text.lower()
