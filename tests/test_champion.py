import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from stocks_ml.models.cv import CandidateResult
from stocks_ml.models.champion import (
    _eligible, _predicts_variation, extract_recipe, holdout_start_date, load_champion,
    run_training, select_champion,
)


def _res(name, ic):
    return CandidateResult(name=name, mean_ic=ic, fold_ics=[ic], n_test_weeks=10,
                           expected_test_weeks=10, expected_folds=1)


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
    cutoff = dates.max() - pd.DateOffset(years=1)
    assert holdout_start_date(dates, 1) == dates[dates >= cutoff][0]
    assert holdout_start_date(dates, 5) is None


def test_holdout_uses_calendar_year_not_fixed_week_count():
    dates = pd.DatetimeIndex(pd.date_range("2019-01-04", "2021-01-08", freq="W-FRI"))
    start = holdout_start_date(dates, 1)
    assert start == pd.Timestamp("2020-01-10")
    assert len(dates[dates >= start]) == 53


def test_degenerate_fold_candidate_ineligible():
    results = {"zero": _res("zero", float("nan")), "momentum": _res("momentum", 0.001),
               "auto": CandidateResult(name="auto", mean_ic=0.05,
                                       fold_ics=[float("nan"), 0.05, 0.05], n_test_weeks=30,
                                       expected_test_weeks=30, expected_folds=3),
               "xgb": _res("xgb", 0.01)}
    assert select_champion(results) == "xgb"   # auto's higher IC can't win with a NaN fold


def test_incomplete_week_coverage_candidate_ineligible():
    results = {
        "zero": _res("zero", float("nan")),
        "momentum": _res("momentum", 0.001),
        "partial": CandidateResult(name="partial", mean_ic=0.08,
                                   fold_ics=[0.08], n_test_weeks=9,
                                   expected_test_weeks=10, expected_folds=1),
        "xgb": _res("xgb", 0.01),
    }
    assert select_champion(results) == "xgb"


def test_nonfinite_mean_candidate_is_ineligible_even_with_finite_folds():
    inconsistent = CandidateResult(name="bad", mean_ic=float("nan"), fold_ics=[0.01],
                                   n_test_weeks=10, expected_test_weeks=10,
                                   expected_folds=1)
    assert not _eligible(inconsistent)


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


class PartlyNonfinite(BaseEstimator, RegressorMixin):
    def predict(self, X):
        out = X["f_mom_4w"].to_numpy().copy()
        out[0] = np.nan
        return out


def test_latest_prediction_check_rejects_nonfinite_values():
    panel = pd.DataFrame({"date": [pd.Timestamp("2024-01-05")] * 3,
                          "f_mom_4w": [0.1, 0.2, 0.3]})
    assert not _predicts_variation(PartlyNonfinite(), panel, ["f_mom_4w"])
    assert _predicts_variation(Signal(), panel, ["f_mom_4w"])


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
