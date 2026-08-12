"""Optuna TPE tuning selected exclusively on purged walk-forward CV."""
import json
import warnings

import optuna
import pytest

from stocks_ml.models.optuna_tuning import (
    DEGENERATE_SENTINEL, suggest_params, tune_optuna,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.mark.parametrize("family", ["xgb", "lgbm", "catboost", "enet"])
def test_suggest_params_in_range(family):
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    seen = []

    def objective(trial):
        p = suggest_params(trial, family)
        seen.append(p)
        return 0.0

    study.optimize(objective, n_trials=3)
    for p in seen:
        if family == "xgb":
            assert 2 <= p["max_depth"] <= 8 and 5e-3 <= p["learning_rate"] <= 3e-1
            assert p["reg_alpha"] == 0.0 or 1e-4 <= p["reg_alpha"] <= 10.0
            assert p["n_estimators"] == 5000
        if family == "enet":
            assert 1e-7 <= p["alpha"] <= 1e-1 and 0.0 <= p["l1_ratio"] <= 1.0
        if family == "catboost":
            assert 3 <= p["depth"] <= 9


def test_objective_returns_sentinel_on_degenerate(monkeypatch, synthetic_store, tiny_cfg):
    """A config that yields a NaN CV fold must score the negative sentinel so
    Optuna avoids that region (mirrors the tournament's eligibility gate)."""
    from stocks_ml.features.panel import build_panel
    from stocks_ml.models import optuna_tuning
    from stocks_ml.models.cv import CandidateResult

    build_panel(synthetic_store, tiny_cfg)

    def fake_eval(name, est, labeled, splits, fcols, **kwargs):
        return CandidateResult(name=name, mean_ic=0.9,
                               fold_ics=[0.9, float("nan"), 0.9], n_test_weeks=10,
                               expected_test_weeks=15, expected_folds=3)

    monkeypatch.setattr(optuna_tuning, "evaluate_candidate", fake_eval)
    result = tune_optuna(synthetic_store, tiny_cfg, family="enet", n_trials=2,
                         out_dir=str(synthetic_store.root.parent / "m"))
    # every trial degenerate -> best CV value is the sentinel
    assert result["best_cv_ic"] == DEGENERATE_SENTINEL
    assert not result["selected"]
    assert not (synthetic_store.root.parent / "m" / "enet_optuna.json").exists()


def test_tune_optuna_writes_cv_winner_without_reading_holdout(
        monkeypatch, synthetic_store, tiny_cfg, tmp_path):
    from dataclasses import replace

    from stocks_ml.features.panel import build_panel
    from stocks_ml.models import optuna_tuning
    from stocks_ml.models.cv import CandidateResult

    cfg = replace(tiny_cfg, holdout_years=1)
    build_panel(synthetic_store, cfg)

    def fake_eval(name, est, labeled, splits, fcols, **kwargs):
        # Holdout rows are physically absent from the tuning frame.
        last_cv_date = max(s.test_end for s in splits)
        assert labeled[labeled["date"] > last_cv_date].shape[0] == 0
        weeks = sum(labeled[labeled["date"].between(s.test_start, s.test_end)]
                    ["date"].nunique() for s in splits)
        return CandidateResult(name=name, mean_ic=0.02,
                               fold_ics=[0.02] * len(splits),
                               n_test_weeks=weeks, expected_test_weeks=weeks,
                               expected_folds=len(splits))

    monkeypatch.setattr(optuna_tuning, "evaluate_candidate", fake_eval)
    out = tmp_path / "models"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = tune_optuna(synthetic_store, cfg, family="enet", n_trials=3, out_dir=out)
    assert (out / "optuna_enet.md").exists()
    assert (out / "optuna_enet_trials.json").exists()
    assert result["selected"]
    assert set(result) == {"family", "best_cv_ic", "selected", "n_trials",
                           "n_eligible", "n_ineligible"}
    audit = json.loads((out / "optuna_enet_trials.json").read_text())
    assert audit["n_trials_recorded"] == 3
    assert audit["n_eligible"] == 3
    assert audit["n_ineligible"] == 0
    assert len(audit["trials"]) == 3
    assert all("best_so_far" in trial for trial in audit["trials"])
    assert (out / "enet_optuna.json").exists()
    json.loads((out / "enet_optuna.json").read_text())
    report = (out / "optuna_enet.md").read_text().lower()
    assert "holdout is never read" in report
    assert "holdout ic" not in report


def test_registry_prefers_optuna_over_random_search(tiny_cfg, tmp_path):
    from stocks_ml.models.candidates import make_tuned

    md = tmp_path / "models"
    md.mkdir()
    (md / "enet_tuned.json").write_text(json.dumps({"alpha": 1e-3, "l1_ratio": 0.5}))
    (md / "enet_optuna.json").write_text(json.dumps({"alpha": 1e-5, "l1_ratio": 0.9}))
    est = make_tuned("enet", md)
    assert est.alpha == 1e-5 and est.l1_ratio == 0.9  # optuna file wins
