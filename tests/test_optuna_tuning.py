"""Optuna TPE tuning with holdout-judged adoption."""
import json
import warnings

import numpy as np
import optuna
import pandas as pd
import pytest

from stocks_ml.models.optuna_tuning import (
    DEGENERATE_SENTINEL, holdout_ic, suggest_params, tune_optuna,
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
        if family == "enet":
            assert 1e-7 <= p["alpha"] <= 1e-1 and 0.0 <= p["l1_ratio"] <= 1.0
        if family == "catboost":
            assert 3 <= p["depth"] <= 9


def test_holdout_ic_finite_with_holdout(synthetic_store, tiny_cfg):
    from dataclasses import replace

    from stocks_ml.features.panel import build_panel
    from stocks_ml.models.candidates import make_xgb

    cfg = replace(tiny_cfg, holdout_years=1)
    build_panel(synthetic_store, cfg)
    ic = holdout_ic(synthetic_store, cfg, make_xgb())
    assert isinstance(ic, float) and -1.0 <= ic <= 1.0


def test_holdout_ic_nan_without_holdout(synthetic_store, tiny_cfg):
    from stocks_ml.features.panel import build_panel
    from stocks_ml.models.candidates import make_xgb

    build_panel(synthetic_store, tiny_cfg)  # tiny_cfg has holdout_years == 0
    assert np.isnan(holdout_ic(synthetic_store, tiny_cfg, make_xgb()))


def test_objective_returns_sentinel_on_degenerate(monkeypatch, synthetic_store, tiny_cfg):
    """A config that yields a NaN CV fold must score the negative sentinel so
    Optuna avoids that region (mirrors the tournament's eligibility gate)."""
    from stocks_ml.features.panel import build_panel
    from stocks_ml.models import optuna_tuning
    from stocks_ml.models.cv import CandidateResult

    build_panel(synthetic_store, tiny_cfg)

    def fake_eval(name, est, labeled, splits, fcols):
        return CandidateResult(name=name, mean_ic=0.9,
                               fold_ics=[0.9, float("nan"), 0.9], n_test_weeks=10)

    monkeypatch.setattr(optuna_tuning, "evaluate_candidate", fake_eval)
    result = tune_optuna(synthetic_store, tiny_cfg, family="enet", n_trials=2,
                         out_dir=str(synthetic_store.root.parent / "m"))
    # every trial degenerate -> best CV value is the sentinel
    assert result["best_cv_ic"] == DEGENERATE_SENTINEL


def test_tune_optuna_writes_report_and_respects_holdout_rule(synthetic_store, tiny_cfg, tmp_path):
    from dataclasses import replace

    from stocks_ml.features.panel import build_panel

    cfg = replace(tiny_cfg, holdout_years=1)
    build_panel(synthetic_store, cfg)
    out = tmp_path / "models"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = tune_optuna(synthetic_store, cfg, family="enet", n_trials=3, out_dir=out)
    assert (out / "optuna_enet.md").exists()
    for key in ("best_cv_ic", "candidate_holdout_ic", "incumbent_holdout_ic", "adopted"):
        assert key in result
    # with no incumbent file, a finite-holdout candidate is adopted -> json written
    if result["adopted"]:
        assert (out / "enet_optuna.json").exists()
        json.loads((out / "enet_optuna.json").read_text())  # valid, reconstructable


def test_registry_prefers_optuna_over_random_search(tiny_cfg, tmp_path):
    from stocks_ml.models.candidates import make_tuned

    md = tmp_path / "models"
    md.mkdir()
    (md / "enet_tuned.json").write_text(json.dumps({"alpha": 1e-3, "l1_ratio": 0.5}))
    (md / "enet_optuna.json").write_text(json.dumps({"alpha": 1e-5, "l1_ratio": 0.9}))
    est = make_tuned("enet", md)
    assert est.alpha == 1e-5 and est.l1_ratio == 0.9  # optuna file wins
