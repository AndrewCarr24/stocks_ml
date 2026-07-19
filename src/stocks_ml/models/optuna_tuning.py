"""Optuna TPE hyperparameter tuning with holdout-judged adoption.

Optuna optimizes each family's hyperparameters on the CV folds (the same purged
walk-forward folds the tournament uses), but a candidate is ADOPTED only if it
also improves on the untouched holdout tail — the weeks make_splits excludes from
every fold. This keeps Optuna's harder, adaptive search from winning purely by
overfitting the tuning folds (a worse winner's curse than random search).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone

from stocks_ml.features.panel import feature_cols
from stocks_ml.models.champion import holdout_start_date
from stocks_ml.models.cv import evaluate_candidate, weekly_rank_ic
from stocks_ml.models.tuning import (
    _FAMILY_SPEC, _full_params, _make_estimator, prepare_tuning_data,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEGENERATE_SENTINEL = -1.0  # objective value for a config with any NaN CV fold


def _suggest_zeroable(trial, name, low, high):
    """A log-uniform draw in [low, high], or exactly 0.0 (regularization off)."""
    if trial.suggest_categorical(f"{name}_zero", [True, False]):
        return 0.0
    return trial.suggest_float(name, low, high, log=True)


def suggest_params(trial, family: str) -> dict:
    """Sample one hyperparameter config for `family` from Optuna's continuous spaces."""
    if family == "xgb":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 3e-1, log=True),
            "n_estimators": 2000,
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 200),
            "reg_alpha": _suggest_zeroable(trial, "reg_alpha", 1e-4, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }
    if family == "lgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 7, 255),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 3e-1, log=True),
            "n_estimators": 2000,
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300),
            "reg_alpha": _suggest_zeroable(trial, "reg_alpha", 1e-4, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }
    if family == "catboost":
        return {
            "depth": trial.suggest_int("depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
            "iterations": 2000,
        }
    if family == "enet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-7, 1e-1, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    raise ValueError(f"unknown family {family!r}")


def holdout_ic(store, cfg, estimator, fcols=None) -> float:
    """Mean weekly rank IC of `estimator` on the holdout tail — the weeks
    make_splits excludes from every CV fold. Trains on all pre-holdout labeled
    rows, predicts the holdout weeks. nan if there is no holdout (holdout_years==0).

    This tail is touched ONLY here, never during Optuna's search."""
    # Deliberately re-derives `labeled` rather than reusing prepare_tuning_data's:
    # the holdout fit trains on the FULL pre-holdout set, not just the CV-fold rows.
    # Do not "DRY" this into the CV-fold frame — that would narrow the training set.
    panel = store.read("panel")
    fcols = fcols or feature_cols(panel)
    labeled = panel[panel["label"].notna()]
    if cfg.train_sample_rows:
        labeled = labeled.sort_values("date").tail(cfg.train_sample_rows)
    dates = pd.DatetimeIndex(sorted(labeled["date"].unique()))
    hstart = holdout_start_date(dates, cfg.holdout_years)
    if hstart is None:
        return float("nan")
    # Same boundary as champion.py's final fit: train strictly before the holdout.
    train = labeled[labeled["date"] < hstart]
    test = labeled[labeled["date"] >= hstart]
    if train.empty or test.empty:
        return float("nan")
    model = clone(estimator).fit(train[fcols], train["label"])
    scored = test[["date", "label"]].copy()
    scored["pred"] = model.predict(test[fcols])
    return float(weekly_rank_ic(scored).mean())


def _incumbent_holdout_ic(store, cfg, family, out, fcols) -> float:
    """Holdout IC of the current random-search config, if one exists, else nan."""
    path = out / f"{family}_tuned.json"
    if not path.exists():
        return float("nan")
    est = _FAMILY_SPEC[family][0](**json.loads(path.read_text()))
    return holdout_ic(store, cfg, est, fcols)


def tune_optuna(store, cfg, family: str, n_trials: int = 100, out_dir="models",
                seed: int = 0) -> dict:
    """TPE search on the CV folds, adopting the winner only if it also beats the
    random-search incumbent on the untouched holdout. Writes {family}_optuna.json
    (only on adoption) and optuna_{family}.md (always). Returns a summary dict."""
    if family not in _FAMILY_SPEC:
        raise ValueError(f"unknown family {family!r}; valid: {sorted(_FAMILY_SPEC)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labeled, fcols, splits = prepare_tuning_data(store, cfg)

    def objective(trial):
        params = suggest_params(trial, family)
        est = _make_estimator(family, params)
        result = evaluate_candidate("optuna", est, labeled, splits, fcols)
        # Mirror the tournament's all-folds-valid gate: steer Optuna away from
        # degenerate regions (a NaN fold would otherwise inflate mean_ic).
        if not result.fold_ics or any(ic != ic for ic in result.fold_ics):
            return DEGENERATE_SENTINEL
        return result.mean_ic

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_cv = study.best_value
    best_params = suggest_params(optuna.trial.FixedTrial(study.best_params), family)
    full = _full_params(best_params, family)
    cand_holdout = holdout_ic(store, cfg, _FAMILY_SPEC[family][0](**full), fcols)
    incumbent_holdout = _incumbent_holdout_ic(store, cfg, family, out, fcols)

    # Adopt only when the honest test agrees (or there is no incumbent to beat).
    adopt = (cand_holdout == cand_holdout) and (
        incumbent_holdout != incumbent_holdout or cand_holdout >= incumbent_holdout)
    if adopt:
        (out / f"{family}_optuna.json").write_text(json.dumps(full, indent=2))

    verdict = ("adopted: holdout improved" if adopt
               else "rejected: holdout did not improve — random-search config retained")
    (out / f"optuna_{family}.md").write_text(
        f"# {family} Optuna tuning\n\n"
        f"TPE search, {n_trials} trials, seed {seed}. Optimized on CV folds; the "
        f"winner is adopted only if it also beats the random-search config on the "
        f"untouched holdout.\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| best CV mean IC | {best_cv:.4f} |\n"
        f"| best config holdout IC | {_fmt(cand_holdout)} |\n"
        f"| incumbent (random-search) holdout IC | {_fmt(incumbent_holdout)} |\n"
        f"| trials | {n_trials} |\n\n"
        f"**Verdict: {verdict}.**\n\n"
        f"Best config: `{full}`\n")

    return {"family": family, "best_cv_ic": best_cv, "candidate_holdout_ic": cand_holdout,
            "incumbent_holdout_ic": incumbent_holdout, "adopted": adopt, "n_trials": n_trials}


def _fmt(x: float) -> str:
    return f"{x:.4f}" if x == x else "nan"
