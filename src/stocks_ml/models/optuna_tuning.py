"""Optuna TPE hyperparameter tuning on purged walk-forward CV only.

The rolling holdout is never read here. Hyperparameters are selected solely by
mean weekly rank IC across the same complete CV calendar used by the tournament.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import optuna

from stocks_ml.models.champion import _eligible
from stocks_ml.models.cv import evaluate_candidate
from stocks_ml.models.tuning import (
    _FAMILY_SPEC, FAMILY_EVAL_OVERRIDES, _full_params, _make_estimator,
    prepare_tuning_data,
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
    if family == "ltr":
        # learning-to-rank shares the xgb tree space; NDCG truncation depth is
        # part of the objective, so it is searched too (top-of-list focus).
        ndcg_at = trial.suggest_categorical("ndcg_at", [8, 16, 32])
        return {
            "eval_metric": f"ndcg@{ndcg_at}",
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 3e-1, log=True),
            "n_estimators": 2000,
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 200),
            "reg_alpha": _suggest_zeroable(trial, "reg_alpha", 1e-4, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }
    if family in ("xgb", "xgb4w"):
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 3e-1, log=True),
            # Safety ceiling only: the time-ordered training tail chooses the
            # effective tree count through early stopping.
            "n_estimators": 5000,
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


def tune_optuna(store, cfg, family: str, n_trials: int = 100, out_dir="models",
                seed: int = 0) -> dict:
    """Select and persist the best full-calendar-eligible TPE trial by CV IC."""
    if family not in _FAMILY_SPEC:
        raise ValueError(f"unknown family {family!r}; valid: {sorted(_FAMILY_SPEC)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_spec = FAMILY_EVAL_OVERRIDES.get(family, {})
    label_col = eval_spec.get("label_col", "label")
    metric = eval_spec.get("metric", "rank_ic")
    prep = {k: v for k, v in eval_spec.items() if k in ("label_col", "purge_days")}
    labeled, fcols, splits = prepare_tuning_data(store, cfg, **prep)

    def objective(trial):
        params = suggest_params(trial, family)
        est = _make_estimator(family, params)
        result = evaluate_candidate("optuna", est, labeled, splits, fcols,
                                    label_col=label_col, metric=metric)
        trial.set_user_attr("mean_ic", result.mean_ic if math.isfinite(result.mean_ic) else None)
        trial.set_user_attr("fold_ics", [x if math.isfinite(x) else None
                           for x in result.fold_ics])
        trial.set_user_attr("n_test_weeks", result.n_test_weeks)
        trial.set_user_attr("expected_test_weeks", result.expected_test_weeks)
        trial.set_user_attr("fold_diagnostics", result.fold_diagnostics or [])
        trial.set_user_attr("eligible", _eligible(result))
        # Mirror the tournament's all-folds-valid gate: steer Optuna away from
        # degenerate regions (a NaN fold would otherwise inflate mean_ic).
        if not _eligible(result):
            return DEGENERATE_SENTINEL
        return result.mean_ic

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials = []
    best_so_far = float("-inf")
    for t in study.trials:
        value = float(t.value) if t.value is not None else None
        if value is not None:
            best_so_far = max(best_so_far, value)
        trials.append({
            "number": t.number,
            "state": t.state.name,
            "value": value,
            "eligible": bool(t.user_attrs.get("eligible", False)),
            "mean_ic": t.user_attrs.get("mean_ic"),
            "fold_ics": t.user_attrs.get("fold_ics", []),
            "n_test_weeks": t.user_attrs.get("n_test_weeks"),
            "expected_test_weeks": t.user_attrs.get("expected_test_weeks"),
            "fold_diagnostics": t.user_attrs.get("fold_diagnostics", []),
            "params": t.params,
            "best_so_far": (best_so_far if math.isfinite(best_so_far) else None),
        })
    audit = {
        "family": family,
        "seed": seed,
        "n_trials_requested": n_trials,
        "n_trials_recorded": len(trials),
        "n_eligible": sum(t["eligible"] for t in trials),
        "n_ineligible": sum(not t["eligible"] for t in trials),
        "n_complete": sum(t["state"] == "COMPLETE" for t in trials),
        "n_failed": sum(t["state"] == "FAIL" for t in trials),
        "n_pruned": sum(t["state"] == "PRUNED" for t in trials),
        "trials": trials,
    }
    (out / f"optuna_{family}_trials.json").write_text(json.dumps(audit, indent=2))

    best_cv = study.best_value
    best_params = suggest_params(optuna.trial.FixedTrial(study.best_params), family)
    full = _full_params(best_params, family)
    eligible = best_cv > DEGENERATE_SENTINEL
    if eligible:
        (out / f"{family}_optuna.json").write_text(json.dumps(full, indent=2))
    else:
        (out / f"{family}_optuna.json").unlink(missing_ok=True)

    verdict = ("selected by CV" if eligible
               else "no eligible trial — no Optuna config written")
    top = sorted((t for t in trials if t["eligible"]),
                 key=lambda t: t["value"], reverse=True)[:10]
    top_lines = ["| trial | mean IC | fold ICs | coverage | best iterations |",
                 "|---:|---:|---|---:|---|"]
    for t in top:
        folds = ", ".join(f"{x:.4f}" for x in t["fold_ics"])
        iterations = ", ".join(str(d["best_iteration"])
                               for d in t["fold_diagnostics"]
                               if "best_iteration" in d)
        top_lines.append(f"| {t['number']} | {t['value']:.4f} | {folds} | "
                         f"{t['n_test_weeks']}/{t['expected_test_weeks']} | "
                         f"{iterations or 'n/a'} |")
    metric_name = {"rank_ic": "mean weekly rank IC",
                   "ndcg8": "mean weekly NDCG@8"}[metric]
    (out / f"optuna_{family}.md").write_text(
        f"# {family} Optuna tuning\n\n"
        f"TPE search, {n_trials} trials, seed {seed}. Selection uses only "
        f"{metric_name} on the pre-holdout purged walk-forward CV folds. The "
        f"holdout is never read by tuning.\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| best CV mean IC | {best_cv:.4f} |\n"
        f"| trials | {n_trials} |\n"
        f"| eligible | {audit['n_eligible']} |\n"
        f"| ineligible | {audit['n_ineligible']} |\n"
        f"| failed | {audit['n_failed']} |\n"
        f"| pruned | {audit['n_pruned']} |\n\n"
        f"**Verdict: {verdict}.**\n\n"
        f"Best config: `{full}`\n\n"
        f"## Top eligible trials\n\n" + "\n".join(top_lines) + "\n\n"
        f"Complete trial details and the best-so-far curve are in "
        f"`optuna_{family}_trials.json`.\n")

    return {"family": family, "best_cv_ic": best_cv, "selected": eligible,
            "n_trials": n_trials, "n_eligible": audit["n_eligible"],
            "n_ineligible": audit["n_ineligible"]}
