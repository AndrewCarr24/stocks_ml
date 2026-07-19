from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from stocks_ml.features.panel import feature_cols
from stocks_ml.models.candidates import TimeTailEarlyStopXGB, make_xgb
from stocks_ml.models.cv import evaluate_candidate, make_splits

SEARCH_SPACE = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [1500],          # ceiling; early stopping picks the real count
    "min_child_weight": [10, 30, 50, 100],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [1.0, 5.0, 20.0],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.3, 0.6, 0.8],
}


def sample_configs(n: int, seed: int = 0) -> list[dict]:
    """n unique random hyperparameter combos from SEARCH_SPACE, deterministic under seed."""
    keys = list(SEARCH_SPACE)
    rng = random.Random(seed)
    seen: set[tuple] = set()
    configs: list[dict] = []
    while len(configs) < n:
        combo = tuple(rng.choice(SEARCH_SPACE[k]) for k in keys)
        if combo in seen:
            continue
        seen.add(combo)
        configs.append(dict(zip(keys, combo)))
    return configs


def _production_params() -> dict:
    """The hand-set config currently live in make_xgb() — the incumbent tuning must beat."""
    live = make_xgb().get_params()
    return {k: live[k] for k in SEARCH_SPACE}


def _full_params(hyperparams: dict) -> dict:
    """The complete kwargs needed to reconstruct an equivalent TimeTailEarlyStopXGB,
    including the wrapper-level defaults that sample_configs/_production_params don't
    sample (eval_fraction, early_stopping_rounds)."""
    defaults = TimeTailEarlyStopXGB()
    return {**hyperparams, "n_jobs": -1, "random_state": 0,
            "eval_fraction": defaults.eval_fraction,
            "early_stopping_rounds": defaults.early_stopping_rounds}


def tune_xgb(store, cfg, n_samples: int = 40, out_dir="models") -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = store.read("panel")
    fcols = feature_cols(panel)
    labeled = panel[panel["label"].notna()]
    assert labeled["date"].is_monotonic_increasing, (
        "panel rows must be date-ordered: TimeTailEarlyStopXGB's early-stop split "
        "takes a positional tail as the validation set, so positional order must "
        "equal chronological order — sort the panel by date if this fires"
    )

    dates = pd.DatetimeIndex(sorted(labeled["date"].unique()))
    # SAME split construction as run_training — holdout stays untouched, identical
    # to the tournament, so plain-CV tuning selection never leaks into the honest test.
    splits = make_splits(dates, cfg.n_cv_folds, cfg.purge_days, cfg.holdout_years * 52)

    hyperparams = [_production_params()] + sample_configs(n_samples)
    records = []
    for i, params in enumerate(hyperparams):
        name = f"cfg{i}"
        est = TimeTailEarlyStopXGB(**params, n_jobs=-1, random_state=0)
        result = evaluate_candidate(name, est, labeled, splits, fcols)
        records.append({
            "name": name,
            "params": params,
            "mean_ic": result.mean_ic,
            "fold_ics": result.fold_ics,
            "n_test_weeks": result.n_test_weeks,
            "is_production": i == 0,
        })

    results = pd.DataFrame(records)
    ranked = results.sort_values("mean_ic", ascending=False, na_position="last").reset_index(drop=True)

    best = ranked.iloc[0]
    (out / "xgb_tuned.json").write_text(json.dumps(_full_params(best["params"]), indent=2))

    lines = ["# XGBoost hyperparameter tuning", "",
             "Selection is by mean weekly rank IC on the pre-holdout purged walk-forward "
             "CV folds (plain CV selection) — the untouched holdout is never used for "
             "tuning and remains the honest test.", "",
             "| config | mean IC | fold ICs | test weeks | params |",
             "|---|---|---|---|---|"]
    for _, row in ranked.iterrows():
        marker = ""
        if row["is_production"]:
            marker += " (production reference)"
        if row["name"] == best["name"]:
            marker += " **← selected**"
        folds = ", ".join(f"{ic:.4f}" for ic in row["fold_ics"])
        mean_ic = row["mean_ic"]
        mean_ic_s = f"{mean_ic:.4f}" if mean_ic == mean_ic else "nan"
        lines.append(f"| {row['name']}{marker} | {mean_ic_s} | {folds} | "
                     f"{row['n_test_weeks']} | {row['params']} |")
    (out / "tuning.md").write_text("\n".join(lines))

    return ranked
