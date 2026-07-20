from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from stocks_ml.features.panel import feature_cols
from stocks_ml.models.candidates import (
    ICElasticNet, TimeTailEarlyStopCatBoost, TimeTailEarlyStopLGBM,
    TimeTailEarlyStopXGB, make_xgb,
)
from stocks_ml.models.champion import _eligible
from stocks_ml.models.cv import evaluate_candidate, make_splits

# Per-family hyperparameter grids. n_estimators/iterations are ceilings — the
# time-ordered early stopping in each wrapper picks the real count.
SEARCH_SPACES = {
    "xgb": {
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [1500],
        "min_child_weight": [10, 30, 50, 100],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 20.0],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.3, 0.6, 0.8],
    },
    "lgbm": {
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [1500],
        "min_child_samples": [20, 50, 100, 200],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 20.0],
        "subsample": [0.6, 0.8, 1.0],
        "subsample_freq": [1],           # required for subsample<1 to take effect
        "colsample_bytree": [0.3, 0.6, 0.8],
    },
    "catboost": {
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1],
        "l2_leaf_reg": [3.0, 10.0, 30.0],
        "iterations": [1500],
    },
    "enet": {
        "alpha": [1e-6, 1e-5, 1e-4, 1e-3],
        "l1_ratio": [0.1, 0.5, 0.9],
    },
}

# Back-compat: the pre-zoo module exposed a single XGBoost SEARCH_SPACE.
SEARCH_SPACE = SEARCH_SPACES["xgb"]

# family -> (wrapper class, production-default hyperparams for config 0).
_FAMILY_SPEC = {
    "xgb": (TimeTailEarlyStopXGB, None),  # None -> derived from make_xgb() below
    "lgbm": (TimeTailEarlyStopLGBM, {
        "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 1500,
        "min_child_samples": 50, "reg_alpha": 0.0, "reg_lambda": 1.0,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8}),
    "catboost": (TimeTailEarlyStopCatBoost, {
        "depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 3.0, "iterations": 1500}),
    "enet": (ICElasticNet, {"alpha": 1e-4, "l1_ratio": 0.5}),
}

_LEGACY_NAMES = {"xgb": ("xgb_tuned.json", "tuning.md")}


def sample_configs(n: int, seed: int = 0, family: str = "xgb") -> list[dict]:
    """n unique random hyperparameter combos from the family's space, deterministic
    under seed. If the grid has fewer than n unique combinations, returns them all."""
    space = SEARCH_SPACES[family]
    keys = list(space)
    max_combos = 1
    for k in keys:
        max_combos *= len(space[k])
    n = min(n, max_combos)
    rng = random.Random(seed)
    seen: set[tuple] = set()
    configs: list[dict] = []
    while len(configs) < n:
        combo = tuple(rng.choice(space[k]) for k in keys)
        if combo in seen:
            continue
        seen.add(combo)
        configs.append(dict(zip(keys, combo)))
    return configs


def _production_params(family: str = "xgb") -> dict:
    """Config 0: the incumbent default the tuning search must beat. For xgb this is
    the live make_xgb() config; for other families a sensible hand-set default."""
    if family == "xgb":
        live = make_xgb().get_params()
        return {k: live[k] for k in SEARCH_SPACES["xgb"]}
    return dict(_FAMILY_SPEC[family][1])


def _full_params(hyperparams: dict, family: str = "xgb") -> dict:
    """Complete kwargs to reconstruct the tuned wrapper, adding the wrapper-level
    defaults (eval_fraction, early_stopping_rounds, seeds) that the grid doesn't
    sample. Family-specific: only tree families take n_jobs/random_state."""
    if family == "enet":
        return dict(hyperparams)  # no early stopping / seeds in its param set
    defaults = _FAMILY_SPEC[family][0]()
    common = {"eval_fraction": defaults.eval_fraction,
              "early_stopping_rounds": defaults.early_stopping_rounds}
    if family in ("xgb", "lgbm"):
        return {**hyperparams, "n_jobs": -1, "random_state": 0, **common}
    return {**hyperparams, "random_state": 0, **common}  # catboost


def _make_estimator(family: str, params: dict):
    """Construct a fresh tunable estimator for the family from sampled hyperparams,
    adding the seed/jobs kwargs the grid doesn't carry (matching _full_params)."""
    klass = _FAMILY_SPEC[family][0]
    if family in ("xgb", "lgbm"):
        return klass(**params, n_jobs=-1, random_state=0)
    if family == "catboost":
        return klass(**params, random_state=0)
    return klass(**params)


def prepare_tuning_data(store, cfg):
    """The leakage-critical setup shared by random-search and Optuna tuning:
    the labeled frame (with the train_sample_rows truncation mirror + date-order
    assertion) and the CV splits (holdout-excluded, identical to run_training).
    Returns (labeled, fcols, splits)."""
    panel = store.read("panel")
    fcols = feature_cols(panel)
    labeled = panel[panel["label"].notna()]
    # Mirrors run_training's truncation (champion.py) exactly — must stay in sync.
    if cfg.train_sample_rows:
        labeled = labeled.sort_values("date").tail(cfg.train_sample_rows)
    assert labeled["date"].is_monotonic_increasing, (
        "panel rows must be date-ordered: the wrappers' early-stop split takes a "
        "positional tail as the validation set, so positional order must equal "
        "chronological order — sort the panel by date if this fires"
    )
    dates = pd.DatetimeIndex(sorted(labeled["date"].unique()))
    # SAME split construction as run_training — holdout stays untouched.
    splits = make_splits(dates, cfg.n_cv_folds, cfg.purge_days, cfg.holdout_years * 52, cfg.eval_start)
    return labeled, fcols, splits


def select_best(results: pd.DataFrame):
    """Best config by mean IC among ELIGIBLE (all-CV-folds-valid) rows — mirrors
    champion.py's tournament eligibility gate (`_eligible`) so tuning never
    selects a config the tournament's champion selection would then reject.
    A config with any NaN fold IC (a degenerate/constant predictor in that
    fold) silently drops those weeks from mean_ic, overstating its skill —
    exactly the reasoning behind champion.py's `_eligible`.

    Returns None if no config is eligible (results must have an "eligible"
    column, e.g. computed via `_eligible` on each config's CandidateResult)."""
    eligible = results[results["eligible"]]
    if eligible.empty:
        return None
    return eligible.sort_values("mean_ic", ascending=False, na_position="last").iloc[0]


def _build_tuning_report(ranked: pd.DataFrame, best, date_min, date_max, n_rows: int,
                         family: str = "xgb") -> str:
    lines = [f"# {family} hyperparameter tuning", "",
             "Selection is by mean weekly rank IC on the pre-holdout purged walk-forward "
             "CV folds (plain CV selection) — the untouched holdout is never used for "
             "tuning and remains the honest test.",
             "A config is only eligible for selection if every CV fold produced a "
             "non-NaN IC (mirrors champion.py's tournament eligibility gate — a NaN "
             "fold means degenerate/constant predictions in that fold, which would "
             "otherwise silently drop those weeks and inflate mean_ic).", "",
             f"Training window: {pd.Timestamp(date_min).date()} → {pd.Timestamp(date_max).date()} "
             f"({n_rows} labeled rows).", ""]
    if best is None:
        lines += ["**No config was eligible (every sampled/production config had at least "
                  "one degenerate CV fold) — xgb_tuned.json was NOT written.**", ""]
    lines += ["| config | mean IC | fold ICs | test weeks | params |",
             "|---|---|---|---|---|"]
    for _, row in ranked.iterrows():
        marker = ""
        if row["is_production"]:
            marker += " (production reference)"
        if not row["eligible"]:
            marker += " (ineligible: degenerate fold)"
        if best is not None and row["name"] == best["name"]:
            marker += " **← selected**"
        folds = ", ".join(f"{ic:.4f}" for ic in row["fold_ics"])
        mean_ic = row["mean_ic"]
        mean_ic_s = f"{mean_ic:.4f}" if mean_ic == mean_ic else "nan"
        lines.append(f"| {row['name']}{marker} | {mean_ic_s} | {folds} | "
                     f"{row['n_test_weeks']} | {row['params']} |")
    return "\n".join(lines)


def tune_model(store, cfg, family: str = "xgb", n_samples: int = 40,
               out_dir="models") -> pd.DataFrame:
    """Tune one model family by mean rank IC over the tournament's CV folds.

    Writes models/{family}_tuned.json (best eligible config) and
    models/tuning_{family}.md. For xgb, also writes the legacy xgb_tuned.json /
    tuning.md names so existing consumers keep working."""
    if family not in _FAMILY_SPEC:
        raise ValueError(f"unknown family {family!r}; valid: {sorted(_FAMILY_SPEC)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labeled, fcols, splits = prepare_tuning_data(store, cfg)

    hyperparams = [_production_params(family)] + sample_configs(n_samples, family=family)
    records = []
    for i, params in enumerate(hyperparams):
        name = f"cfg{i}"
        est = _make_estimator(family, params)
        result = evaluate_candidate(name, est, labeled, splits, fcols)
        records.append({
            "name": name,
            "params": params,
            "mean_ic": result.mean_ic,
            "fold_ics": result.fold_ics,
            "n_test_weeks": result.n_test_weeks,
            "is_production": i == 0,
            "eligible": _eligible(result),
        })

    results = pd.DataFrame(records)
    ranked = results.sort_values("mean_ic", ascending=False, na_position="last").reset_index(drop=True)

    best = select_best(ranked)
    if best is not None:
        payload = json.dumps(_full_params(best["params"], family), indent=2)
        (out / f"{family}_tuned.json").write_text(payload)
        if family in _LEGACY_NAMES:
            (out / _LEGACY_NAMES[family][0]).write_text(payload)

    report = _build_tuning_report(ranked, best, labeled["date"].min(),
                                  labeled["date"].max(), len(labeled), family)
    (out / f"tuning_{family}.md").write_text(report)
    if family in _LEGACY_NAMES:
        (out / _LEGACY_NAMES[family][1]).write_text(report)

    return ranked


def tune_xgb(store, cfg, n_samples: int = 40, out_dir="models") -> pd.DataFrame:
    """Back-compat shim: tune the xgb family (writes legacy artifact names too)."""
    return tune_model(store, cfg, family="xgb", n_samples=n_samples, out_dir=out_dir)
