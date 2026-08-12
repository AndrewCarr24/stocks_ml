from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from stocks_ml.features.panel import feature_cols
from stocks_ml.models.candidates import (
    BASELINE_NAMES, AutoMLRegressor, dated_features, get_candidates,
)
from stocks_ml.models.cv import CandidateResult, evaluate_candidate, make_splits


def _eligible(r: CandidateResult) -> bool:
    """Eligible contenders cover every expected fold and test week.

    Missing or constant weeks must make a candidate ineligible rather than
    silently disappearing from mean IC.
    """
    return (np.isfinite(r.mean_ic)
            and r.expected_folds > 0
            and len(r.fold_ics) == r.expected_folds
            and all(np.isfinite(ic) for ic in r.fold_ics)
            and r.expected_test_weeks > 0
            and r.n_test_weeks == r.expected_test_weeks)


def select_champion(results: dict[str, CandidateResult], baselines=BASELINE_NAMES) -> str:
    def _ic(r):
        return r.mean_ic if r.mean_ic == r.mean_ic else 0.0  # NaN -> no skill -> 0.0

    eligible_baselines = {b: results[b] for b in baselines
                          if b in results and _eligible(results[b])}
    baseline_best = max((_ic(r) for r in eligible_baselines.values()), default=0.0)
    contenders = {n: r for n, r in results.items()
                  if n not in baselines and _eligible(r)}
    if contenders:
        best = max(contenders.values(), key=lambda r: r.mean_ic if r.mean_ic == r.mean_ic else float("-inf"))
        if best.mean_ic == best.mean_ic and best.mean_ic > baseline_best:
            return best.name
    if "momentum" in eligible_baselines:
        return "momentum"
    if eligible_baselines:
        return max(eligible_baselines.values(), key=_ic).name
    raise ValueError("no eligible baseline is available for champion fallback")


def holdout_start_date(dates, holdout_years: int):
    dates = pd.DatetimeIndex(dates).sort_values()
    if holdout_years <= 0 or dates.empty:
        return None
    cutoff = dates.max() - pd.DateOffset(years=holdout_years)
    candidates = dates[dates >= cutoff]
    if candidates.empty or len(candidates) == len(dates):
        return None
    return candidates[0]


def extract_recipe(name: str, fitted_estimator):
    if isinstance(fitted_estimator, AutoMLRegressor) and hasattr(fitted_estimator, "automl_"):
        return clone(fitted_estimator.best_estimator())
    return clone(fitted_estimator)


def _predicts_variation(fitted, panel, fcols) -> bool:
    """A fitted champion must produce varying predictions on the latest panel date.

    A constant predictor cannot rank stocks — live signals would be alphabetical.
    """
    latest = panel[panel["date"] == panel["date"].max()]
    if len(latest) < 3:
        return True
    preds = np.asarray(fitted.predict(latest[fcols]), dtype=float)
    return (len(preds) == len(latest)
            and np.isfinite(preds).all()
            and len(np.unique(np.round(preds, 12))) > 1)


def run_training(store, cfg, candidates: dict | None = None, out_dir="models") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = store.read("panel")
    fcols = feature_cols(panel)
    labeled = panel[panel["label"].notna()]

    all_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    holdout_start = holdout_start_date(all_dates, cfg.holdout_years)
    if holdout_start is not None and "label_end_date" in labeled:
        labeled = labeled[labeled["label_end_date"] < holdout_start]
    if cfg.train_sample_rows:
        labeled = labeled.sort_values("date").tail(cfg.train_sample_rows)
    dates = pd.DatetimeIndex(sorted(labeled["date"].unique()))
    splits = make_splits(dates, cfg.n_cv_folds, cfg.purge_days,
                         0, cfg.eval_start, cfg.cv_train_years,
                         holdout_start=holdout_start)

    candidates = candidates or get_candidates(cfg)
    results = {name: evaluate_candidate(name, est, labeled, splits, fcols)
               for name, est in candidates.items()}

    from stocks_ml.models.trials import record_trials
    record_trials([{
        "kind": "tournament_candidate", "name": n,
        "cv_metric": r.mean_ic if np.isfinite(r.mean_ic) else None,
        "eligible": _eligible(r),
    } for n, r in results.items()], out / "trials_ledger.json")

    fit_df = labeled if holdout_start is None else labeled[labeled["date"] < holdout_start]
    fit_end = fit_df["date"].max()
    fit_df = fit_df[fit_df["date"] >= fit_end - pd.DateOffset(years=cfg.cv_train_years)]

    ineligible_coverage = {
        n: (sum(1 for ic in r.fold_ics if not np.isfinite(ic)),
            r.n_test_weeks, r.expected_test_weeks)
        for n, r in results.items()
        if n not in BASELINE_NAMES and not _eligible(r)
    }
    pool = dict(results)
    final_fit_excluded = []
    while True:
        champ = select_champion(pool)
        fitted = clone(candidates[champ]).fit(dated_features(fit_df, fcols), fit_df["label"])
        # Never let holdout covariates influence selection. Rankability is
        # checked on the latest pre-holdout fit week only.
        if _predicts_variation(fitted, fit_df, fcols) or champ == "momentum":
            break
        final_fit_excluded.append(champ)
        pool.pop(champ, None)
    recipe = extract_recipe(champ, fitted)

    joblib.dump(recipe, out / "champion.joblib")
    (out / "champion.json").write_text(json.dumps(
        {"name": champ, "selected_at": str(date.today()),
         "mean_ic": results[champ].mean_ic}, indent=2))

    lines = ["# Champion selection", "", "| candidate | mean rank IC | fold ICs | test weeks |",
             "|---|---|---|---|"]
    for name, r in sorted(results.items(), key=lambda kv: -(kv[1].mean_ic if kv[1].mean_ic == kv[1].mean_ic else -9e9)):
        marker = " **← champion**" if name == champ else ""
        folds = ", ".join(f"{ic:.4f}" for ic in r.fold_ics)
        lines.append(f"| {name}{marker} | {r.mean_ic:.4f} | {folds} | "
                 f"{r.n_test_weeks}/{r.expected_test_weeks} |")
    lines.append("")
    lines.append(f"Baselines: {', '.join(BASELINE_NAMES)}. "
                 "A champion must beat every baseline or selection falls back to momentum.")
    for name, (n_bad, valid_weeks, expected_weeks) in ineligible_coverage.items():
        lines.append(f"excluded (invalid predictions in {n_bad} folds; coverage "
                     f"{valid_weeks}/{expected_weeks} weeks): {name}")
    for name in final_fit_excluded:
        lines.append(f"{name} excluded: final fit predicts a constant")
    (out / "selection.md").write_text("\n".join(lines))
    return results


def load_champion(out_dir="models"):
    out = Path(out_dir)
    meta = json.loads((out / "champion.json").read_text())
    return meta["name"], joblib.load(out / "champion.joblib")
