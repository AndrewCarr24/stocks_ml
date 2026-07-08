from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

from stocks_ml.features.panel import feature_cols
from stocks_ml.models.candidates import BASELINE_NAMES, AutoMLRegressor, get_candidates
from stocks_ml.models.cv import CandidateResult, evaluate_candidate, make_splits


def select_champion(results: dict[str, CandidateResult], baselines=BASELINE_NAMES) -> str:
    baseline_best = max(results[b].mean_ic for b in baselines if b in results)
    contenders = {n: r for n, r in results.items() if n not in baselines}
    if contenders:
        best = max(contenders.values(), key=lambda r: (r.mean_ic if r.mean_ic == r.mean_ic else -9e9))
        if best.mean_ic == best.mean_ic and best.mean_ic > baseline_best:
            return best.name
    return "momentum"


def extract_recipe(name: str, fitted_estimator):
    if isinstance(fitted_estimator, AutoMLRegressor) and hasattr(fitted_estimator, "automl_"):
        return clone(fitted_estimator.best_estimator())
    return clone(fitted_estimator)


def run_training(store, cfg, candidates: dict | None = None, out_dir="models") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = store.read("panel")
    fcols = feature_cols(panel)
    labeled = panel[panel["label"].notna()]
    if cfg.train_sample_rows:
        labeled = labeled.sort_values("date").tail(cfg.train_sample_rows)

    dates = pd.DatetimeIndex(sorted(labeled["date"].unique()))
    splits = make_splits(dates, cfg.n_cv_folds, cfg.purge_days, cfg.holdout_years * 52)

    candidates = candidates or get_candidates(cfg)
    results = {name: evaluate_candidate(name, est, labeled, splits, fcols)
               for name, est in candidates.items()}

    champ = select_champion(results)
    holdout_start = dates[-cfg.holdout_years * 52] if cfg.holdout_years else None
    fit_df = labeled if holdout_start is None else labeled[labeled["date"] < holdout_start]
    fitted = clone(candidates[champ]).fit(fit_df[fcols], fit_df["label"])
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
        lines.append(f"| {name}{marker} | {r.mean_ic:.4f} | {folds} | {r.n_test_weeks} |")
    lines.append("")
    lines.append(f"Baselines: {', '.join(BASELINE_NAMES)}. "
                 "A champion must beat every baseline or selection falls back to momentum.")
    (out / "selection.md").write_text("\n".join(lines))
    return results


def load_champion(out_dir="models"):
    out = Path(out_dir)
    meta = json.loads((out / "champion.json").read_text())
    return meta["name"], joblib.load(out / "champion.joblib")
