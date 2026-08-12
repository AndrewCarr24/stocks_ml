"""Paired-ΔIC ablation harness: the t >= 3 feature-adoption rule.

Harvey–Liu–Zhu (2016): with the profession's accumulated data-mining, a
home-grown effect needs t ≳ 3, not 1.96. The paired construction gives exactly
that test on our fixed CV design: evaluate the same estimator on identical
folds with and without a feature family, difference the ~488 per-week ICs, and
t = mean(Δ) / SE(Δ). Family-level (one test per bundle), so our own N of
hypotheses stays small; every ablation is a ledger trial either way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.models.candidates import make_tuned, make_xgb
from stocks_ml.models.cv import evaluate_candidate
from stocks_ml.models.trials import record_trials
from stocks_ml.models.tuning import prepare_tuning_data

ADOPTION_T = 3.0


def run_ablation(store, cfg, family_name: str, features: list[str],
                 models_dir: str = "models", out_dir: str = "reports",
                 estimator=None, label_col: str = "label",
                 purge_days: int | None = None) -> dict:
    """Paired comparison: full feature set vs. feature set minus `features`.

    Call on a panel that already contains the candidate features; "adopt" means
    the WITH run beats the WITHOUT run at t >= ADOPTION_T on paired weekly ICs.
    Uses the champion's tuned config unless `estimator` is given."""
    labeled, fcols, splits = prepare_tuning_data(store, cfg, label_col=label_col,
                                                 purge_days=purge_days)
    missing = [f for f in features if f not in fcols]
    if missing:
        raise ValueError(f"features not in the admitted model matrix: {missing}")
    est = estimator or make_tuned("xgb", models_dir) or make_xgb()

    with_res = evaluate_candidate("with_family", est, labeled, splits, fcols,
                                  label_col=label_col)
    reduced = [c for c in fcols if c not in set(features)]
    without_res = evaluate_candidate("without_family", est, labeled, splits, reduced,
                                     label_col=label_col)

    joint = pd.concat([with_res.weekly_scores, without_res.weekly_scores],
                      axis=1, keys=["with_f", "without_f"]).dropna()
    diff = joint["with_f"] - joint["without_f"]
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    t = float(diff.mean() / se) if se and np.isfinite(se) and se > 0 else float("nan")
    adopt = bool(np.isfinite(t) and t >= ADOPTION_T)

    result = {
        "family": family_name, "n_features": len(features), "n_weeks": n,
        "mean_ic_with": with_res.mean_ic, "mean_ic_without": without_res.mean_ic,
        "delta_ic": float(diff.mean()), "t_stat": t, "adopt": adopt,
        "fold_ics_with": with_res.fold_ics, "fold_ics_without": without_res.fold_ics,
    }

    record_trials([{"kind": "ablation", "name": f"ablation_{family_name}",
                    "cv_metric": result["delta_ic"], "t_stat": round(t, 3)
                    if np.isfinite(t) else None, "adopted": adopt}])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds_w = ", ".join(f"{x:.4f}" for x in with_res.fold_ics)
    folds_wo = ", ".join(f"{x:.4f}" for x in without_res.fold_ics)
    (out / f"ablation_{family_name}.md").write_text(
        f"# Ablation: {family_name}\n\n"
        f"Paired weekly-ΔIC test on the fixed pre-holdout CV design "
        f"(adoption hurdle t ≥ {ADOPTION_T:.0f}, per Harvey–Liu–Zhu).\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| features | {len(features)}: {', '.join(features)} |\n"
        f"| weeks compared | {n} |\n"
        f"| mean IC with / without | {with_res.mean_ic:.4f} / {without_res.mean_ic:.4f} |\n"
        f"| mean ΔIC (with − without) | {diff.mean():+.5f} |\n"
        f"| paired t | {t:.2f} |\n"
        f"| fold ICs with | {folds_w} |\n"
        f"| fold ICs without | {folds_wo} |\n\n"
        f"**Verdict: {'ADOPT' if adopt else 'do not adopt'}** "
        f"(t {'≥' if adopt else '<'} {ADOPTION_T:.0f}).\n")
    return result
