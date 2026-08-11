from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.base import clone


class Split(NamedTuple):
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class CandidateResult:
    name: str
    mean_ic: float
    fold_ics: list
    n_test_weeks: int
    expected_test_weeks: int
    expected_folds: int
    fold_diagnostics: list[dict] | None = None


def make_splits(dates: pd.DatetimeIndex, n_folds: int, purge_days: int,
                holdout_weeks: int, eval_start: pd.Timestamp | None = None,
                train_years: int = 2,
                holdout_start: pd.Timestamp | None = None) -> list[Split]:
    """Purged rolling-window splits with one frozen model per test fold.

    Test blocks cover the usable dates after a 40% calendar warmup and before
    the holdout tail. Each fold fits only the trailing ``train_years`` ending at
    ``test_start - purge_days``; the window advances once per fold, not once per
    scored week.

    `eval_start` additionally restricts TEST blocks to `date >= eval_start`. It is
    part of the fixed evaluation design and must not be moved to accommodate a
    newly added feature. Training windows are unaffected:
    each fold's `train_end = test_start - purge_days` still reaches back over all
    earlier history, so models may still learn from pre-eval_start price data."""
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    if train_years < 1:
        raise ValueError("train_years must be at least 1")
    if holdout_start is not None:
        usable = dates[dates < pd.Timestamp(holdout_start)]
    else:
        usable = dates[: len(dates) - holdout_weeks] if holdout_weeks else dates
    min_train = int(len(usable) * 0.4)
    test_dates = usable[min_train:]
    if eval_start is not None:
        test_dates = test_dates[test_dates >= pd.Timestamp(eval_start)]
    if len(test_dates) < n_folds:
        raise ValueError(
            f"evaluation calendar has {len(test_dates)} test weeks for {n_folds} folds"
        )
    blocks = np.array_split(np.arange(len(test_dates)), n_folds)
    splits = []
    for block in blocks:
        if len(block) == 0:
            continue
        test_start, test_end = test_dates[block[0]], test_dates[block[-1]]
        train_end = test_start - pd.Timedelta(days=purge_days)
        train_start = train_end - pd.DateOffset(years=train_years)
        splits.append(Split(train_start, train_end, test_start, test_end))
    return splits


def weekly_rank_ic(df: pd.DataFrame) -> pd.Series:
    def _ic(g):
        values = g[["label", "pred"]]
        if (len(g) < 3 or not np.isfinite(values.to_numpy()).all()
                or g["label"].nunique() < 2 or g["pred"].nunique() < 2):
            return np.nan
        return g["label"].corr(g["pred"], method="spearman")

    return df.groupby("date").apply(_ic, include_groups=False).dropna()


def evaluate_candidate(name: str, estimator, panel: pd.DataFrame, splits: list[Split],
                       fcols: list[str]) -> CandidateResult:
    from stocks_ml.models.candidates import dated_features

    labeled = panel[np.isfinite(panel["label"])]
    if labeled.duplicated(["date", "ticker"]).any():
        raise ValueError("panel must have exactly one labeled row per date and ticker")
    fold_ics, all_ics, fold_diagnostics = [], [], []
    expected_test_weeks = 0
    for s in splits:
        train = labeled[labeled["date"].between(s.train_start, s.train_end)]
        test = labeled[(labeled["date"] >= s.test_start) & (labeled["date"] <= s.test_end)]
        expected_dates = pd.DatetimeIndex(test["date"].unique()).sort_values()
        expected_test_weeks += len(expected_dates)
        if train.empty or test.empty:
            fold_ics.append(float("nan"))
            continue
        model = clone(estimator)
        model.fit(dated_features(train, fcols), train["label"])
        diagnostic = {
            "train_start": str(pd.Timestamp(train["date"].min()).date()),
            "train_end": str(pd.Timestamp(train["date"].max()).date()),
            "test_start": str(pd.Timestamp(test["date"].min()).date()),
            "test_end": str(pd.Timestamp(test["date"].max()).date()),
            "n_train_rows": int(len(train)),
            "n_train_weeks": int(train["date"].nunique()),
            "n_test_rows": int(len(test)),
            "n_test_weeks": int(test["date"].nunique()),
        }
        if hasattr(model, "early_stop_validation_dates_"):
            inner_train = pd.DatetimeIndex(model.early_stop_train_dates_)
            inner_valid = pd.DatetimeIndex(model.early_stop_validation_dates_)
            best_iteration = getattr(model, "best_iteration", None)
            if best_iteration is None:
                best_iteration = getattr(model, "best_iteration_", None)
            if best_iteration is None and hasattr(model, "model_"):
                best_iteration = model.model_.get_best_iteration()
            diagnostic.update({
                "early_stop_train_end": str(inner_train.max().date()),
                "early_stop_validation_start": str(inner_valid.min().date()),
                "early_stop_validation_end": str(inner_valid.max().date()),
                "early_stop_train_weeks": int(inner_train.nunique()),
                "early_stop_validation_weeks": int(inner_valid.nunique()),
                "best_iteration": int(best_iteration) if best_iteration is not None else None,
            })
        fold_diagnostics.append(diagnostic)
        scored = test[["date", "label"]].copy()
        preds = np.asarray(model.predict(test[fcols]), dtype=float)
        if len(preds) != len(test):
            raise ValueError(
                f"{name} returned {len(preds)} predictions for {len(test)} test rows"
            )
        scored["pred"] = preds
        ics = weekly_rank_ic(scored)
        complete_ics = ics.reindex(expected_dates)
        fold_ics.append(float(complete_ics.mean())
                        if complete_ics.notna().all() else float("nan"))
        all_ics.append(ics)
    combined = pd.concat(all_ics) if all_ics else pd.Series(dtype=float)
    mean_ic = float(combined.mean()) if not combined.empty else float("nan")
    return CandidateResult(name=name, mean_ic=mean_ic, fold_ics=fold_ics,
                           n_test_weeks=int(len(combined)),
                           expected_test_weeks=int(expected_test_weeks),
                           expected_folds=len(splits),
                           fold_diagnostics=fold_diagnostics)
