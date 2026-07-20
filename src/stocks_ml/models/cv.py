from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.base import clone


class Split(NamedTuple):
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class CandidateResult:
    name: str
    mean_ic: float
    fold_ics: list
    n_test_weeks: int


def make_splits(dates: pd.DatetimeIndex, n_folds: int, purge_days: int,
                holdout_weeks: int, eval_start: pd.Timestamp | None = None) -> list[Split]:
    """Purged walk-forward splits. Test blocks cover the usable dates after a 40%
    minimum-train warmup and before the holdout tail.

    `eval_start` additionally restricts TEST blocks to `date >= eval_start` — used
    to keep every scored fold inside the window where all features are available
    (EDGAR fundamentals/filing data begin ~2010). Training windows are unaffected:
    each fold's `train_end = test_start - purge_days` still reaches back over all
    earlier history, so models may still learn from pre-eval_start price data."""
    usable = dates[: len(dates) - holdout_weeks] if holdout_weeks else dates
    min_train = int(len(usable) * 0.4)
    test_dates = usable[min_train:]
    if eval_start is not None:
        test_dates = test_dates[test_dates >= pd.Timestamp(eval_start)]
    blocks = np.array_split(np.arange(len(test_dates)), n_folds)
    splits = []
    for block in blocks:
        if len(block) == 0:
            continue
        test_start, test_end = test_dates[block[0]], test_dates[block[-1]]
        splits.append(Split(test_start - pd.Timedelta(days=purge_days), test_start, test_end))
    return splits


def weekly_rank_ic(df: pd.DataFrame) -> pd.Series:
    def _ic(g):
        if len(g) < 3 or g["label"].nunique() < 2 or g["pred"].nunique() < 2:
            return np.nan
        return g["label"].corr(g["pred"], method="spearman")

    return df.groupby("date").apply(_ic, include_groups=False).dropna()


def evaluate_candidate(name: str, estimator, panel: pd.DataFrame, splits: list[Split],
                       fcols: list[str]) -> CandidateResult:
    labeled = panel[panel["label"].notna()]
    fold_ics, all_ics = [], []
    for s in splits:
        train = labeled[labeled["date"] <= s.train_end]
        test = labeled[(labeled["date"] >= s.test_start) & (labeled["date"] <= s.test_end)]
        if train.empty or test.empty:
            continue
        model = clone(estimator)
        model.fit(train[fcols], train["label"])
        scored = test[["date", "label"]].copy()
        scored["pred"] = model.predict(test[fcols])
        ics = weekly_rank_ic(scored)
        fold_ics.append(float(ics.mean()))
        all_ics.append(ics)
    combined = pd.concat(all_ics) if all_ics else pd.Series(dtype=float)
    mean_ic = float(combined.mean()) if not combined.empty else float("nan")
    return CandidateResult(name=name, mean_ic=mean_ic, fold_ics=fold_ics,
                           n_test_weeks=int(len(combined)))
