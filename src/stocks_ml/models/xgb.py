"""The champion's model: depth-3 XGBoost with a purged, time-ordered early stop.

selection.MODEL_PARAMS fixes the hyperparameters (never searched — the
procedure card treats tuning as noise); selection.fixed() supplies the
early-stopping settings. WeekBootstrapEstimator (models/replication.py)
wraps K copies of this class for the ensemble.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRanker, XGBRegressor

GRADE_CUTS = (0.02, 0.05, 0.10, 0.25)     # top-heavy relevance: top 2% -> 4, 5% -> 3, 10% -> 2, 25% -> 1, else 0
RANK_PARAMS = {"objective": "rank:ndcg", "lambdarank_pair_method": "topk", "lambdarank_num_pair_per_sample": 10,
               "ndcg_exp_gain": True, "grades": "top"}
# grades: "top" = GRADE_CUTS (top 2% -> 4 ... top 25% -> 1); "decile" = 0..9 by within-week decile (the whole
# ordering; pair with ndcg_exp_gain=False for a linear gain)
# The learning-to-rank recipe keys (2026-09-18, the owner's "penalise the top-10 ranking more"): a recipe
# naming objective=rank:* is fit by TimeTailEarlyStopRanker — pairs formed within the week (qid), the
# top-k pair method so misordering the top of the week costs most, NDCG@10 on the purged tail to stop.


def relevance_grades(y, dates, cuts=GRADE_CUTS, scheme: str = "top") -> np.ndarray:
    """Integer relevance per row from a continuous within-week label. "top":
    the top 2% of the week grade 4, top 5% 3, top 10% 2, top 25% 1, the rest
    0. "decile": 9 for the top tenth down to 0 for the bottom tenth — the
    whole ordering, evenly."""
    y = pd.Series(np.asarray(y, float))
    pct = y.groupby(np.asarray(dates)).rank(ascending=False, pct=True)
    if scheme == "decile":
        bucket = np.ceil(pct.to_numpy() * 10 - 1e-9)             # 1 = the top tenth ... 10 = the bottom tenth
        return np.clip(10 - bucket, 0, 9).astype(float)
    out = np.zeros(len(y))
    for g, c in zip(range(len(cuts), 0, -1), cuts):
        out = np.where((pct <= c) & (out == 0), g, out)
    return out.astype(float)


def estimator_for(params: dict, fixed: dict):
    """The estimator a recipe's params call for: the ranker when the
    objective is rank:*, else the champion's regressor."""
    if str(params.get("objective", "")).startswith("rank:"):
        return TimeTailEarlyStopRanker(**params, **fixed)
    return TimeTailEarlyStopXGB(**params, **fixed)


class TimeTailEarlyStopRanker(XGBRanker):
    """XGBRanker on within-week groups with the same purged, time-ordered
    early stop as TimeTailEarlyStopXGB, scored by NDCG@10 on the tail. The
    continuous label is turned into relevance grades (relevance_grades)."""

    def __init__(self, eval_fraction: float = 0.1, early_stopping_rounds: int = 75,
                 early_stop_purge_days: int = 10, early_stop_metric: str = "ndcg@10", grades: str = "top", **kwargs):
        self.eval_fraction = eval_fraction
        self.early_stop_purge_days = early_stop_purge_days
        self.early_stop_metric = early_stop_metric
        self.grades = grades
        metric = early_stop_metric if early_stop_metric.startswith("ndcg") else "ndcg@10"
        kwargs.pop("eval_metric", None)        # sklearn.clone passes it back from get_params(); ours wins
        super().__init__(early_stopping_rounds=early_stopping_rounds, eval_metric=metric, **kwargs)

    def _wrapper_params(self) -> set:
        return super()._wrapper_params() | {"eval_fraction", "early_stop_purge_days", "early_stop_metric", "grades"}

    def fit(self, X, y):
        dates = X.attrs.get("dates") if hasattr(X, "attrs") else None
        if dates is None:
            raise ValueError("the ranker needs dated_features frames (X.attrs['dates'])")
        dates = pd.DatetimeIndex(dates)
        qid = pd.factorize(dates, sort=True)[0]          # the walk's frames are date-sorted; a bootstrap keeps duplicates adjacent
        rel = relevance_grades(y, dates.to_numpy(), scheme=self.grades)
        unique = dates.unique().sort_values()
        n_eval = max(1, int(np.ceil(len(unique) * self.eval_fraction)))
        val_start = unique[-n_eval]
        train_end = val_start - pd.Timedelta(days=self.early_stop_purge_days)
        tr, va = (dates <= train_end), (dates >= val_start)
        if not tr.any() or not va.any():
            raise ValueError("time-tail early-stop split has an empty train or validation block")
        self.early_stop_train_dates_ = dates[tr]
        self.early_stop_validation_dates_ = dates[va]
        super().fit(X.loc[tr], rel[tr], qid=qid[tr], eval_set=[(X.loc[va], rel[va])], eval_qid=[qid[va]], verbose=False)
        return self


class TimeTailEarlyStopXGB(XGBRegressor):
    """XGBRegressor with early stopping on the LAST eval_fraction of rows (no shuffle).

    Panel rows are date-major sorted, so the positional tail is the most recent
    data — a time-ordered validation split (never random: random splits leak
    future rows into the stopping decision on temporal data)."""

    def __init__(self, eval_fraction: float = 0.1, early_stopping_rounds: int = 75,
                 early_stop_purge_days: int = 10,
                 early_stop_metric: str = "weekly_spearman", **kwargs):
        self.eval_fraction = eval_fraction
        self.early_stop_purge_days = early_stop_purge_days
        self.early_stop_metric = early_stop_metric
        super().__init__(early_stopping_rounds=early_stopping_rounds, **kwargs)

    def _wrapper_params(self) -> set:
        # eval_fraction is wrapper-only bookkeeping, not a native XGBoost booster
        # parameter — without this override it gets forwarded to the C++ learner
        # and triggers a "Parameters: { eval_fraction } are not used" warning.
        return super()._wrapper_params() | {
            "eval_fraction", "early_stop_purge_days", "early_stop_metric",
        }

    def fit(self, X, y):
        dates = X.attrs.get("dates") if hasattr(X, "attrs") else None
        if dates is not None:
            dates = pd.DatetimeIndex(dates)
            unique = dates.unique().sort_values()
            n_eval = max(1, int(np.ceil(len(unique) * self.eval_fraction)))
            val_start = unique[-n_eval]
            train_end = val_start - pd.Timedelta(days=self.early_stop_purge_days)
            tr_mask = dates <= train_end
            va_mask = dates >= val_start
            if not tr_mask.any() or not va_mask.any():
                raise ValueError("time-tail early-stop split has an empty train or validation block")
            Xtr, Xva = X.loc[tr_mask], X.loc[va_mask]
            ytr, yva = y.loc[tr_mask], y.loc[va_mask]
            self.early_stop_train_dates_ = dates[tr_mask]
            self.early_stop_validation_dates_ = dates[va_mask]
        else:
            # Generic sklearn callers may not supply temporal metadata. Keep a
            # deterministic ordered fallback; production paths attach dates.
            n = len(X)
            cut = max(1, int(n * (1 - self.eval_fraction)))
            Xtr, Xva = X.iloc[:cut], X.iloc[cut:]
            ytr, yva = y.iloc[:cut], y.iloc[cut:]
        if self.early_stop_metric == "weekly_spearman" and dates is not None:
            from scipy.stats import rankdata

            validation_dates = pd.DatetimeIndex(self.early_stop_validation_dates_)
            groups = [np.flatnonzero(validation_dates == d)
                      for d in validation_dates.unique()]

            def negative_weekly_spearman(y_true, y_pred):
                ics = []
                for idx in groups:
                    true, pred = np.asarray(y_true)[idx], np.asarray(y_pred)[idx]
                    if len(idx) < 3 or np.unique(true).size < 2 or np.unique(pred).size < 2:
                        ics.append(0.0)
                        continue
                    ics.append(float(np.corrcoef(rankdata(true), rankdata(pred))[0, 1]))
                return -float(np.mean(ics))

            # XGBoost minimizes custom sklearn metrics, hence negative IC.
            self.set_params(eval_metric=negative_weekly_spearman)
        elif dates is not None and self.early_stop_metric != "rmse":
            raise ValueError("early_stop_metric must be 'weekly_spearman' or 'rmse'")
        try:
            super().fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        finally:
            # Keep the estimator cloneable/serializable; the fitted Booster has
            # already retained its stopping history and selected tree limit.
            self.set_params(eval_metric=None)
        return self


def dated_features(frame: pd.DataFrame, fcols: list[str]) -> pd.DataFrame:
    """Feature matrix carrying dates for time-ordered estimator internals."""
    X = frame[fcols].copy()
    X.attrs["dates"] = frame["date"].to_numpy()
    return X
