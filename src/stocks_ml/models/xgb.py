"""The champion's model: depth-3 XGBoost with a purged, time-ordered early stop.

selection.MODEL_PARAMS fixes the hyperparameters (never searched — the
procedure card treats tuning as noise); selection.fixed() supplies the
early-stopping settings. WeekBootstrapEstimator (models/replication.py)
wraps K copies of this class for the ensemble.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


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
