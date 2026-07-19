from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor


class ZeroForecast(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MomentumRank(BaseEstimator, RegressorMixin):
    def __init__(self, column: str = "f_mom_26w"):
        self.column = column

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[self.column].to_numpy()


class AutoMLRegressor(BaseEstimator, RegressorMixin):
    """Adapter over AndrewCarr24/automl_tool: AutoML(X, y, outcome_name).fit_pipeline().

    The fitted GridSearchCV/Pipeline lands on .fitted_pipeline. If the installed
    version's import path or attribute names differ from what Task 1 Step 6 found,
    fix them HERE ONLY — the fit/predict contract must not change.
    """

    def fit(self, X, y):
        from automl_tool.automl import AutoML

        X = pd.DataFrame(X).copy()
        y = pd.Series(np.asarray(y), index=X.index, name="label")
        self.automl_ = AutoML(X, y, "label")
        self.automl_.fit_pipeline()
        return self

    def predict(self, X):
        return np.asarray(self.automl_.fitted_pipeline.predict(X))

    def best_estimator(self):
        """Unfitted-cloneable inner estimator for cheap refits (Task 12)."""
        fitted = self.automl_.fitted_pipeline
        return getattr(fitted, "best_estimator_", fitted)


def make_xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=50, n_jobs=-1, random_state=0,
    )


class TimeTailEarlyStopXGB(XGBRegressor):
    """XGBRegressor with early stopping on the LAST eval_fraction of rows (no shuffle).

    Panel rows are date-major sorted, so the positional tail is the most recent
    data — a time-ordered validation split (never random: random splits leak
    future rows into the stopping decision on temporal data)."""

    def __init__(self, eval_fraction: float = 0.1, early_stopping_rounds: int = 20, **kwargs):
        self.eval_fraction = eval_fraction
        super().__init__(early_stopping_rounds=early_stopping_rounds, **kwargs)

    def _wrapper_params(self) -> set:
        # eval_fraction is wrapper-only bookkeeping, not a native XGBoost booster
        # parameter — without this override it gets forwarded to the C++ learner
        # and triggers a "Parameters: { eval_fraction } are not used" warning.
        return super()._wrapper_params() | {"eval_fraction"}

    def fit(self, X, y):
        n = len(X)
        cut = max(1, int(n * (1 - self.eval_fraction)))
        Xtr, Xva = X.iloc[:cut], X.iloc[cut:]
        ytr, yva = y.iloc[:cut], y.iloc[cut:]
        super().fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return self


def make_xgb_tuned(models_dir="models"):
    """TimeTailEarlyStopXGB from models/xgb_tuned.json; None if the file doesn't exist."""
    path = Path(models_dir) / "xgb_tuned.json"
    if not path.exists():
        return None
    return TimeTailEarlyStopXGB(**json.loads(path.read_text()))


BASELINE_NAMES = ("zero", "momentum")


def get_candidates(cfg, models_dir="models") -> dict:
    candidates = {"zero": ZeroForecast(), "momentum": MomentumRank(),
                  "xgb": make_xgb(), "automl": AutoMLRegressor()}
    tuned = make_xgb_tuned(models_dir)
    if tuned is not None:
        candidates["xgb_tuned"] = tuned
    return candidates
