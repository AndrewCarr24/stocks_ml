from __future__ import annotations

import numpy as np
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

        y = y.rename("label") if hasattr(y, "rename") else y
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


BASELINE_NAMES = ("zero", "momentum")


def get_candidates(cfg) -> dict:
    return {"zero": ZeroForecast(), "momentum": MomentumRank(),
            "xgb": make_xgb(), "automl": AutoMLRegressor()}
