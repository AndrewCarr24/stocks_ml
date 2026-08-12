from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from xgboost import XGBClassifier, XGBRanker, XGBRegressor


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


def _time_tail_masks(X, eval_fraction: float, purge_days: int):
    """Return complete-date train/validation masks when date metadata exists."""
    dates = X.attrs.get("dates") if hasattr(X, "attrs") else None
    if dates is None:
        n = len(X)
        cut = max(1, int(n * (1 - eval_fraction)))
        return np.arange(n) < cut, np.arange(n) >= cut, None
    dates = pd.DatetimeIndex(dates)
    unique = dates.unique().sort_values()
    n_eval = max(1, int(np.ceil(len(unique) * eval_fraction)))
    val_start = unique[-n_eval]
    train_end = val_start - pd.Timedelta(days=purge_days)
    tr_mask, va_mask = dates <= train_end, dates >= val_start
    if not tr_mask.any() or not va_mask.any():
        raise ValueError("time-tail early-stop split has an empty train or validation block")
    return tr_mask, va_mask, dates


class TimeTailEarlyStopLGBM(LGBMRegressor):
    """LGBMRegressor with early stopping on the LAST eval_fraction of rows (no shuffle).

    Same time-ordered validation rationale as TimeTailEarlyStopXGB: a random split
    would leak future rows into the stopping decision on temporal data. verbosity=-1
    keeps LightGBM silent (the 0-warnings constraint)."""

    def __init__(self, eval_fraction: float = 0.1, early_stopping_rounds: int = 20,
                 early_stop_purge_days: int = 10, verbosity: int = -1, **kwargs):
        self.eval_fraction = eval_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stop_purge_days = early_stop_purge_days
        super().__init__(verbosity=verbosity, **kwargs)

    def fit(self, X, y):
        import lightgbm as lgb

        tr_mask, va_mask, dates = _time_tail_masks(
            X, self.eval_fraction, self.early_stop_purge_days)
        Xtr, Xva = X.loc[tr_mask], X.loc[va_mask]
        ytr, yva = y.loc[tr_mask], y.loc[va_mask]
        if dates is not None:
            self.early_stop_train_dates_ = dates[tr_mask]
            self.early_stop_validation_dates_ = dates[va_mask]
        # lightgbm 4.4+ accepts eval_X/eval_y single frames (the modern form of
        # the older eval_set list-of-tuples API).
        super().fit(Xtr, ytr, eval_X=Xva, eval_y=yva,
                    callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                               lgb.log_evaluation(period=0)])
        return self


class TimeTailEarlyStopCatBoost(BaseEstimator, RegressorMixin):
    """CatBoostRegressor with early stopping on the LAST eval_fraction of rows (no shuffle).

    Composed (not subclassed) because CatBoost's sklearn surface does not clone
    cleanly via inheritance. Same time-ordered validation rationale as the XGB/LGBM
    wrappers. verbose=False + allow_writing_files=False keep it silent and side-effect
    free (no catboost_info/ dir, satisfying the 0-warnings constraint)."""

    def __init__(self, eval_fraction: float = 0.1, early_stopping_rounds: int = 20,
                 early_stop_purge_days: int = 10,
                 depth: int = 6, learning_rate: float = 0.1, l2_leaf_reg: float = 3.0,
                 iterations: int = 1500, random_state: int = 0):
        self.eval_fraction = eval_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stop_purge_days = early_stop_purge_days
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, X, y):
        from catboost import CatBoostRegressor

        tr_mask, va_mask, dates = _time_tail_masks(
            X, self.eval_fraction, self.early_stop_purge_days)
        if dates is not None:
            self.early_stop_train_dates_ = dates[tr_mask]
            self.early_stop_validation_dates_ = dates[va_mask]
        self.model_ = CatBoostRegressor(
            depth=self.depth, learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg, iterations=self.iterations,
            random_state=self.random_state, allow_writing_files=False, verbose=False,
        )
        self.model_.fit(X.loc[tr_mask], y.loc[tr_mask],
                eval_set=(X.loc[va_mask], y.loc[va_mask]),
                        early_stopping_rounds=self.early_stopping_rounds, use_best_model=True)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X))


class WeekGroupedXGBRanker(XGBRanker):
    """Learning-to-rank XGBoost: each week is one query group (rank:ndcg).

    Directly optimizes the ordering that top-k strategies consume, closing the
    train-on-RMSE / select-on-rank-IC mismatch of the regression candidates.
    Rows must be date-major sorted (the panel's layout) so qid groups are
    contiguous. Time-tail early stopping on NDCG, same purge rationale as
    TimeTailEarlyStopXGB. Wave-1 defaults reuse the tuned regressor's tree
    shape; the learning rate is conventional for LTR — retune if promising.

    Raw LTR scores are only comparable within a group, so predict() centers
    each call on its median: production paths predict one cross-section at a
    time, making "positive = above-median conviction" hold for select_top_k.
    (Multi-week predict calls get a constant global shift, which cannot change
    within-week orderings used by rank-IC scoring.)"""

    def __init__(self, eval_fraction: float = 0.1, early_stopping_rounds: int = 75,
                 early_stop_purge_days: int = 10, **kwargs):
        self.eval_fraction = eval_fraction
        self.early_stop_purge_days = early_stop_purge_days
        defaults = dict(objective="rank:ndcg", eval_metric="ndcg@8",
                        n_estimators=1500, learning_rate=0.05, max_depth=2,
                        min_child_weight=20, subsample=0.8, colsample_bytree=0.5,
                        reg_lambda=0.25)
        defaults.update(kwargs)
        super().__init__(early_stopping_rounds=early_stopping_rounds, **defaults)

    def _wrapper_params(self) -> set:
        return super()._wrapper_params() | {"eval_fraction", "early_stop_purge_days"}

    def fit(self, X, y):
        tr_mask, va_mask, dates = _time_tail_masks(
            X, self.eval_fraction, self.early_stop_purge_days)
        if dates is None:
            # generic sklearn callers: one flat group, ordered split
            dates = pd.DatetimeIndex([pd.Timestamp("2000-01-01")] * len(X))
        qid = pd.factorize(dates, sort=True)[0]
        # NDCG needs non-negative integer relevance, not continuous returns:
        # grade each stock by its within-week label quintile (0 = worst .. 4 = best)
        frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "date": np.asarray(dates)})
        pct = frame.groupby("date")["y"].rank(pct=True)
        grades = (np.ceil(pct * 5).astype(int) - 1).clip(0, 4)
        self.early_stop_train_dates_ = dates[tr_mask]
        self.early_stop_validation_dates_ = dates[va_mask]
        super().fit(X.loc[tr_mask], grades[tr_mask], qid=qid[tr_mask],
                    eval_set=[(X.loc[va_mask], grades[va_mask])],
                    eval_qid=[qid[va_mask]], verbose=False)
        return self

    def predict(self, X):
        scores = np.asarray(super().predict(X), dtype=float)
        return scores - np.median(scores)


class TopQuintileClassifier(BaseEstimator, RegressorMixin):
    """P(stock lands in the week's top label quintile), trained on extremes.

    Per training week, the top `quantile` of labels becomes class 1 and the
    bottom `quantile` class 0; the middle is dropped (standard practice — the
    middle of the cross-section is mostly noise). XGBClassifier with
    time-tail early stopping on AUC. predict() returns the raw probability of
    the top class, so strategies can apply a calibrated confidence floor
    (e.g. ConfidenceTopK: only probabilities above 0.5 count as conviction)."""

    def __init__(self, quantile: float = 0.2, eval_fraction: float = 0.1,
                 early_stopping_rounds: int = 75, early_stop_purge_days: int = 10,
                 n_estimators: int = 1500, learning_rate: float = 0.05,
                 max_depth: int = 3, min_child_weight: int = 20,
                 subsample: float = 0.8, colsample_bytree: float = 0.5,
                 reg_lambda: float = 0.25, random_state: int = 0):
        self.quantile = quantile
        self.eval_fraction = eval_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stop_purge_days = early_stop_purge_days
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def _extreme_classes(self, y: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
        frame = pd.DataFrame({"y": np.asarray(y, dtype=float),
                              "date": np.asarray(dates)})
        hi = frame.groupby("date")["y"].transform(lambda v: v.quantile(1 - self.quantile))
        lo = frame.groupby("date")["y"].transform(lambda v: v.quantile(self.quantile))
        cls = pd.Series(np.nan, index=frame.index)
        cls[frame["y"] >= hi] = 1.0
        cls[frame["y"] <= lo] = 0.0
        return cls

    def fit(self, X, y):
        tr_mask, va_mask, dates = _time_tail_masks(
            X, self.eval_fraction, self.early_stop_purge_days)
        if dates is None:
            dates = pd.DatetimeIndex([pd.Timestamp("2000-01-01")] * len(X))
        cls = self._extreme_classes(pd.Series(np.asarray(y)), pd.DatetimeIndex(dates))
        tr = tr_mask & cls.notna().to_numpy()
        va = va_mask & cls.notna().to_numpy()
        if not tr.any() or not va.any():
            raise ValueError("top-quintile classifier has an empty train or validation block")
        self.early_stop_train_dates_ = pd.DatetimeIndex(dates)[tr]
        self.early_stop_validation_dates_ = pd.DatetimeIndex(dates)[va]
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, min_child_weight=self.min_child_weight,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda, random_state=self.random_state,
            eval_metric="auc", early_stopping_rounds=self.early_stopping_rounds)
        self.model_.fit(X.loc[tr], cls[tr].astype(int),
                        eval_set=[(X.loc[va], cls[va].astype(int))], verbose=False)
        return self

    def predict(self, X):
        return self.model_.predict_proba(X)[:, 1]


class ICElasticNet(BaseEstimator, RegressorMixin):
    """NaN-safe ElasticNet: median-impute → standardize → ElasticNet.

    The panel carries NaNs (sparse fundamentals/insider/short features); plain
    ElasticNet raises on them, so imputation is mandatory. Selection is by rank IC
    in our evaluate_candidate — never RMSE — so this cannot repeat automl_tool's
    constant-predictor failure (a constant has zero IC and is gate-excluded)."""

    def __init__(self, alpha: float = 1e-4, l1_ratio: float = 0.5,
                 max_iter: int = 10000, tol: float = 1e-2):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        # tol is intentionally loose: only the RANK of predictions matters
        # downstream, so near-convergence is fully adequate and avoids
        # ConvergenceWarnings on ill-conditioned columns.
        self.tol = tol

    def _build(self):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Two-stage impute: median for normal NaNs, then constant 0 for any column
        # that is ENTIRELY NaN within a fold (e.g. short-interest features before
        # 2018). keep_empty_features avoids SimpleImputer's drop-and-warn on all-NaN
        # columns, keeping the feature matrix shape stable across folds.
        return Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("enet", ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                max_iter=self.max_iter, tol=self.tol, random_state=0)),
        ])

    def fit(self, X, y):
        self.pipeline_ = self._build().fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.pipeline_.predict(X))


class EnsembleCandidate(BaseEstimator, RegressorMixin):
    """Averages z-scored predictions of its member estimators.

    Each member's prediction vector is standardized (mean 0, unit std) within the
    scored batch before averaging, so members with different output scales (a linear
    model vs. a boosted tree) contribute equally to the rank the strategy consumes.
    Only the rank of the averaged score matters downstream, so z-scoring is a pure
    scale-alignment step. Members are cloned at fit so the ensemble is reusable."""

    def __init__(self, estimators: list | None = None):
        self.estimators = estimators

    def fit(self, X, y):
        self.fitted_ = [clone(e).fit(X, y) for e in (self.estimators or [])]
        return self

    @staticmethod
    def _z(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        sd = v.std()
        return (v - v.mean()) / sd if sd > 0 else v - v.mean()

    def predict(self, X):
        if not self.fitted_:
            return np.zeros(len(X))
        return np.mean([self._z(e.predict(X)) for e in self.fitted_], axis=0)


def make_xgb_tuned(models_dir="models"):
    """TimeTailEarlyStopXGB from models/xgb_tuned.json; None if the file doesn't exist."""
    path = Path(models_dir) / "xgb_tuned.json"
    if not path.exists():
        return None
    return TimeTailEarlyStopXGB(**json.loads(path.read_text()))


# family name -> (params-file stem, wrapper class) for the tunable zoo members.
_TUNED_FAMILIES = {
    "xgb": ("xgb_tuned", TimeTailEarlyStopXGB),
    "lgbm": ("lgbm_tuned", TimeTailEarlyStopLGBM),
    "catboost": ("catboost_tuned", TimeTailEarlyStopCatBoost),
    "enet": ("enet_tuned", ICElasticNet),
}


def make_tuned(family: str, models_dir="models"):
    """Reconstruct a tuned candidate for `family` from its params file; None if absent.

    Prefers {family}_optuna.json (the CV-selected Optuna refinement) over
    {family}_tuned.json (random search) when both exist. Both searches select by
    the same pre-holdout purged walk-forward CV metric; the holdout is untouched."""
    stem, klass = _TUNED_FAMILIES[family]
    d = Path(models_dir)
    for candidate_path in (d / f"{family}_optuna.json", d / f"{stem}.json"):
        if candidate_path.exists():
            return klass(**json.loads(candidate_path.read_text()))
    return None


BASELINE_NAMES = ("zero", "momentum")


def get_candidates(cfg, models_dir="models") -> dict:
    candidates = {"zero": ZeroForecast(), "momentum": MomentumRank(),
                  "xgb": make_xgb(), "automl": AutoMLRegressor()}
    tuned = {}
    for family in _TUNED_FAMILIES:
        est = make_tuned(family, models_dir)
        if est is not None:
            tuned[f"{family}_tuned"] = est
    candidates.update(tuned)
    # Ensemble of every tuned member present — only meaningful with 2+ members.
    if len(tuned) >= 2:
        candidates["ensemble"] = EnsembleCandidate(
            estimators=[make_tuned(f, models_dir) for f in _TUNED_FAMILIES
                        if f"{f}_tuned" in tuned])
    return candidates
