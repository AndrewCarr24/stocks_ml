import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from stocks_ml.models.cv import CandidateResult, evaluate_candidate, make_splits, weekly_rank_ic


def _panel(n_weeks=60, n_tickers=10, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-08", periods=n_weeks, freq="W-FRI")
    rows = []
    for d in dates:
        signal = rng.normal(0, 1, n_tickers)
        noise = rng.normal(0, 0.1, n_tickers)
        for i in range(n_tickers):
            rows.append({"date": d, "ticker": f"T{i}", "f_signal": signal[i],
                         "f_junk": rng.normal(), "label": signal[i] + noise[i]})
    return pd.DataFrame(rows)


class Oracle(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["f_signal"].to_numpy()


def test_make_splits_no_overlap_and_purge():
    dates = pd.DatetimeIndex(pd.date_range("2020-01-03", periods=100, freq="W-FRI"))
    splits = make_splits(dates, n_folds=4, purge_days=10, holdout_weeks=10)
    assert len(splits) == 4
    for s in splits:
        assert (s.test_start - s.train_end).days >= 10
    for a, b in zip(splits, splits[1:]):
        assert a.test_end < b.test_start
        assert b.train_end > a.train_end          # expanding
    assert splits[-1].test_end <= dates[-11]       # holdout untouched


def test_weekly_rank_ic_perfect_and_random():
    df = _panel()
    df["pred"] = df["label"]
    assert weekly_rank_ic(df).mean() > 0.99
    rng = np.random.default_rng(0)
    df["pred"] = rng.normal(size=len(df))
    assert abs(weekly_rank_ic(df).mean()) < 0.2


def test_evaluate_candidate_oracle_beats_noise():
    panel = _panel()
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    splits = make_splits(dates, n_folds=3, purge_days=10, holdout_weeks=0)
    res = evaluate_candidate("oracle", Oracle(), panel, splits, ["f_signal", "f_junk"])
    assert isinstance(res, CandidateResult)
    assert res.mean_ic > 0.9
    assert len(res.fold_ics) == 3
