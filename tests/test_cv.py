import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from stocks_ml.models.champion import _eligible
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
        assert s.train_end - s.train_start == pd.Timedelta(days=730) or (
            s.train_end - s.train_start == pd.Timedelta(days=731))
    for a, b in zip(splits, splits[1:]):
        assert a.test_end < b.test_start
        assert b.train_start > a.train_start      # fixed window rolls forward
        assert b.train_end > a.train_end
    assert splits[-1].test_end <= dates[-11]       # holdout untouched


def test_make_splits_eval_start_restricts_test_but_not_train():
    dates = pd.DatetimeIndex(pd.date_range("2020-01-03", periods=200, freq="W-FRI"))
    cut = pd.Timestamp("2023-01-01")
    splits = make_splits(dates, n_folds=4, purge_days=10, holdout_weeks=10,
                         eval_start=cut)
    # every scored (test) window starts at/after eval_start ...
    assert all(s.test_start >= cut for s in splits)
    # ... but training may still reach back before it (train_end < eval_start is fine)
    assert splits[0].train_end < cut


def test_make_splits_exact_two_year_training_calendars():
    dates = pd.DatetimeIndex(pd.date_range("2012-01-06", "2026-07-17", freq="W-FRI"))
    splits = make_splits(dates, n_folds=4, purge_days=10, holdout_weeks=104,
                         eval_start=pd.Timestamp("2015-03-01"), train_years=2)
    for split in splits:
        assert split.train_start == split.train_end - pd.DateOffset(years=2)
        assert split.train_end == split.test_start - pd.Timedelta(days=10)
    assert splits[-1].test_end < dates[-104]


def test_make_splits_rejects_incomplete_fold_calendar():
    dates = pd.DatetimeIndex(pd.date_range("2020-01-03", periods=10, freq="W-FRI"))
    with pytest.raises(ValueError, match="test weeks"):
        make_splits(dates, n_folds=7, purge_days=10, holdout_weeks=0)


def test_weekly_rank_ic_perfect_and_random():
    df = _panel()
    df["pred"] = df["label"]
    assert weekly_rank_ic(df).mean() > 0.99
    rng = np.random.default_rng(0)
    df["pred"] = rng.normal(size=len(df))
    assert abs(weekly_rank_ic(df).mean()) < 0.2


def test_weekly_rank_ic_rejects_partial_nonfinite_predictions():
    df = _panel(n_weeks=2)
    df["pred"] = df["label"]
    df.loc[df.index[0], "pred"] = np.nan
    ics = weekly_rank_ic(df)
    assert len(ics) == 1
    assert ics.index[0] == df["date"].iloc[-1]


def test_evaluate_candidate_oracle_beats_noise():
    panel = _panel()
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    splits = make_splits(dates, n_folds=3, purge_days=10, holdout_weeks=0)
    res = evaluate_candidate("oracle", Oracle(), panel, splits, ["f_signal", "f_junk"])
    assert isinstance(res, CandidateResult)
    assert res.mean_ic > 0.9
    assert len(res.fold_ics) == 3
    assert res.n_test_weeks == res.expected_test_weeks
    assert res.expected_folds == 3
    assert _eligible(res)


class OneConstantWeekPerFold(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        pred = X["f_signal"].to_numpy().copy()
        pred[:10] = 0.0
        return pred


def test_candidate_with_one_constant_week_is_ineligible():
    panel = _panel()
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    splits = make_splits(dates, n_folds=3, purge_days=10, holdout_weeks=0)
    res = evaluate_candidate("partial", OneConstantWeekPerFold(), panel, splits,
                             ["f_signal", "f_junk"])
    assert res.n_test_weeks < res.expected_test_weeks
    assert all(np.isnan(ic) for ic in res.fold_ics)
    assert not _eligible(res)


def test_evaluate_candidate_rejects_duplicate_stock_week_rows():
    panel = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    splits = make_splits(dates, n_folds=3, purge_days=10, holdout_weeks=0)
    with pytest.raises(ValueError, match="one labeled row"):
        evaluate_candidate("duplicate", Oracle(), panel, splits, ["f_signal", "f_junk"])


def test_evaluate_candidate_alternate_label_col():
    panel = _panel()
    # move the signal into a 4-week-style label column; weekly label all-NaN
    panel = panel.rename(columns={"label": "label_4w"})
    panel["label"] = np.nan
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    splits = make_splits(dates, 2, purge_days=42, holdout_weeks=0)
    res = evaluate_candidate("oracle4w", Oracle(), panel, splits,
                             ["f_signal", "f_junk"], label_col="label_4w")
    assert res.mean_ic > 0.8          # oracle ranks the 4w label near-perfectly
    assert res.n_test_weeks == res.expected_test_weeks
