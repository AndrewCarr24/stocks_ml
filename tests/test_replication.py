import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from stocks_ml.models.replication import WeekBootstrapEstimator, average_books


class RecordingEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, X, y):
        self.seen_X_ = X
        self.seen_y_ = np.asarray(y)
        return self

    def predict(self, X):
        return np.zeros(len(X))


def _dated_frame():
    dates = np.repeat(pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17",
                                      "2020-01-24"]), 3)
    X = pd.DataFrame({"f_a": np.arange(12.0), "f_b": np.arange(12.0) * 2})
    X.attrs["dates"] = dates
    return X, pd.Series(np.arange(12.0) * 10)


def test_bootstrap_resamples_whole_weeks_chronologically():
    X, y = _dated_frame()
    wrap = WeekBootstrapEstimator(RecordingEstimator(), bootstrap_seed=1).fit(X, y)
    inner = wrap.model_
    seen_dates = pd.to_datetime(inner.seen_X_.attrs["dates"])
    assert len(inner.seen_X_) == len(X)                     # same-size resample
    assert list(seen_dates) == sorted(seen_dates)           # chronological
    # whole weeks travel together: every sampled date contributes its full block
    for d, grp in inner.seen_X_.groupby(seen_dates):
        assert len(grp) % 3 == 0
    # rows stay aligned with labels (y = 10 * f_a everywhere)
    assert np.allclose(inner.seen_y_, inner.seen_X_["f_a"].to_numpy() * 10)


def test_bootstrap_is_seeded_and_seeds_differ():
    X, y = _dated_frame()
    a1 = WeekBootstrapEstimator(RecordingEstimator(), 1).fit(X, y).model_
    a2 = WeekBootstrapEstimator(RecordingEstimator(), 1).fit(X, y).model_
    b = WeekBootstrapEstimator(RecordingEstimator(), 2).fit(X, y).model_
    assert np.array_equal(a1.seen_X_.attrs["dates"], a2.seen_X_.attrs["dates"])
    assert not np.array_equal(a1.seen_X_.attrs["dates"], b.seen_X_.attrs["dates"])
    # the model's internal seed is aligned with the copy index
    assert a1.random_state == 1 and b.random_state == 2


def test_bootstrap_refuses_undated_frames():
    X = pd.DataFrame({"f_a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        WeekBootstrapEstimator(RecordingEstimator(), 1).fit(X, pd.Series([1.0, 2.0]))


def test_average_books_union_and_scale():
    idx = pd.to_datetime(["2020-01-03", "2020-01-10"])
    b1 = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.0]}, index=idx)
    b2 = pd.DataFrame({"A": [0.5, 0.0], "C": [0.5, 0.5]}, index=idx)
    avg = average_books([b1, b2])
    assert set(avg.columns) == {"A", "B", "C"}
    assert avg.loc[idx[0], "A"] == pytest.approx(0.5)
    assert avg.loc[idx[0], "B"] == pytest.approx(0.25)
    assert (avg.sum(axis=1) <= 1 + 1e-9).all()
