import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone

from stocks_ml.models.xgb import TimeTailEarlyStopXGB, dated_features


def test_time_tail_early_stop_xgb_clone_roundtrips_eval_fraction():
    est = TimeTailEarlyStopXGB(eval_fraction=0.25, max_depth=4)
    cloned = clone(est)
    params = cloned.get_params()
    assert params["eval_fraction"] == 0.25
    assert params["max_depth"] == 4


def _tail_flip_xy(n=300, seed=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f_a": rng.normal(size=n), "f_b": rng.normal(size=n)})
    y = pd.Series(2.0 * X["f_a"], name="label")
    y.iloc[int(n * 0.9):] *= -1  # last 10% wildly different (flipped sign)
    return X, y


def test_time_tail_early_stop_xgb_fits_and_predicts_finite_values():
    X, y = _tail_flip_xy()
    model = TimeTailEarlyStopXGB(n_estimators=200, max_depth=3, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(X, y)
    assert model.get_params()["eval_fraction"] == 0.1
    preds = model.predict(X)
    assert np.all(np.isfinite(preds))
    assert model.best_iteration is not None  # early stopping used the eval_set


def test_time_tail_xgb_uses_complete_dates_and_inner_purge():
    dates = pd.date_range("2022-01-07", periods=20, freq="W-FRI")
    frame = pd.DataFrame({
        "date": np.repeat(dates, 4),
        "f_x": np.tile(np.arange(4, dtype=float), len(dates)),
    })
    y = pd.Series(np.random.default_rng(1).normal(size=len(frame)), index=frame.index)
    model = TimeTailEarlyStopXGB(
        n_estimators=10, max_depth=2, eval_fraction=0.2,
        early_stopping_rounds=3, early_stop_purge_days=10, random_state=0,
    )
    model.fit(dated_features(frame, ["f_x"]), y)
    train_dates = pd.DatetimeIndex(model.early_stop_train_dates_)
    validation_dates = pd.DatetimeIndex(model.early_stop_validation_dates_)
    assert validation_dates.nunique() == 4
    assert all((validation_dates == d).sum() == 4 for d in validation_dates.unique())
    assert validation_dates.min() - train_dates.max() >= pd.Timedelta(days=10)


def test_xgb_learns_signal_and_handles_nan():
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame({"f_mom_26w": rng.normal(size=n), "f_other": rng.normal(size=n)})
    y = pd.Series(2.0 * X["f_mom_26w"] + rng.normal(0, 0.1, n), name="label")
    X.loc[::7, "f_other"] = np.nan
    model = TimeTailEarlyStopXGB(n_estimators=300, max_depth=3, learning_rate=0.1,
                                 random_state=0).fit(X, y)
    r = np.corrcoef(model.predict(X), y)[0, 1]
    assert r > 0.8


def test_dated_features_carries_dates_and_only_feature_columns():
    frame = pd.DataFrame({"date": pd.to_datetime(["2022-01-07", "2022-01-14"]),
                          "ticker": ["A", "B"], "f_x": [1.0, 2.0], "label": [0.1, 0.2]})
    X = dated_features(frame, ["f_x"])
    assert list(X.columns) == ["f_x"]
    assert list(pd.DatetimeIndex(X.attrs["dates"])) == list(frame["date"])
