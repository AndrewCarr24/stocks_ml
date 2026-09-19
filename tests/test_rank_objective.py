"""Learning-to-rank (2026-09-18): relevance grades, the ranker's purged tail stop, the recipe keys, the grade label."""
import numpy as np
import pandas as pd
import pytest

from stocks_ml.models.xgb import (GRADE_CUTS, RANK_PARAMS, TimeTailEarlyStopRanker, dated_features, estimator_for,
                                  relevance_grades)


def test_relevance_grades_are_top_heavy_within_the_week():
    dates = np.repeat(pd.to_datetime(["2010-01-08", "2010-01-15"]), 100)
    y = np.concatenate([np.arange(100), np.arange(100)[::-1]]).astype(float)
    g = relevance_grades(y, dates)
    first = g[:100]
    assert first[99] == 4 and first[98] == 4 and first[97] == 3 and first[95] == 3 and first[94] == 2 and first[90] == 2
    assert first[89] == 1 and first[75] == 1 and first[74] == 0 and first[0] == 0
    assert (g[100:] == first[::-1]).all()                       # per week, not across weeks


def test_ranker_fits_on_week_groups_with_a_purged_tail_and_scores():
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2010-01-08", periods=30, freq="W-FRI")
    rows = pd.DataFrame({"date": np.repeat(weeks, 40), "f_a": rng.normal(size=1200), "f_b": rng.normal(size=1200)})
    y = pd.Series(rows["f_a"] * 0.5 + rng.normal(size=1200) * 0.1)
    X = dated_features(rows, ["f_a", "f_b"])
    m = TimeTailEarlyStopRanker(n_estimators=30, max_depth=2, learning_rate=0.3, eval_fraction=0.2,
                                early_stop_purge_days=10, **RANK_PARAMS).fit(X, y)
    assert m.early_stop_validation_dates_.min() > m.early_stop_train_dates_.max() + pd.Timedelta(days=9)
    s = m.predict(X.iloc[:40])
    assert np.corrcoef(s, rows["f_a"].iloc[:40])[0, 1] > 0.5           # ranks by the signal
    from sklearn.base import clone
    c = clone(m).fit(X, y)                                              # the ensemble clones it: get_params round-trips
    assert c.get_params()["eval_metric"] == "ndcg@10" and c.get_params()["objective"] == "rank:ndcg"
    with pytest.raises(ValueError, match="dated"):
        TimeTailEarlyStopRanker(**RANK_PARAMS).fit(rows[["f_a", "f_b"]], y)


def test_estimator_follows_the_objective_and_recipe_accepts_rank_params():
    from stocks_ml.models.xgb import TimeTailEarlyStopXGB
    from stocks_ml.train import recipe
    fixed = dict(n_jobs=1, random_state=0, eval_fraction=0.1, early_stopping_rounds=5, early_stop_purge_days=10,
                 early_stop_metric="weekly_spearman")
    assert isinstance(estimator_for({"max_depth": 3}, fixed), TimeTailEarlyStopXGB)
    assert isinstance(estimator_for({"max_depth": 3, **RANK_PARAMS}, fixed), TimeTailEarlyStopRanker)
    r = recipe("label_4w_sector_rank", 8, params={"objective": "rank:ndcg", "lambdarank_num_pair_per_sample": "10"})
    assert r["params"] == {"objective": "rank:ndcg", "lambdarank_num_pair_per_sample": 10}
    with pytest.raises(ValueError):
        recipe("label_4w_sector_rank", 8, params={"objective_x": "rank:ndcg"})


def test_sector_grade_label_grades_the_top_of_the_week():
    from stocks_ml.features.panel import LABEL_TRANSFORMS, sector_grade_label
    n = 200
    fwd = pd.Series(np.arange(n, dtype=float) / n)
    date = pd.Series(pd.Timestamp("2010-01-08")).repeat(n).reset_index(drop=True)
    sector = pd.Series(["T"] * n)
    g = sector_grade_label(fwd, date, sector)
    assert g.iloc[-1] == 4 and g.iloc[-4] == 4 and g.iloc[-5] == 3 and g.iloc[-11] == 2 and g.iloc[-50] == 1 and g.iloc[0] == 0
    assert LABEL_TRANSFORMS["label_4w_sector_grade"] is sector_grade_label
    fwd2 = fwd.copy(); fwd2.iloc[3] = np.nan
    assert np.isnan(sector_grade_label(fwd2, date, sector).iloc[3])


def test_decile_relevance_covers_the_whole_ordering():
    dates = np.repeat(pd.to_datetime(["2010-01-08"]), 100)
    g = relevance_grades(np.arange(100, dtype=float), dates, scheme="decile")
    assert g.min() == 0 and g.max() == 9 and (np.bincount(g.astype(int)) == 10).all()


def test_volatility_controlled_grade_label_grades_within_terciles():
    from stocks_ml.features.panel import sector_grade_vol_label
    n = 300
    fwd = pd.Series(np.arange(n, dtype=float))
    date = pd.Series(pd.Timestamp("2010-01-08")).repeat(n).reset_index(drop=True)
    sector = pd.Series(["T"] * n)
    vol = pd.Series(np.tile([-0.9, 0.0, 0.9], n // 3))                # three volatility terciles interleaved
    g = sector_grade_vol_label(fwd, date, sector, vol=vol)
    assert (g == 4).sum() == 6 and (g >= 1).sum() == 75                # 2% and 25% of EACH tercile of 100
    assert g[vol == -0.9].max() == 4 and g[vol == 0.9].max() == 4    # the calm tercile has its own top
    with pytest.raises(ValueError, match="volatility"):
        sector_grade_vol_label(fwd, date, sector)
