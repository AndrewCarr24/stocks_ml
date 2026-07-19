import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone

from stocks_ml.models.candidates import (
    TimeTailEarlyStopXGB, get_candidates, make_xgb_tuned,
)
from stocks_ml.models.tuning import SEARCH_SPACE, sample_configs, tune_xgb


def test_sample_configs_deterministic_under_seed():
    a = sample_configs(10, seed=5)
    b = sample_configs(10, seed=5)
    assert a == b


def test_sample_configs_n_unique_configs_from_the_space():
    configs = sample_configs(15, seed=1)
    assert len(configs) == 15
    assert len({tuple(sorted(c.items())) for c in configs}) == 15
    for c in configs:
        assert set(c) == set(SEARCH_SPACE)
        for k, v in c.items():
            assert v in SEARCH_SPACE[k]


def test_sample_configs_different_seeds_differ():
    a = sample_configs(10, seed=1)
    b = sample_configs(10, seed=2)
    assert a != b


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


def test_tune_xgb_writes_artifacts(synthetic_store, tiny_cfg, tmp_path):
    from stocks_ml.features.panel import build_panel

    build_panel(synthetic_store, tiny_cfg)
    out = tmp_path / "models"
    results = tune_xgb(synthetic_store, tiny_cfg, n_samples=3, out_dir=out)

    assert (out / "xgb_tuned.json").exists()
    assert (out / "tuning.md").exists()
    assert len(results) == 4  # 1 production reference + 3 sampled

    params = json.loads((out / "xgb_tuned.json").read_text())
    est = TimeTailEarlyStopXGB(**params)
    Xy_X, Xy_y = _tail_flip_xy(120)
    est.fit(Xy_X, Xy_y)
    preds = est.predict(Xy_X)
    assert np.all(np.isfinite(preds))

    text = (out / "tuning.md").read_text()
    assert "production reference" in text.lower()


def test_get_candidates_without_tuned_json_has_four(tiny_cfg, tmp_path):
    cands = get_candidates(tiny_cfg, models_dir=tmp_path / "models")
    assert set(cands) == {"zero", "momentum", "xgb", "automl"}


def test_get_candidates_includes_xgb_tuned_when_json_exists(tiny_cfg, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    params = {**{k: v[0] for k, v in SEARCH_SPACE.items()},
              "n_jobs": -1, "random_state": 0,
              "eval_fraction": 0.1, "early_stopping_rounds": 20}
    (models_dir / "xgb_tuned.json").write_text(json.dumps(params))

    cands = get_candidates(tiny_cfg, models_dir=models_dir)
    assert set(cands) == {"zero", "momentum", "xgb", "automl", "xgb_tuned"}
    clone(cands["xgb_tuned"])  # must be cloneable like the rest of the registry


def test_make_xgb_tuned_returns_none_when_missing(tmp_path):
    assert make_xgb_tuned(tmp_path / "models") is None
