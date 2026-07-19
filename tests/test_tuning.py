import dataclasses
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone

from stocks_ml.models.candidates import (
    TimeTailEarlyStopXGB, get_candidates, make_xgb_tuned,
)
from stocks_ml.models.champion import _eligible
from stocks_ml.models.cv import CandidateResult
from stocks_ml.models.tuning import SEARCH_SPACE, sample_configs, select_best, tune_xgb


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


def test_tune_xgb_respects_train_sample_rows_truncation(synthetic_store, tiny_cfg, tmp_path):
    """run_training truncates via labeled.sort_values("date").tail(train_sample_rows)
    before building dates/splits (champion.py); tune_xgb must mirror this exactly or
    it would select hyperparameters on a different data window than the tournament
    scores candidates on whenever the (currently-dormant) knob is set."""
    from stocks_ml.features.panel import build_panel

    build_panel(synthetic_store, tiny_cfg)
    panel = synthetic_store.read("panel")
    full_min_date = panel[panel["label"].notna()]["date"].min()

    small_cfg = dataclasses.replace(tiny_cfg, train_sample_rows=300)  # ~half of 630 labeled rows
    out = tmp_path / "models"
    tune_xgb(synthetic_store, small_cfg, n_samples=2, out_dir=out)

    text = (out / "tuning.md").read_text()
    m = re.search(r"Training window: (\S+) \S+ \S+ \((\d+) labeled rows\)", text)
    assert m is not None, "tuning.md must report the training window it actually evaluated"
    truncated_min_date, n_rows = pd.Timestamp(m.group(1)), int(m.group(2))
    assert n_rows == 300
    assert truncated_min_date > full_min_date  # truncation must have dropped early rows


def _row(name, mean_ic, fold_ics, is_production=False, n_test_weeks=10, params=None):
    """Build a results-frame row the way tune_xgb does: eligibility computed via
    champion.py's _eligible on the underlying CandidateResult, not re-derived here."""
    eligible = _eligible(CandidateResult(name=name, mean_ic=mean_ic, fold_ics=fold_ics,
                                         n_test_weeks=n_test_weeks))
    return {"name": name, "params": params or {}, "mean_ic": mean_ic, "fold_ics": fold_ics,
            "n_test_weeks": n_test_weeks, "is_production": is_production, "eligible": eligible}


def test_select_best_skips_ineligible_top_mean_config():
    # cfg0 has the higher mean IC but a NaN fold (degenerate predictor); cfg1 is
    # lower-mean but clean across all folds — selection must prefer cfg1.
    results = pd.DataFrame([
        _row("cfg0", 0.05, [0.1, float("nan"), 0.02], is_production=True),
        _row("cfg1", 0.03, [0.03, 0.03, 0.03]),
    ])
    best = select_best(results)
    assert best["name"] == "cfg1"


def test_select_best_returns_none_when_no_config_eligible():
    results = pd.DataFrame([
        _row("cfg0", 0.05, [0.1, float("nan"), 0.02]),
        _row("cfg1", 0.04, [float("nan"), 0.02, 0.01]),
    ])
    assert select_best(results) is None


def test_tune_xgb_selects_runner_up_when_top_mean_config_ineligible(
        synthetic_store, tiny_cfg, tmp_path, monkeypatch):
    """End-to-end reproduction of the real-run bug: cfg0 (production reference)
    has the highest mean IC but a degenerate (NaN) fold from early stopping;
    the tuner must exclude it exactly as champion.py's tournament would, and
    select an eligible runner-up instead."""
    from stocks_ml.features.panel import build_panel
    import stocks_ml.models.tuning as tuning_mod

    build_panel(synthetic_store, tiny_cfg)

    def fake_evaluate(name, est, labeled, splits, fcols):
        if name == "cfg0":
            return CandidateResult(name=name, mean_ic=0.05,
                                   fold_ics=[0.1, float("nan"), 0.02], n_test_weeks=20)
        return CandidateResult(name=name, mean_ic=0.02, fold_ics=[0.02, 0.02, 0.02],
                               n_test_weeks=30)

    monkeypatch.setattr(tuning_mod, "evaluate_candidate", fake_evaluate)

    out = tmp_path / "models"
    ranked = tune_xgb(synthetic_store, tiny_cfg, n_samples=2, out_dir=out)

    assert (out / "xgb_tuned.json").exists()
    row0 = ranked[ranked["name"] == "cfg0"].iloc[0]
    assert not row0["eligible"]

    text = (out / "tuning.md").read_text()
    cfg0_line = next(l for l in text.splitlines() if l.startswith("| cfg0"))
    assert "(ineligible: degenerate fold)" in cfg0_line
    assert "selected" not in cfg0_line
    selected_lines = [l for l in text.splitlines() if "selected" in l]
    assert len(selected_lines) == 1 and not selected_lines[0].startswith("| cfg0")


def test_tune_xgb_writes_no_json_when_no_config_eligible(
        synthetic_store, tiny_cfg, tmp_path, monkeypatch):
    from stocks_ml.features.panel import build_panel
    import stocks_ml.models.tuning as tuning_mod

    build_panel(synthetic_store, tiny_cfg)

    def fake_evaluate(name, est, labeled, splits, fcols):
        return CandidateResult(name=name, mean_ic=0.01, fold_ics=[0.01, float("nan")],
                               n_test_weeks=5)

    monkeypatch.setattr(tuning_mod, "evaluate_candidate", fake_evaluate)

    out = tmp_path / "models"
    ranked = tune_xgb(synthetic_store, tiny_cfg, n_samples=2, out_dir=out)

    assert not (out / "xgb_tuned.json").exists()
    assert not ranked["eligible"].any()
    text = (out / "tuning.md").read_text()
    assert "no config was eligible" in text.lower()
    assert "xgb_tuned.json" in text


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
