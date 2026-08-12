import numpy as np
import pandas as pd
import pytest

from stocks_ml.backtest.metrics import deflated_sharpe
from stocks_ml.models.trials import ledger_stats, load_ledger, record_trials


def test_record_trials_appends_and_upserts(tmp_path):
    p = tmp_path / "ledger.json"
    n = record_trials([{"kind": "tune_trial", "name": "a", "cv_metric": 0.01},
                       {"kind": "tune_trial", "name": "b", "cv_metric": 0.02}], p)
    assert n == 2
    # rerunning the same config updates in place — N must not inflate
    n = record_trials([{"kind": "tune_trial", "name": "b", "cv_metric": 0.03}], p)
    assert n == 2
    rows = load_ledger(p)
    assert {r["name"]: r["cv_metric"] for r in rows} == {"a": 0.01, "b": 0.03}


def test_record_trials_sanitizes_nonfinite(tmp_path):
    p = tmp_path / "ledger.json"
    record_trials([{"kind": "x", "name": "n", "cv_metric": float("nan")}], p)
    assert load_ledger(p)[0]["cv_metric"] is None


def test_ledger_stats_variance_needs_sharpes(tmp_path):
    p = tmp_path / "ledger.json"
    record_trials([{"kind": "t", "name": f"c{i}", "cv_metric": 0.01} for i in range(5)], p)
    n, var = ledger_stats(p)
    assert n == 5 and var is None            # no Sharpes recorded -> no variance
    record_trials([{"kind": "s", "name": f"s{i}", "pre_holdout_sharpe": sr}
                   for i, sr in enumerate([0.4, 0.6, 0.9])], p)
    n, var = ledger_stats(p)
    assert n == 8
    assert var == pytest.approx(np.var([0.4, 0.6, 0.9], ddof=1))


def test_deflated_sharpe_uses_cross_trial_variance():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 500))
    base = deflated_sharpe(rets, n_trials=50)
    # larger cross-trial dispersion -> higher expected-max benchmark -> lower DSR
    tight = deflated_sharpe(rets, n_trials=50, cross_trial_var=0.0001)
    wide = deflated_sharpe(rets, n_trials=50, cross_trial_var=1.0)
    assert wide < base < 1.0
    assert wide < tight
