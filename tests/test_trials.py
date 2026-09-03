from stocks_ml.models.trials import load_ledger, record_trials


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

