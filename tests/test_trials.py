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



def test_record_trials_sanitizes_nested_values(tmp_path):
    """np scalars and NaN inside nested dicts/lists crashed json.dumps or
    emitted bare NaN the next load could not parse."""
    import numpy as np
    p = tmp_path / "ledger.json"
    record_trials([{"kind": "x", "name": "n",
                    "windows": {"2016": {"sharpe": np.float64("nan"), "n": np.int64(4)},
                                "flags": [np.bool_(True), float("inf")]}}], p)
    row = load_ledger(p)[0]
    assert row["windows"]["2016"] == {"sharpe": None, "n": 4}
    assert row["windows"]["flags"] == [True, None]


def test_record_trials_survives_concurrent_writers(tmp_path):
    """Two processes upserting different rows at once must both land (the
    unlocked read-modify-write dropped one)."""
    import multiprocessing as mp
    p = tmp_path / "ledger.json"
    record_trials([{"kind": "x", "name": "seed"}], p)
    names = [f"w{i}" for i in range(8)]
    with mp.Pool(4) as pool:
        pool.starmap(record_trials, [([{"kind": "x", "name": n}], p) for n in names])
    got = {r["name"] for r in load_ledger(p)}
    assert got == {"seed", *names}
