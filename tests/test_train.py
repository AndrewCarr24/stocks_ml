"""stocks_ml.train: the walk's record, its guard, the holdout refusal."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stocks_ml.selection as sel
import stocks_ml.train as train


def test_record_names_what_made_the_walk():
    rec = train.record("data/w", "2006-01-01", "2015-12-31", "label_4w_sector", 8, 16, 1,
                       "last_print", "nominal")
    assert rec["code"] == "stocks_ml.train.walk"
    assert rec["recipe"] == {"label": "label_4w_sector", "train_years": 8}
    assert (rec["k"], rec["store"], rec["delist"], rec["price_basis"]) == (16, "data/w", "last_print", "nominal")
    assert rec["base_params"] == {p: str(v) for p, v in sel.MODEL_PARAMS.items()}
    assert rec["weeks"] == "every week of 2006-01-01 -> 2015-12-31" and rec["every"] == 1
    sample = train.record("data/w", "2006-01-01", "2015-12-31", "label_4w", 5, 4, 4, "drop", "closeadj")
    assert sample["weeks"] == "every 4th week of 2006-01-01 -> 2015-12-31 (a sample)"
    json.dumps(rec)                                                     # spec.json is plain JSON


def test_guard_writes_the_record_once_and_refuses_another(tmp_path):
    rec = train.record("data/w", "2006-01-01", "2015-12-31", "label_4w_sector", 8, 16, 1, "last_print", "nominal")
    out = tmp_path / "walk" / "select"
    train.guard(out, rec)
    assert json.loads((out / "spec.json").read_text()) == rec
    train.guard(out, dict(rec))                                         # the same record resumes
    other = train.record("data/w", "2006-01-01", "2015-12-31", "label_4w", 8, 16, 1, "last_print", "nominal")
    with pytest.raises(RuntimeError, match="was walked under"):
        train.guard(out, other)


def test_walk_refuses_the_holdout_and_unknown_labels(tmp_path):
    with pytest.raises(SystemExit, match="holdout"):
        train.walk("unused", "2006-01-01", sel.HOLDOUT_START, "label_4w_sector", 8, 16, tmp_path)
    with pytest.raises(SystemExit, match="label must be one of"):
        train.walk("unused", "2006-01-01", "2015-12-31", "label_4w_gauss", 8, 16, tmp_path)
    assert not (tmp_path / "spec.json").exists()


def test_walk_writes_week_ticker_copies_and_resumes(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=5, freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks, delist_labels="last_print", cfg=SimpleNamespace(price_basis="nominal"))
    monkeypatch.setattr(train, "context", lambda store: (sel, ctx, 64))
    fitted = []

    def fake_copy(sel_, ctx_, t, c, label, train_years, features=(), params=None, n_jobs=None, drop=(), train_top=None, block=None):
        fitted.append((t, c))
        return pd.Series({"A": 1.0 * c, "B": 2.0 * c}, name=t)
    monkeypatch.setattr(train, "copy_preds", fake_copy)
    out = tmp_path / "select"
    path = train.walk("data/w", weeks[0], weeks[2], "label_4w_sector", 8, 2, out, checkpoint=2, log=lambda m: None)
    df = pd.read_parquet(path)
    assert list(df.columns) == ["week", "ticker", "c1", "c2"]
    assert df.week.nunique() == 3 and len(df) == 6 and df.c2.iloc[0] == 2.0
    assert len(fitted) == 6
    # the record beside it, then a resume over a longer window fits only the new weeks
    rec = json.loads((out / "spec.json").read_text())
    assert rec["recipe"] == {"label": "label_4w_sector", "train_years": 8} and rec["k"] == 2
    with pytest.raises(RuntimeError, match="was walked under"):        # a different window: a different record
        train.walk("data/w", weeks[0], weeks[4], "label_4w_sector", 8, 2, out, log=lambda m: None)
    fitted.clear()
    train.walk("data/w", weeks[0], weeks[2], "label_4w_sector", 8, 2, out, log=lambda m: None)
    assert fitted == []                                                 # everything was done
    sample = train.walk("data/w", weeks[0], weeks[4], "label_4w_sector", 8, 2, tmp_path / "s", every=2,
                        log=lambda m: None)
    assert sorted(pd.read_parquet(sample).week.unique()) == [np.datetime64(w) for w in weeks[::2]]
    assert "(a sample)" in json.loads((tmp_path / "s" / "spec.json").read_text())["weeks"]


def test_check_reproduces_compares_the_saved_copies(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=3, freq="W-FRI"))
    out = tmp_path / "select"
    out.mkdir()
    (out / "spec.json").write_text(json.dumps({"recipe": {"label": "label_4w_sector", "train_years": 8}}))
    pd.DataFrame([{"week": t, "ticker": tk, "c1": float(i)} for t in weeks for i, tk in enumerate("AB")]
                 ).to_parquet(out / "preds.parquet", index=False)
    monkeypatch.setattr(train, "context", lambda store: (sel, None, 64))
    monkeypatch.setattr(train, "copy_preds", lambda *a, **k: pd.Series({"A": 0.0, "B": 1.0}))
    got = train.check_reproduces("w", out / "preds.parquet", weeks=2, log=lambda m: None)
    assert got == {"2006-01-06 c1": True, "2006-01-13 c1": True}
    monkeypatch.setattr(train, "copy_preds", lambda *a, **k: pd.Series({"A": 0.0, "B": 1.001}))
    assert train.check_reproduces("w", out / "preds.parquet", weeks=1, log=lambda m: None) == {"2006-01-06 c1": False}
    monkeypatch.setattr(train, "copy_preds", lambda *a, **k: pd.Series({"A": 0.0}))     # a missing name differs
    assert train.check_reproduces("w", out / "preds.parquet", weeks=1, log=lambda m: None) == {"2006-01-06 c1": False}


def test_recipe_carries_features_and_typed_params_only_when_given():
    assert train.recipe("label_4w_sector", 8) == {"label": "label_4w_sector", "train_years": 8}
    r = train.recipe("label_4w_sector_rank", 5, ["x_a", "x_b"], {"learning_rate": "0.01", "n_estimators": "300"})
    assert r["features"] == ["x_a", "x_b"]
    assert r["params"] == {"learning_rate": 0.01, "n_estimators": 300}     # typed like MODEL_PARAMS
    with pytest.raises(ValueError):
        train.recipe("label_4w_sector", 8, params={"depth": 3})           # not a MODEL_PARAMS key
    rec = train.record("data/w", "2006-01-01", "2015-12-31", "label_4w_sector", 8, 16, 1, "last_print",
                       "nominal", features=["x_a"], params={"max_depth": 4})
    assert rec["recipe"] == {"label": "label_4w_sector", "train_years": 8, "features": ["x_a"],
                             "params": {"max_depth": 4}}
    assert train.recipe("label_4w_sector", 8, drop=["f_z"]) == {"label": "label_4w_sector", "train_years": 8,
                                                                "drop": ["f_z"]}
    rec = train.record("data/w", "2006-01-01", "2015-12-31", "label_4w_sector", 8, 16, 1, "last_print",
                       "nominal", drop=["f_z"])
    assert rec["recipe"] == {"label": "label_4w_sector", "train_years": 8, "drop": ["f_z"]}


def test_walk_takes_an_explicit_week_list_described_as_a_sample(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=6, freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks, delist_labels="last_print", cfg=SimpleNamespace(price_basis="nominal"))
    monkeypatch.setattr(train, "context", lambda store: (sel, ctx, 64))
    monkeypatch.setattr(train, "copy_preds", lambda *a, **k: pd.Series({"A": 1.0, "B": 2.0}))
    with pytest.raises(ValueError):
        train.walk("data/w", weeks[0], weeks[-1], "label_4w_sector", 8, 1, tmp_path / "x", weeks=weeks[::3],
                   log=lambda m: None)
    path = train.walk("data/w", weeks[0], weeks[-1], "label_4w_sector", 8, 1, tmp_path / "s", weeks=weeks[::3],
                      sample="2 weeks, a stratified random sample (1 per year, seed 0)", log=lambda m: None)
    df = pd.read_parquet(path)
    assert sorted(df.week.unique()) == weeks[::3]
    rec = json.loads((tmp_path / "s" / "spec.json").read_text())
    assert rec["sample"].startswith("2 weeks") and "(a sample)" in rec["weeks"]


def test_walk_takes_an_explicit_copy_range_for_a_seed_twin(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=3, freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks, delist_labels="last_print", cfg=SimpleNamespace(price_basis="nominal"))
    monkeypatch.setattr(train, "context", lambda store: (sel, ctx, 64))
    seen = []
    monkeypatch.setattr(train, "copy_preds", lambda s_, c_, t, c, *a, **k: (seen.append(c), pd.Series({"A": float(c)}))[1])
    path = train.walk("data/w", weeks[0], weeks[-1], "label_4w_sector", 8, 2, tmp_path / "twin",
                      copies=range(17, 19), log=lambda m: None)
    df = pd.read_parquet(path)
    assert list(df.columns) == ["week", "ticker", "c17", "c18"] and sorted(set(seen)) == [17, 18]
    assert json.loads((tmp_path / "twin" / "spec.json").read_text())["copies"] == [17, 18]
    with pytest.raises(ValueError):
        train.walk("data/w", weeks[0], weeks[-1], "label_4w_sector", 8, 3, tmp_path / "x", copies=range(17, 19),
                   log=lambda m: None)


def test_thread_budget_shares_the_cores_or_honours_the_cap(monkeypatch):
    monkeypatch.setattr(train.os, "cpu_count", lambda: 14)
    monkeypatch.delenv(train.CORES_ENV, raising=False)
    assert train.thread_budget(7) == (7, 2)
    assert train.thread_budget(1) == (1, None)               # in-process: XGBoost's default, every core
    monkeypatch.setenv(train.CORES_ENV, "4")
    assert train.thread_budget(7) == (4, 1)                  # never more workers than the cap
    assert train.thread_budget(2) == (2, 2)
    assert train.thread_budget(1) == (1, 4)


def test_fit_context_keeps_only_what_a_fit_sees():
    import stocks_ml.selection as sel
    from stocks_ml.features.panel import feature_cols
    pan = pd.DataFrame({"date": pd.to_datetime(["2006-01-06"] * 2), "ticker": ["A", "B"],
                        "f_mom_4w": [0.1, 0.2], "f_sec_x": [1.0, 2.0], "x_cand": [3.0, 4.0],
                        "x_other": [5.0, 6.0], "g_old": [7.0, 8.0], "label_4w": [0.0, 1.0],
                        "fwd_ret_4w": [0.0, 1.0], "aux_sector": ["T", "T"]})
    fake = SimpleNamespace(pan=pan, extra=["x_zz"], cfg=SimpleNamespace(cv_train_years=1))
    fc = train.FitContext(fake, features=["x_cand"])
    assert list(fc.pan.columns) == ["date", "ticker", "f_mom_4w", "x_cand", "label_4w", "fwd_ret_4w", "aux_sector"]
    wide = SimpleNamespace(pan=pd.concat([pan.assign(date=pd.Timestamp(d)) for d in
                                          ("1996-01-05", "1997-06-06", "2006-01-06", "2015-12-31", "2016-01-08")]),
                           extra=[], cfg=fake.cfg)
    cut = train.FitContext(wide, lo="2006-01-06", hi="2015-12-31", train_years=8)
    assert sorted(cut.pan["date"].dt.year.unique()) == [1997, 2006, 2015]     # 8y + 1 before lo .. hi
    assert feature_cols(fc.pan) == feature_cols(pan) == ["f_mom_4w"]
    assert fc.extra == ["x_zz"] and fc.world_cfg(5).cv_train_years == 5
    assert train.fit_context(fake) is fake                    # a stand-in passes through
    real = sel.Ctx.__new__(sel.Ctx)
    real.pan, real.extra, real.cfg = pan, [], fake.cfg
    assert isinstance(train.fit_context(real), train.FitContext)


def test_recipe_and_cap_rank_for_a_training_universe(tmp_path):
    assert train.recipe("label_4w_sector", 8, train_top=500) == {"label": "label_4w_sector", "train_years": 8, "train_top": 500}
    assert "train_top" not in train.recipe("label_4w_sector", 8, train_top=None)
    snaps = pd.DataFrame([("2010-03-31", "A", 100), ("2010-03-31", "B", 50), ("2010-03-31", "C", 10),
                          ("2010-06-30", "A", 10), ("2010-06-30", "B", 50), ("2010-06-30", "C", 100)],
                         columns=["date", "ticker", "marketcap"])
    snaps.to_parquet(tmp_path / "daily_snapshots.parquet", index=False)
    pan = pd.DataFrame({"date": pd.to_datetime(["2010-03-31", "2010-05-07", "2010-05-07", "2010-07-02", "2010-07-02"]),
                        "ticker": ["A", "A", "C", "A", "D"]})
    r = train.cap_rank_asof(str(tmp_path), pan)
    assert r.isna().tolist() == [True, False, False, False, True]     # nothing strictly before 2010-03-31; D unknown
    assert r.dropna().tolist() == [1.0, 3.0, 3.0]                      # A 1st then 3rd; C 3rd at the first snapshot
    with pytest.raises(SystemExit, match="daily_snapshots"):
        train.cap_rank_asof(str(tmp_path / "nowhere"), pan)


def test_walk_refit_every_fits_once_per_block_and_scores_every_week(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=7, freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks, delist_labels="last_print", cfg=SimpleNamespace(price_basis="nominal"))
    monkeypatch.setattr(train, "context", lambda store: (sel, ctx, 64))
    fitted = []
    def fake_copy(sel_, ctx_, t, c, label, train_years, features=(), params=None, n_jobs=None, drop=(), train_top=None, block=None):
        fitted.append((t, c, block))
        return {w: pd.Series({"A": 1.0 * c, "B": 2.0 * c}, name=w) for w in block}
    monkeypatch.setattr(train, "copy_preds", fake_copy)
    out = tmp_path / "r3"
    path = train.walk("data/w", weeks[0], weeks[-1], "label_4w_sector", 8, 2, out, refit_every=3, log=lambda m: None)
    assert [f[0] for f in fitted if f[1] == 1] == [weeks[0], weeks[3], weeks[6]]      # one fit per block of 3
    assert fitted[0][2] == tuple(weeks[:3]) and fitted[-1][2] == (weeks[6],)
    df = pd.read_parquet(path)
    assert df.week.nunique() == 7 and set(df.columns) == {"week", "ticker", "c1", "c2"}   # every week scored
    rec = json.loads((out / "spec.json").read_text())
    assert rec["refit_every"] == 3 and "refit every 3 weeks" in rec["weeks"]
    with pytest.raises(ValueError, match="every week"):
        train.walk("data/w", weeks[0], weeks[-1], "label_4w_sector", 8, 2, tmp_path / "x", every=2, refit_every=3, log=lambda m: None)


def test_walk_takes_a_block_sample_at_the_cadence(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=12, freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks, delist_labels="last_print", cfg=SimpleNamespace(price_basis="nominal"))
    monkeypatch.setattr(train, "context", lambda store: (sel, ctx, 64))
    fitted = []
    def fake_copy(sel_, ctx_, t, c, label, train_years, features=(), params=None, n_jobs=None, drop=(), train_top=None, block=None):
        fitted.append(block)
        return {w: pd.Series({"A": 1.0}, name=w) for w in block}
    monkeypatch.setattr(train, "copy_preds", fake_copy)
    sample = weeks[0:4] + weeks[8:12]                                # two runs of four
    path = train.walk("data/w", weeks[0], weeks[-1], "label_4w_sector", 8, 1, tmp_path / "b", weeks=sample,
                      sample="two runs", refit_every=4, log=lambda m: None)
    assert fitted == [tuple(weeks[0:4]), tuple(weeks[8:12])]
    assert pd.read_parquet(path).week.nunique() == 8
    with pytest.raises(ValueError, match="consecutive"):
        train.walk("data/w", weeks[0], weeks[-1], "label_4w_sector", 8, 1, tmp_path / "c", weeks=weeks[0:3] + weeks[5:6],
                   sample="broken", refit_every=4, log=lambda m: None)
