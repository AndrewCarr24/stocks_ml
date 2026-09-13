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

    def fake_copy(sel_, ctx_, t, c, label, train_years):
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
    monkeypatch.setattr(train, "copy_preds", lambda *a: pd.Series({"A": 0.0, "B": 1.0}))
    got = train.check_reproduces("w", out / "preds.parquet", weeks=2, log=lambda m: None)
    assert got == {"2006-01-06 c1": True, "2006-01-13 c1": True}
    monkeypatch.setattr(train, "copy_preds", lambda *a: pd.Series({"A": 0.0, "B": 1.001}))
    assert train.check_reproduces("w", out / "preds.parquet", weeks=1, log=lambda m: None) == {"2006-01-06 c1": False}
    monkeypatch.setattr(train, "copy_preds", lambda *a: pd.Series({"A": 0.0}))     # a missing name differs
    assert train.check_reproduces("w", out / "preds.parquet", weeks=1, log=lambda m: None) == {"2006-01-06 c1": False}
