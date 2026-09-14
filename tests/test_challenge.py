"""The challenger protocol: recipes parsed over the incumbent's, the sample
ranks and advances, the every-week comparison is same-basis and the argmax
decides, the leak audit gates, and the record is written."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stocks_ml.challenge as ch

BASE = {"label": "label_4w_sector", "train_years": 8}


def test_parse_candidate_inherits_the_incumbent_and_types_params():
    assert ch.parse_candidate("label=label_4w_sector_rank", BASE) == \
        {"label": "label_4w_sector_rank", "train_years": 8}
    r = ch.parse_candidate("train_years=5, features=x_a+x_b, params=learning_rate:0.01+n_estimators:300", BASE)
    assert r == {"label": "label_4w_sector", "train_years": 5, "features": ["x_a", "x_b"],
                 "params": {"learning_rate": 0.01, "n_estimators": 300}}
    with pytest.raises(ValueError):
        ch.parse_candidate("window=5", BASE)
    with pytest.raises(ValueError):
        ch.parse_candidate("params=depth:4", BASE)                       # not a MODEL_PARAMS key
    with pytest.raises(ValueError):
        ch.parse_candidate("label", BASE)


def test_candidate_names_are_readable_and_distinct():
    a = ch.candidate_name({"label": "label_4w_sector_rank", "train_years": 8})
    b = ch.candidate_name({"label": "label_4w_sector_rank", "train_years": 8, "features": ["x_a"]})
    c = ch.candidate_name({"label": "label_4w_sector_rank", "train_years": 8, "params": {"max_depth": 4}})
    assert a == "label_4w_sector_rank_8y" and b.startswith(a + "_") and c.startswith(a + "_") and b != c


def _walk(path: Path, weeks, k, scale=1.0):
    rows = [{"week": w, "ticker": t, **{f"c{c}": scale * c * (i + 1) for c in range(1, k + 1)}}
            for w in weeks for i, t in enumerate("ABC")]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    (path.parent / "spec.json").write_text(json.dumps({"recipe": BASE, "k": k}))
    return path


def test_cut_walk_keeps_the_sample_weeks_and_copies(tmp_path):
    weeks = list(pd.date_range("2006-01-06", periods=8, freq="W-FRI"))
    p = _walk(tmp_path / "inc" / "preds.parquet", weeks, 16)
    cut = ch.cut_walk(p, weeks[::4], 4)
    assert list(cut.columns) == ["week", "ticker", "c1", "c2", "c3", "c4"]
    assert sorted(cut.week.unique()) == weeks[::4]
    with pytest.raises(SystemExit):
        ch.cut_walk(p, weeks, 17)


def test_rank_by_metric_is_highest_first():
    M = {"a": {3: 0.0, 6: 3.0, 10: 0.0}, "b": {3: 2.0, 6: 2.0, 10: 2.0}, "c": {3: 1.0, 6: 1.0, 10: 4.0}, "d": {6: 9.0}}
    assert ch.rank_by_metric(M, ["a", "b", "c", "d"]) == ["b", "c", "a", "d"]   # by the mean; d lacks books
    assert ch.model_score(M["b"]) == 2.0 and ch.model_score(M["d"]) == float("-inf")


def test_run_samples_advances_compares_audits_and_records(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=12, freq="W-FRI"))
    inc = _walk(tmp_path / "inc" / "select" / "preds.parquet", weeks, 16)
    (tmp_path / "inc" / "extend").mkdir()
    ctx = SimpleNamespace(weeks=weeks)
    monkeypatch.setattr(ch, "context", lambda store: (None, ctx, 64))
    walked = []

    def fake_walk(store, lo, hi, label, train_years, k, out, every=1, log=None, features=(), params=None):
        walked.append((label, train_years, k, every, Path(out).name, tuple(features), params))
        return _walk(Path(out) / "preds.parquet", weeks[::every], k)
    monkeypatch.setattr(ch, "walk", fake_walk)
    # the metric is canned by candidate name: rank best, clip worst, incumbent in between at stage 2
    score = {"label_4w_sector_rank_8y": 11.0, "label_4w_sector_log_8y": 6.1,
             "label_4w_sector_clip_8y": 4.0, "incumbent": 6.75}

    def fake_metrics(sel, ctx_, walks, k, lo=None, hi=None):
        H = {n: pd.DataFrame({"week": weeks, "top3": 0.01, "top6": 0.01, "top10": 0.01}) for n in walks}
        return {n: {3: score[n], 6: score[n], 10: score[n]} for n in walks}, H, len(weeks)
    monkeypatch.setattr(ch, "metrics_on_common", fake_metrics)
    monkeypatch.setattr(ch, "copy_metrics", lambda *a, **k: [1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(ch, "paired_t_books", lambda a, b: 0.24)
    monkeypatch.setattr(ch, "audit_segments", lambda store, paths, ctx_: {"VERDICT": "PASS", "worst_retention": 0.9})
    monkeypatch.setattr(ch, "leak_line", lambda la: "PASS")
    led = []
    monkeypatch.setattr(ch, "record_trials", lambda rows: led.extend(rows))
    cands = [ch.parse_candidate(f"label=label_4w_sector_{s}", BASE) for s in ("log", "clip", "rank")]
    res = ch.run(cands, inc, tmp_path / "ch", store="s", log=lambda m: None)
    # stage 1: three sample walks (every 4th week, K=4), the top two advance
    assert [w for w in walked if w[3] == ch.SAMPLE_EVERY] == \
        [(f"label_4w_sector_{s}", 8, ch.SAMPLE_K, ch.SAMPLE_EVERY, "sample", (), None) for s in ("log", "clip", "rank")]
    assert res["stage1"]["advance"] == ["label_4w_sector_rank_8y", "label_4w_sector_log_8y"]
    # stage 2: only the advanced two walk every week; the argmax is rank; the incumbent is in the table
    assert [w[0] for w in walked if w[3] == 1] == ["label_4w_sector_rank", "label_4w_sector_log"]
    assert res["stage2"]["winner"] == "label_4w_sector_rank_8y"
    assert res["stage2"]["order"] == ["label_4w_sector_rank_8y", "incumbent", "label_4w_sector_log_8y"]
    assert res["stage2"]["detail"]["label_4w_sector_rank_8y"]["leak_audit"]["verdict"] == "PASS"
    assert res["stage2"]["detail"]["incumbent"]["paired_t_vs_incumbent"] is None
    assert "stage3" not in res                                            # --k16 not asked
    assert (tmp_path / "ch" / "challenge.json").exists()
    kinds = [(r["name"].split("_")[2], r["config"]["label"]) for r in led]
    assert ("stage1", "label_4w_sector_clip") in kinds and ("stage2", "label_4w_sector_rank") in kinds
    assert ("stage2", "label_4w_sector_clip") not in kinds                # never walked every week


def test_run_refuses_the_incumbents_own_recipe(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=4, freq="W-FRI"))
    inc = _walk(tmp_path / "inc" / "select" / "preds.parquet", weeks, 4)
    with pytest.raises(SystemExit):
        ch.run([dict(BASE)], inc, tmp_path / "ch", store="s", log=lambda m: None)


def test_leak_audit_failure_removes_a_candidate_from_the_argmax(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=8, freq="W-FRI"))
    inc = _walk(tmp_path / "inc" / "select" / "preds.parquet", weeks, 4)
    monkeypatch.setattr(ch, "context", lambda store: (None, SimpleNamespace(weeks=weeks), 64))
    monkeypatch.setattr(ch, "walk", lambda store, lo, hi, label, ty, k, out, every=1, log=None,
                        features=(), params=None: _walk(Path(out) / "preds.parquet", weeks[::every], k))
    monkeypatch.setattr(ch, "metrics_on_common", lambda sel, c, walks, k, lo=None, hi=None:
                        ({n: {b: (9.0 if n != "incumbent" else 5.0) for b in (3, 6, 10)} for n in walks},
                         {n: pd.DataFrame({"week": weeks, "top3": 0.0, "top6": 0.0, "top10": 0.0}) for n in walks},
                         len(weeks)))
    monkeypatch.setattr(ch, "copy_metrics", lambda *a, **k: [9.0])
    monkeypatch.setattr(ch, "paired_t_books", lambda a, b: 1.0)
    monkeypatch.setattr(ch, "audit_segments", lambda store, paths, ctx: {"VERDICT": "FAIL", "worst_retention": 0.4})
    monkeypatch.setattr(ch, "leak_line", lambda la: "FAIL")
    monkeypatch.setattr(ch, "record_trials", lambda rows: None)
    res = ch.run([ch.parse_candidate("label=label_4w_sector_rank", BASE)], inc, tmp_path / "ch",
                 store="s", log=lambda m: None)
    assert res["stage2"]["winner"] == "incumbent"                         # a leaky winner cannot win


def test_stratified_weeks_draw_the_same_seeded_weeks_per_year():
    weeks = list(pd.date_range("2006-01-06", "2008-12-26", freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks)
    a = ch.stratified_weeks(ctx, "2006-01-01", "2008-12-31", per_year=5, seed=1)
    b = ch.stratified_weeks(ctx, "2006-01-01", "2008-12-31", per_year=5, seed=1)
    c = ch.stratified_weeks(ctx, "2006-01-01", "2008-12-31", per_year=5, seed=2)
    assert a == b and a != c and a == sorted(a) and len(set(a)) == 15
    assert pd.Series([w.year for w in a]).value_counts().to_dict() == {2006: 5, 2007: 5, 2008: 5}
    assert all(w in weeks for w in a)
    assert len(ch.stratified_weeks(ctx, "2006-01-01", "2006-02-10", per_year=5)) == 5   # a short year: all of it


def test_candidate_text_names_only_what_differs():
    base = {"label": "label_4w_sector", "train_years": 8}
    assert ch.candidate_text({"label": "label_4w_sector_rank", "train_years": 8}, base) == "label=label_4w_sector_rank"
    assert ch.candidate_text({"label": "label_4w_sector", "train_years": 5, "features": ["x_a"],
                              "params": {"max_depth": 4}}, base) == "train_years=5,features=x_a,params=max_depth:4"


def test_run_fast_walks_the_sample_ranks_and_prints_the_promotion(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=20, freq="W-FRI"))
    inc = _walk(tmp_path / "inc" / "select" / "preds.parquet", weeks, 16)
    monkeypatch.setattr(ch, "context", lambda store: (None, SimpleNamespace(weeks=weeks), 64))
    walked = []

    def fake_walk(store, lo, hi, label, train_years, k, out, every=1, log=None, features=(), params=None,
                  weeks=None, sample=None):
        walked.append((label, k, Path(out).name, len(weeks), sample))
        return _walk(Path(out) / "preds.parquet", weeks, k)
    monkeypatch.setattr(ch, "walk", fake_walk)
    score = {"label_4w_sector_rank_8y": 4.9, "label_4w_sector_log_8y": -2.7, "incumbent": -5.2}
    monkeypatch.setattr(ch, "metrics_on_common", lambda sel, c, walks, k, lo=None, hi=None:
                        ({n: {3: score[n], 6: score[n], 10: score[n]} for n in walks},
                         {n: pd.DataFrame({"week": weeks[:5], "top3": 0.02, "top6": 0.02, "top10": 0.02,
                                           "rand_mean": 0.01}) for n in walks}, 5))
    monkeypatch.setattr(ch, "copy_metrics", lambda *a, **k: [1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(ch, "paired_t_books", lambda a, b: 0.5)
    led, lines = [], []
    monkeypatch.setattr(ch, "record_trials", lambda rows: led.extend(rows))
    cands = [ch.parse_candidate(f"label=label_4w_sector_{s}", BASE) for s in ("log", "rank")]
    res = ch.run_fast(cands, inc, tmp_path / "fast", store="s", per_year=2, seed=3, log=lines.append)
    assert [w[:3] for w in walked] == [("label_4w_sector_log", ch.SAMPLE_K, "fast_s3"),
                                       ("label_4w_sector_rank", ch.SAMPLE_K, "fast_s3")]
    assert walked[0][3] == 2 and "stratified random sample (2 per year, seed 3)" in walked[0][4]
    assert res["order"] == ["label_4w_sector_rank_8y", "label_4w_sector_log_8y"]
    assert res["better_than_incumbent"] == ["label_4w_sector_rank_8y", "label_4w_sector_log_8y"]
    assert res["rows"]["incumbent"]["mean_excess_pp"] == pytest.approx(1.0)
    assert any("stocks-ml challenge --out <dir> --candidate 'label=label_4w_sector_rank'" in l for l in lines)
    assert (tmp_path / "fast" / "fast_s3.json").exists()
    assert [r["kind"] for r in led] == ["challenge_fast", "challenge_fast"]
