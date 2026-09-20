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
    assert ch.parse_candidate("features=x_a, drop=f_z+f_y", BASE) == \
        {"label": "label_4w_sector", "train_years": 8, "features": ["x_a"], "drop": ["f_z", "f_y"]}
    assert ch.parse_candidate("store=data/other_world", BASE) == \
        {"label": "label_4w_sector", "train_years": 8, "store": "data/other_world"}
    assert ch.parse_candidate("store=data/w2, train_top=500", BASE) == \
        {"label": "label_4w_sector", "train_years": 8, "train_top": 500, "store": "data/w2"}
    assert ch.candidate_text({"label": "label_4w_sector", "train_years": 8, "train_top": 500}, BASE) == "train_top=500"
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
    d = ch.candidate_name({"label": "label_4w_sector_rank", "train_years": 8, "drop": ["f_z"]})
    e = ch.candidate_name({"label": "label_4w_sector_rank", "train_years": 8, "store": "data/w2"})
    assert a == "label_4w_sector_rank_8y" and b.startswith(a + "_") and c.startswith(a + "_") and b != c
    assert d.startswith(a + "_") and d not in (b, c) and e.startswith(a + "_") and e not in (b, c, d)


def _walk(path: Path, weeks, k, scale=1.0, first=1):
    rows = [{"week": w, "ticker": t, **{f"c{c}": scale * c * (i + 1) for c in range(first, first + k)}}
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

    def fake_walk(store, lo, hi, label, train_years, k, out, every=1, log=None, features=(), params=None,
                  workers=1, drop=(), train_top=None, refit_every=1):
        walked.append((label, train_years, k, every, Path(out).name, tuple(features), params))
        return _walk(Path(out) / "preds.parquet", weeks[::every], k)
    monkeypatch.setattr(ch, "walk", fake_walk)
    # the metric is canned by candidate name: rank best, clip worst, incumbent in between at stage 2
    score = {"label_4w_sector_rank_8y": 11.0, "label_4w_sector_log_8y": 6.1,
             "label_4w_sector_clip_8y": 4.0, "incumbent": 6.75}

    def fake_metrics(sel, ctx_, walks, k, lo=None, hi=None, ctxs=None):
        H = {n: pd.DataFrame({"week": weeks, "top3": 0.01, "top6": 0.01, "top10": 0.01}) for n in walks}
        return {n: {3: score[n], 6: score[n], 10: score[n]} for n in walks}, H, len(weeks)
    monkeypatch.setattr(ch, "metrics_on_common", fake_metrics)
    monkeypatch.setattr(ch, "copy_metrics", lambda *a, **k: [1.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(ch, "seed_band", lambda *a, **k: {"half_mean": 0.0, "half_sd": 1.0, "sd16": 0.7, "draws": 40, "copies": 16})
    monkeypatch.setattr(ch, "paired_t_books", lambda a, b: 2.4)
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
    inc = _walk(tmp_path / "inc" / "select" / "preds.parquet", weeks, ch.FULL_K)
    monkeypatch.setattr(ch, "context", lambda store: (None, SimpleNamespace(weeks=weeks), 64))
    monkeypatch.setattr(ch, "walk", lambda store, lo, hi, label, ty, k, out, every=1, log=None,
                        features=(), params=None, workers=1, drop=(), train_top=None, refit_every=1: _walk(Path(out) / "preds.parquet", weeks[::every], k))
    monkeypatch.setattr(ch, "metrics_on_common", lambda sel, c, walks, k, lo=None, hi=None, ctxs=None:
                        ({n: {b: (9.0 if n != "incumbent" else 5.0) for b in (3, 6, 10)} for n in walks},
                         {n: pd.DataFrame({"week": weeks, "top3": 0.0, "top6": 0.0, "top10": 0.0}) for n in walks},
                         len(weeks)))
    monkeypatch.setattr(ch, "copy_metrics", lambda *a, **k: [9.0])
    monkeypatch.setattr(ch, "seed_band", lambda *a, **k: {"half_mean": 0.0, "half_sd": 1.0, "sd16": 0.7, "draws": 40, "copies": 16})
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
    assert ch.candidate_text({"label": "label_4w_sector", "train_years": 8, "drop": ["f_z", "f_y"]}, base) == "drop=f_z+f_y"
    assert ch.candidate_text({"label": "label_4w_sector", "train_years": 8, "store": "data/w2"}, base) == "store=data/w2"


def test_run_fast_walks_the_sample_flags_by_the_null_and_prints_the_promotion(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=20, freq="W-FRI"))
    inc = _walk(tmp_path / "inc" / "select" / "preds.parquet", weeks, 16)
    monkeypatch.setattr(ch, "context", lambda store: (None, SimpleNamespace(weeks=weeks), 64))
    walked = []

    def fake_walk(store, lo, hi, label, train_years, k, out, every=1, log=None, features=(), params=None,
                  weeks=None, sample=None, workers=1, copies=None, drop=(), train_top=None, refit_every=1):
        walked.append((label, k, Path(out).name, len(weeks), sample))
        return _walk(Path(out) / "preds.parquet", weeks, k)
    monkeypatch.setattr(ch, "walk", fake_walk)
    twin = _walk(tmp_path / "inc" / "twin" / "preds.parquet", weeks, 16, first=17)     # seeds 17..32
    monkeypatch.setattr(ch, "ensure_twin", lambda *a, **k: twin)
    null = np.concatenate([np.linspace(-6, 6, 99), [100.0]])            # centred: 90th percentile ~3.9
    monkeypatch.setattr(ch, "null_gaps", lambda *a, **k: null)
    score = {"label_4w_sector_rank_8y": 8.0, "label_4w_sector_log_8y": 3.0, "label_4w_sector_clip_8y": -2.0,
             "incumbent": 1.0, "incumbent (twin seeds)": 2.0}            # the luckier seed set is the bar
    monkeypatch.setattr(ch, "metrics_on_common", lambda sel, c, walks, k, lo=None, hi=None, ctxs=None:
                        ({n: {3: score[n], 6: score[n], 10: score[n]} for n in walks},
                         {n: pd.DataFrame({"week": weeks[:5], "top3": 0.02, "top6": 0.02, "top10": 0.02,
                                           "rand_mean": 0.01}) for n in walks}, 5))
    monkeypatch.setattr(ch, "copy_metrics", lambda *a, **k: [1.0] * ch.FAST_K)
    monkeypatch.setattr(ch, "paired_t_books", lambda a, b: 2.5)
    led, lines = [], []
    monkeypatch.setattr(ch, "record_trials", lambda rows: led.extend(rows))
    cands = [ch.parse_candidate(f"label=label_4w_sector_{s}", BASE) for s in ("log", "clip", "rank")]
    res = ch.run_fast(cands, inc, tmp_path / "fast", store="s", per_year=2, seed=3, log=lines.append)
    assert [w[:3] for w in walked] == [("label_4w_sector_log", ch.FAST_K, "fast_2x_s3"),
                                       ("label_4w_sector_clip", ch.FAST_K, "fast_2x_s3"),
                                       ("label_4w_sector_rank", ch.FAST_K, "fast_2x_s3")]
    assert ch.FAST_K == 16 and walked[0][3] == 2 and "stratified random sample (2 per year, seed 3)" in walked[0][4]
    assert res["order"] == ["label_4w_sector_rank_8y", "label_4w_sector_log_8y", "label_4w_sector_clip_8y"]
    r = res["rows"]
    assert r["label_4w_sector_rank_8y"]["gap"] == 6.0 and r["label_4w_sector_rank_8y"]["flagged"]      # 8 - 2 > ~3.9
    assert r["label_4w_sector_rank_8y"]["gap_vs_incumbent_seeds_1_16"] == 7.0
    assert r["label_4w_sector_log_8y"]["gap"] == 1.0 and not r["label_4w_sector_log_8y"]["flagged"]    # inside the null
    assert r["incumbent (twin seeds)"]["gap"] is None and res["null"]["centred"]
    assert r["label_4w_sector_clip_8y"]["null_percentile"] < 0.5 < r["label_4w_sector_rank_8y"]["null_percentile"]
    assert res["flagged"] == ["label_4w_sector_rank_8y"] and res["null"]["flag_percentile"] == ch.FLAG_PERCENTILE
    assert r["incumbent"]["gap"] is None and r["incumbent"]["mean_excess_pp"] == pytest.approx(1.0)
    cut = ch.cut_walk(twin, weeks[:2], 16, copies=ch.twin_copies())
    assert list(cut.columns) == ["week", "ticker", *[f"c{i}" for i in range(1, 17)]]                 # renamed
    assert any("stocks-ml challenge --out <dir> --candidate 'label=label_4w_sector_rank' --k16" in l for l in lines)
    assert (tmp_path / "fast" / "fast_2x_s3.json").exists()
    assert [x["kind"] for x in led] == ["challenge_fast"] * 3


def test_null_percentile_twin_dir_and_copies():
    null = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert ch.null_percentile(2.0, null) == 0.8 and ch.null_percentile(-5.0, null) == 0.0
    assert ch.twin_dir(Path("data/experiments/labels/rank_k16/select/preds.parquet")) == \
        Path("data/experiments/labels/rank_k16/twin")
    assert list(ch.twin_copies()) == list(range(17, 33))


def test_run_walks_and_scores_a_store_candidate_on_its_own_world(tmp_path, monkeypatch):
    """A candidate that names another store (a wider universe) walks there and
    is scored with that world's context; the incumbent keeps its own."""
    weeks = list(pd.date_range("2006-01-06", periods=12, freq="W-FRI"))
    inc = _walk(tmp_path / "inc" / "select" / "preds.parquet", weeks, 16)
    ctx_a, ctx_b = SimpleNamespace(weeks=weeks, name="a"), SimpleNamespace(weeks=weeks, name="b")
    loaded = []
    def fake_context(store):
        loaded.append(store)
        return None, {"s": ctx_a, "data/w2": ctx_b}[store], 64
    monkeypatch.setattr(ch, "context", fake_context)
    walked = []
    def fake_walk(store, lo, hi, label, train_years, k, out, every=1, log=None, features=(), params=None,
                  workers=1, drop=(), train_top=None, refit_every=1):
        walked.append((store, k, every, Path(out).name))
        return _walk(Path(out) / "preds.parquet", weeks[::every], k)
    monkeypatch.setattr(ch, "walk", fake_walk)
    seen_ctx = {}
    def fake_metrics(sel, ctx_, walks, k, lo=None, hi=None, ctxs=None):
        key = k if isinstance(k, int) else max(k.values())            # stage 2 passes {name: copies}
        seen_ctx[key] = {n: (ctxs or {}).get(n, ctx_).name for n in walks}
        H = {n: pd.DataFrame({"week": weeks, "top3": 0.01, "top6": 0.01, "top10": 0.01}) for n in walks}
        sc = {"incumbent": 6.0}
        return {n: {3: sc.get(n, 9.0), 6: sc.get(n, 9.0), 10: sc.get(n, 9.0)} for n in walks}, H, len(weeks)
    monkeypatch.setattr(ch, "metrics_on_common", fake_metrics)
    copied = []
    monkeypatch.setattr(ch, "copy_metrics", lambda sel, ctx_, *a, **k: (copied.append(ctx_.name), [1.0])[1])
    monkeypatch.setattr(ch, "seed_band", lambda *a, **k: {"half_mean": 0.0, "half_sd": 1.0, "sd16": 0.7, "draws": 40, "copies": 16})
    monkeypatch.setattr(ch, "paired_t_books", lambda a, b: 2.5)
    audited = []
    monkeypatch.setattr(ch, "audit_segments", lambda store, paths, ctx_: (audited.append((store, ctx_.name)),
                                                                          {"VERDICT": "PASS"})[1])
    monkeypatch.setattr(ch, "leak_line", lambda la: "PASS")
    monkeypatch.setattr(ch, "record_trials", lambda rows: None)
    cand = ch.parse_candidate("store=data/w2", BASE)
    name = ch.candidate_name(cand)
    res = ch.run([cand], inc, tmp_path / "ch", store="s", log=lambda m: None)
    assert walked == [("data/w2", ch.SAMPLE_K, ch.SAMPLE_EVERY, "sample"), ("data/w2", ch.FULL_K, 1, "select")]
    assert loaded == ["s", "data/w2"]                                  # the second world loaded once, lazily
    assert seen_ctx[ch.FULL_K] == {"incumbent": "a", name: "b"}         # each walk scored on its own world
    assert copied == ["a", "b"] and audited == [("data/w2", "b")]  # copies scored per world; audit on its own
    assert res["stage2"]["winner"] == name


def test_run_cuts_stage_1_from_a_complete_select_walk(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", periods=12, freq="W-FRI"))
    inc = _walk(tmp_path / "inc" / "select" / "preds.parquet", weeks, 16)
    monkeypatch.setattr(ch, "context", lambda store: (None, SimpleNamespace(weeks=weeks), 64))
    cand = ch.parse_candidate("label=label_4w_sector_rank", BASE)
    name = ch.candidate_name(cand)
    _walk(tmp_path / "ch" / name / "select" / "preds.parquet", weeks, 16)     # already walked every week
    walked = []
    monkeypatch.setattr(ch, "walk", lambda store, lo, hi, label, ty, k, out, **kw: (walked.append(Path(out).name),
                                                                                    Path(out) / "preds.parquet")[1])
    monkeypatch.setattr(ch, "metrics_on_common", lambda sel, c, walks, k, lo=None, hi=None, ctxs=None: (
        {n: {3: 1.0, 6: 1.0, 10: 1.0} for n in walks},
        {n: pd.DataFrame({"week": weeks, "top3": 0.0, "top6": 0.0, "top10": 0.0}) for n in walks}, len(weeks)))
    monkeypatch.setattr(ch, "copy_metrics", lambda *a, **k: [1.0])
    monkeypatch.setattr(ch, "seed_band", lambda *a, **k: {"half_mean": 0.0, "half_sd": 1.0, "sd16": 0.7, "draws": 40, "copies": 16})
    monkeypatch.setattr(ch, "paired_t_books", lambda a, b: 0.0)
    monkeypatch.setattr(ch, "audit_segments", lambda store, paths, ctx_: {"VERDICT": "PASS"})
    monkeypatch.setattr(ch, "leak_line", lambda la: "PASS")
    monkeypatch.setattr(ch, "record_trials", lambda rows: None)
    ch.run([cand], inc, tmp_path / "ch", store="s", log=lambda m: None)
    assert walked == ["select"]                     # no sample walk; stage 2 resumes the complete one


def test_adjudicate_walks_the_twin_and_the_candidate_on_the_window_and_decides(tmp_path, monkeypatch):
    sel_weeks = list(pd.date_range("2006-01-06", periods=12, freq="W-FRI"))
    adj_weeks = list(pd.date_range("2016-01-08", periods=10, freq="W-FRI"))
    inc = _walk(tmp_path / "inc" / "select" / "preds.parquet", sel_weeks, 16)
    _walk(tmp_path / "inc" / "extend" / "preds.parquet", adj_weeks, 16)
    ctx_a, ctx_b = SimpleNamespace(weeks=sel_weeks + adj_weeks, name="a"), SimpleNamespace(weeks=sel_weeks + adj_weeks, name="b")
    monkeypatch.setattr(ch, "context", lambda store: (None, {"s": ctx_a, "data/w2": ctx_b}[store], 64))
    walked = []
    def fake_walk(store, lo, hi, label, ty, k, out, log=None, features=(), params=None, workers=1,
                  copies=None, drop=(), **kw):
        walked.append((store, str(lo.date()), str(hi.date()), k, Path(out).name, None if copies is None else list(copies)))
        return _walk(Path(out) / "preds.parquet", adj_weeks, k, first=1 if copies is None else copies.start)
    monkeypatch.setattr(ch, "walk", fake_walk)
    def fake_metrics(sel, ctx_, walks, k, lo=None, hi=None, ctxs=None):
        sc = {"incumbent": 8.0, "incumbent (twin seeds)": 9.0}
        H = {n: pd.DataFrame({"week": adj_weeks, "top3": 0.01, "top6": 0.01, "top10": 0.01}) for n in walks}
        return {n: {3: sc.get(n, 12.0), 6: sc.get(n, 12.0), 10: sc.get(n, 12.0)} for n in walks}, H, len(adj_weeks)
    monkeypatch.setattr(ch, "metrics_on_common", fake_metrics)
    monkeypatch.setattr(ch, "seed_band", lambda *a, **k: {"half_mean": 0.0, "half_sd": 1.0, "sd16": 0.7, "draws": 40, "copies": 16})
    monkeypatch.setattr(ch, "paired_t_books", lambda a, b: 2.5)
    monkeypatch.setattr(ch, "window_table", lambda *a, **k: {"rows": {}, "md": ["| table |"], "settings": {}})
    led = []
    monkeypatch.setattr(ch, "record_trials", lambda rows: led.extend(rows))
    cand = ch.parse_candidate("store=data/w2", BASE)
    name = ch.candidate_name(cand)
    _, ctx_a2, _ = ch.context("s")
    worlds = ch.Worlds(None, ctx_a2, "s")
    res = ch.adjudicate(name, cand, inc, tmp_path / "ch", worlds, log=lambda m: None)
    assert walked == [("s", "2016-01-01", "2019-12-31", 16, "twin_adjudicate", list(range(17, 33))),
                      ("data/w2", "2016-01-01", "2019-12-31", 16, "adjudicate", None)]
    assert res["scores"] == {"incumbent": 8.0, name: 12.0} and res["verdict"] == "candidate"   # one 32-copy incumbent
    assert res["winner"] == name and abs(res["threshold"] - 2 * (0.7 ** 2 * 2) ** 0.5) < 0.01 and res["weeks"] == 10
    assert led[0]["name"].startswith("challenge_ch_adjudicate_") and json.loads(led[0]["notes"])["winner"] == name
    # inside the seed band: a tie, the incumbent keeps its place
    monkeypatch.setattr(ch, "metrics_on_common", lambda *a, **k: (
        {n: {3: v, 6: v, 10: v} for n, v in (("incumbent", 8.0), (name, 8.5))},
        {n: pd.DataFrame({"week": adj_weeks, "top3": 0.0, "top6": 0.0, "top10": 0.0}) for n in ("incumbent", name)},
        10))
    r2 = ch.adjudicate(name, cand, inc, tmp_path / "ch", worlds, log=lambda m: None)
    assert r2["winner"] == "incumbent" and r2["verdict"] == "tie"


def test_window_table_scores_each_model_at_its_own_settings_with_the_sp500_row(tmp_path, monkeypatch):
    """The real window_table with the backtest helpers faked at the seams: own
    settings come from each model's OWN 2006-2015 walk on its own world."""
    import stocks_ml.backtest as bt
    sel_weeks = list(pd.date_range("2006-01-06", periods=60, freq="W-FRI"))
    adj_weeks = list(pd.date_range("2016-01-08", periods=10, freq="W-FRI"))
    inc = _walk(tmp_path / "inc" / "select" / "preds.parquet", sel_weeks, 16)
    cand_sel = _walk(tmp_path / "c" / "select" / "preds.parquet", sel_weeks, 16)
    spy = pd.Series(0.001, index=pd.DatetimeIndex(adj_weeks))
    ctx_a = SimpleNamespace(weeks=sel_weeks + adj_weeks, name="a", wret={"SPY": spy})
    ctx_b = SimpleNamespace(weeks=sel_weeks + adj_weeks, name="b")
    decided = []
    monkeypatch.setattr(bt, "holdings", lambda sel, ctx_, p, copies: pd.DataFrame({"week": sel_weeks, "top3": 0.01, "top6": 0.01, "top10": 0.01}))
    monkeypatch.setattr(bt, "own_settings", lambda sel, ctx_, h, lo=None, hi=None: (decided.append(ctx_.name),
                                                                                {"book": 3 if ctx_.name == "a" else 6, "floor": "halfgate", "stop": None, "cap": 2, "vol_cut": None})[1])
    monkeypatch.setattr(bt, "simulate_holdings", lambda sel, ctx_, h, st: pd.Series(0.002 if st["book"] == 3 else 0.001, index=pd.DatetimeIndex(adj_weeks)))
    monkeypatch.setattr(ch, "holdings", bt.holdings)
    sel = SimpleNamespace(metrics=lambda s, a, b: {"terminal_100": 101.0, "cagr_pct": 1.0, "sharpe": 0.5, "max_dd": 0.1})
    H = {"incumbent": pd.DataFrame({"week": adj_weeks}), "cand": pd.DataFrame({"week": adj_weeks})}
    out = ch.window_table(sel, ctx_a, ctx_b, "cand", inc, cand_sel, H, pd.Timestamp("2016-01-01"), pd.Timestamp("2019-12-31"))
    assert decided == ["a", "b"]                                    # each model's settings on its own world
    assert out["settings"] == {"incumbent": {"book": 3, "floor": "halfgate", "stop": None, "cap": 2, "vol_cut": None},
                               "cand": {"book": 6, "floor": "halfgate", "stop": None, "cap": 2, "vol_cut": None}}
    assert set(out["rows"]) == {"incumbent (top-3 / halfgate / stop None / cap 2)", "cand (top-6 / halfgate / stop None / cap 2)", "sp500"}
    assert out["md"][0].startswith("| model | 2016-2019: $100, %/yr | SR, DD |")


def test_stratified_blocks_are_runs_of_consecutive_rank_weeks():
    weeks = list(pd.date_range("2006-01-06", periods=104, freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks)
    b = ch.stratified_blocks(ctx, weeks[0], weeks[-1], per_year=8, seed=0, block=4)
    assert len(b) == 16 and len(set(b)) == 16                      # 2 runs of 4 per year, two years
    pos = {t: i for i, t in enumerate(weeks)}
    runs = [b[i:i + 4] for i in range(0, 16, 4)]
    assert all(pos[r[j + 1]] == pos[r[j]] + 1 for r in runs for j in range(3))
    assert ch.stratified_blocks(ctx, weeks[0], weeks[-1], per_year=8, seed=0, block=4) == b   # seeded


def test_refuse_leaky_features_judges_a_sector_relative_column_by_its_parent(monkeypatch):
    seen = {}
    def fake_check(ctx, feats, lo, hi):
        vals = {"f_a": 0.20, "x_sr_f_a": 0.23, "x_sr_f_b": 0.40, "f_b": 0.10, "x_new": 0.05, "f_c": float("nan"), "x_sr_f_c": -0.03}
        out = {f: {"corr_with_future_split_factor": vals[f], "weeks": 500, "verdict": "PASS" if abs(vals[f]) <= 0.15 else "FAIL"} for f in feats}
        out["VERDICT"] = "PASS" if all(v["verdict"] == "PASS" for v in out.values() if isinstance(v, dict)) else "FAIL"
        seen["feats"] = list(feats); return out
    import stocks_ml.leak_audit as la
    monkeypatch.setattr(la, "feature_factor_check", fake_check)
    ctx = SimpleNamespace(pan=pd.DataFrame(columns=["f_a", "f_b", "f_c", "x_sr_f_a", "x_sr_f_b", "x_sr_f_c", "x_new"]))
    ch.refuse_leaky_features(ctx, [{"features": ["x_sr_f_a", "x_new", "x_sr_f_c"]}], "2006-01-01", "2015-12-31", log=lambda m: None)   # 0.23 vs parent 0.20; a NaN parent -> the absolute line
    assert set(seen["feats"]) >= {"x_sr_f_a", "f_a", "x_new"}
    with pytest.raises(SystemExit, match="x_sr_f_b"):
        ch.refuse_leaky_features(ctx, [{"features": ["x_sr_f_b"]}], "2006-01-01", "2015-12-31", log=lambda m: None)   # 0.40 vs parent 0.10


def test_merged_copies_continues_the_numbering(tmp_path):
    weeks = list(pd.date_range("2006-01-06", periods=3, freq="W-FRI"))
    a = pd.read_parquet(_walk(tmp_path / "a" / "preds.parquet", weeks, 2))                 # c1, c2
    b = pd.read_parquet(_walk(tmp_path / "b" / "preds.parquet", weeks, 2, scale=2.0))      # c1, c2 (a twin cut to c1..c2)
    m = ch.merged_copies(a, b)
    assert [c for c in m.columns if c.startswith("c")] == ["c1", "c2", "c3", "c4"] and len(m) == len(a)
    assert (m["c3"] == 2 * m["c1"]).all()


def test_decide_needs_the_seed_band_and_the_weeks_to_agree():
    band = {"sd16": 1.0}
    assert ch.decide(10.0, 5.0, band, band, paired_t=2.5) == ("candidate", 5.0, 2 * 2 ** 0.5)
    assert ch.decide(10.0, 5.0, band, band, paired_t=1.5)[0] == "tie"          # the weeks do not agree
    assert ch.decide(10.0, 5.0, band, band, paired_t=-2.5)[0] == "tie"         # nor in that direction
    assert ch.decide(5.0, 10.0, band, band, paired_t=-2.5)[0] == "incumbent"
    assert ch.decide(6.0, 5.0, band, band, paired_t=3.0)[0] == "tie"           # inside the seed band
    assert ch.decide(10.0, 5.0, band, band)[0] == "candidate"                  # no t given: the band alone
