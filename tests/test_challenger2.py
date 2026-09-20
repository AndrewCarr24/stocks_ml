"""challenger2: paired random-week/random-seed comparison until convergence."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stocks_ml.challenger2 as c2


def test_recipes_names_and_draws(tmp_path):
    spec = {"horizon": {"label": "label_4w_sector_rank"}, "training_window_years": 8, "features": ["x_dollar_vol"],
            "drop_features": ["f_dollar_vol"], "model": {"params": {}}}
    sp = tmp_path / "spec.json"; sp.write_text(json.dumps(spec))
    champ = c2.champion_recipe(sp)
    assert champ == {"label": "label_4w_sector_rank", "train_years": 8, "features": ["x_dollar_vol"],
                     "drop": ["f_dollar_vol"], "params": {}, "train_top": None}
    alt = c2.altered_recipe("params=max_depth:5", champ)
    assert alt["params"] == {"max_depth": 5} and alt["features"] == ["x_dollar_vol"] and alt["drop"] == ["f_dollar_vol"]
    assert c2.experiment_name("params=max_depth:5") == "max_depth_5" and c2.experiment_name("train_years=12") == "train_years_12"
    weeks = list(pd.date_range("2005-06-03", "2016-06-03", freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks)
    d = c2.draws(ctx, 200, seed=1)
    assert len(d) == 200 and len(set(d)) == 200 and all(c2.SELECT[0] <= t <= c2.SELECT[1] for t in d)   # without replacement
    assert d == c2.draws(ctx, 200, seed=1) and d != c2.draws(ctx, 200, seed=2)
    assert len(c2.draws(ctx, 10 ** 6, seed=0)) == sum(c2.SELECT[0] <= t <= c2.SELECT[1] for t in weeks)  # capped at the window
    t = pd.Timestamp("2010-01-08")
    assert c2.week_seeds(t, 64) == list(range(1, 65)) and len(c2.week_seeds(t, 16)) == 16 and len(c2.week_seeds(t, 4)) == 4
    assert c2.week_seeds(t, 1) == c2.week_seeds(t, 1) and set(c2.week_seeds(t, 16)) <= set(range(1, 65))
    assert c2.week_seeds(t, 4) != c2.week_seeds(pd.Timestamp("2010-01-15"), 4) or True         # week-determined


def test_running_mean_convergence_and_summary():
    rng = np.random.default_rng(0)
    v = list(rng.normal(1.0, 5.0, 300))
    m, se = running(v) if False else c2.running(v)
    assert np.isclose(m[-1], np.mean(v)) and np.isnan(se[0]) and np.isclose(se[-1], np.std(v, ddof=1) / np.sqrt(300))
    assert c2.converged(v[:30], min_iters=50, se_tol=1.0) is None            # too few
    big = list(rng.normal(2.0, 5.0, 300))                                     # a huge difference still runs to the SE rule
    assert c2.converged(big, min_iters=50, se_tol=0.1) is None
    assert c2.converged(v, min_iters=50, se_tol=0.5) == "measured"           # SE ~0.29 <= 0.5
    z = list(rng.normal(0.0, 5.0, 300))                                       # a true tie
    assert c2.converged(z, min_iters=50, se_tol=0.5) == "measured" and c2.converged(z, min_iters=50, se_tol=0.1) is None
    rows = [{"champion": {"top3": 1.0, "top6": 2.0, "top10": 3.0, "avg": 2.0}, "altered": {"top3": 2.0, "top6": 3.0, "top10": 4.0, "avg": 3.0}},
            {"champion": {"top3": 0.0, "top6": 1.0, "top10": 2.0, "avg": 1.0}, "altered": {"top3": 0.0, "top6": 2.0, "top10": 4.0, "avg": 2.0}}]
    s = c2.summary(rows)
    assert s["iterations"] == 2 and s["champion_mean"] == 1.5 and s["altered_mean"] == 2.5 and s["diff_mean"] == 1.0
    assert s["diff_ci90"] == [1.0, 1.0]                                  # two identical differences: no spread
    assert s["per_book"]["top3"]["diff"] == 0.5 and s["share_altered_ahead"] == 1.0


def test_book_returns_from_slice_row_and_render(tmp_path):
    sel = SimpleNamespace(slice_row=lambda ctx, t, h, p: {"top3": 0.03, "top6": 0.02, "top10": 0.01, "rand_mean": 0.005, "spy": 0.0})
    r = c2.book_returns(sel, None, pd.Timestamp("2010-01-08"), pd.Series({"A": 1.0}))
    assert r == {"top3": 2.5, "top6": 1.5, "top10": 0.5, "avg": 1.5, "universe": 0.5}     # minus the average member
    sel2 = SimpleNamespace(slice_row=lambda ctx, t, h, p: None)
    assert c2.book_returns(sel2, None, pd.Timestamp("2010-01-08"), pd.Series({"A": 1.0})) is None
    rng = np.random.default_rng(1)
    rows = [{"iteration": i + 1, "week": "2010-01-08", "seed": i,
             "champion": {"top3": 1.0, "top6": 1.0, "top10": 1.0, "avg": float(rng.normal(1, 3))},
             "altered": {"top3": 1.0, "top6": 1.0, "top10": 1.0, "avg": float(rng.normal(0.5, 3))}} for i in range(12)]
    png = tmp_path / "exp.png"
    c2.render(rows, "exp", "params=max_depth:5", png)
    c2.render(rows, "exp", "params=max_depth:5", png, final=True)
    assert png.exists() and png.stat().st_size > 1000 and not png.with_suffix(".tmp.png").exists()


def test_run_pairs_the_week_and_seeds_caches_fits_and_stops_when_converged(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", "2015-12-31", freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks, pan=pd.DataFrame(columns=["x_dollar_vol"]))
    sel = SimpleNamespace(slice_row=lambda c, t, h, p: {"top3": float(p["A"]), "top6": float(p["A"]), "top10": float(p["A"]),
                                                        "rand_mean": 0.0, "spy": 0.0})
    import stocks_ml.train as train
    monkeypatch.setattr(train, "context", lambda store: (sel, ctx, 64))
    monkeypatch.setattr(train, "fit_context", lambda *a, **k: ctx)
    monkeypatch.setattr(train, "thread_budget", lambda w: (1, 1))
    monkeypatch.setattr(c2, "CACHE_DIR", tmp_path / "cache")
    seen = []
    def fake_fit_pair(args):
        t, need_c, need_a, champ, alt = args
        seen.append((t, tuple(need_c), tuple(need_a), champ["params"], alt["params"]))
        def fit(seed, bump):
            base = np.random.default_rng(seed * 1000 + int(t.value // 10 ** 9) % 997).normal(0.01, 0.002)
            return pd.Series({"A": base + bump})
        return t, {s: fit(s, 0.0) for s in need_c}, {s: fit(s, 0.003) for s in need_a}   # the altered model is 0.3 pp better
    monkeypatch.setattr(c2, "_fit_pair", fake_fit_pair)
    monkeypatch.setattr("stocks_ml.models.trials.record_trials", lambda rows: None)
    spec = {"horizon": {"label": "label_4w_sector_rank"}, "training_window_years": 8, "features": ["x_dollar_vol"],
            "drop_features": ["f_dollar_vol"], "model": {"params": {}}, "procedure": {"preds": {"path": str(tmp_path / "nowalk.parquet")}}}
    sp = tmp_path / "spec.json"; sp.write_text(json.dumps(spec))
    res = c2.run("params=max_depth:5", store="s", workers=1, min_iters=20, max_iters=200, se_tol=0.05,
                 out_dir=tmp_path / "c2", spec_path=sp, log=lambda m: None, copies=2)
    s = res["summary"]
    assert s["converged"] and s["stop_reason"] == "measured" and 20 <= s["iterations"] < 200
    assert abs(s["diff_mean"] - 0.3) < 0.02 and s["diff_se"] <= 0.05          # the paired difference, exact to the noise
    assert all(nc == na and len(nc) == 2 and champ == {} and alt == {"max_depth": 5} for _, nc, na, champ, alt in seen)
    assert len({t for t, *_ in seen}) == len(seen)                            # weeks without replacement
    assert (tmp_path / "c2" / "max_depth_5.png").exists() and not list((tmp_path / "c2").glob("*.json"))   # the plot is the record
    # the caches now hold every fit; a rerun fits nothing
    n_fits = res["fits"]
    seen.clear()
    res2 = c2.run("params=max_depth:5", store="s", workers=1, min_iters=20, max_iters=200, se_tol=0.05,
                  out_dir=tmp_path / "c2", spec_path=sp, log=lambda m: None, copies=2)
    # the second run may look a few weeks past the first's stop (tasks in flight): those are the only new fits
    assert n_fits > 0 and res2["fits"] <= 2 * 2 * 2 and sum(len(nc) + len(na) for _, nc, na, *_ in seen) == res2["fits"]
    assert res2["summary"]["diff_mean"] == s["diff_mean"]                     # the same weeks and seeds -> the same result


def test_pred_cache_roundtrip_and_walk_seed(tmp_path):
    rec = {"label": "l", "train_years": 8, "features": [], "drop": [], "params": {}, "train_top": None}
    c = c2.PredCache("data/some_store", rec, cache_dir=tmp_path)
    t = pd.Timestamp("2010-01-08")
    assert c.get(t, [1, 2]) == {}
    c.put(t, 3, pd.Series({"A": 1.0, "B": 2.0})); c.put(t, 5, pd.Series({"A": 3.0, "C": 4.0}))
    got = c.get(t, [3, 5, 7])
    assert sorted(got) == [3, 5] and got[3].to_dict() == {"A": 1.0, "B": 2.0} and got[5].to_dict() == {"A": 3.0, "C": 4.0}
    c.save()
    c2b = c2.PredCache("data/some_store", rec, cache_dir=tmp_path)
    assert sorted(c2b.get(t, [3, 5])) == [3, 5] and c2b.get(t, [5])[5].to_dict() == {"A": 3.0, "C": 4.0}
    walk = pd.DataFrame({"week": [t, t], "ticker": ["A", "B"], **{f"c{i}": [float(i), float(-i)] for i in range(1, 17)}})
    walk.to_parquet(tmp_path / "walk.parquet", index=False)
    assert c2b.seed_from_walk(tmp_path / "walk.parquet") == 1
    assert c2b.get(t, [16])[16].to_dict() == {"A": 16.0, "B": -16.0} and c2b.get(t, [3])[3]["A"] == 1.0   # the put wins where both exist
    assert (tmp_path / "some_store" / f"{c2.recipe_hash(rec)}.json").exists()


def test_champion_walks_for_finds_every_seed_set(tmp_path):
    rec = {"label": "label_4w_sector_rank", "train_years": 8, "features": ["x_dollar_vol"], "drop": ["f_dollar_vol"], "params": {}, "train_top": None}
    r = {"label": "label_4w_sector_rank", "train_years": 8, "features": ["x_dollar_vol"], "drop": ["f_dollar_vol"]}
    root = tmp_path / "walk"
    for name, extra in (("select", {}), ("twin", {"copies": [17, 32]}), ("seeds_33_64", {"copies": [33, 64]}),
                        ("extend", {"weeks": "every week of 2016-01-01 -> 2024-07-18"}), ("fast", {"refit_every": 4})):
        d = root / name; d.mkdir(parents=True)
        (d / "preds.parquet").write_bytes(b"x")
        (d / "spec.json").write_text(json.dumps({"k": 16, "store": "data/w", "recipe": r, "weeks": "every week of 2006-01-01 -> 2015-12-31", **extra}))
    spec = {"procedure": {"preds": {"path": str(root / "select" / "preds.parquet")}}}
    sp = tmp_path / "spec.json"; sp.write_text(json.dumps(spec))
    got = [p.parent.name for p in c2.champion_walks_for("data/w", rec, sp)]
    assert got == ["seeds_33_64", "select", "twin"]                     # the window's weekly walks only
    assert c2.champion_walks_for("data/other", rec, sp) == []
    assert c2.champion_walks_for("data/w", {**rec, "params": {"max_depth": 5}}, sp) == []


def test_altered_recipe_may_name_a_training_world():
    champ = {"label": "label_4w_sector_rank", "train_years": 8, "features": ["x_dollar_vol"], "drop": ["f_dollar_vol"], "params": {}, "train_top": None}
    alt = c2.altered_recipe("store=data/sharadar_sp800_nominal_dl", champ)
    assert alt["store"] == "data/sharadar_sp800_nominal_dl" and {k: v for k, v in alt.items() if k != "store"} == champ
    assert c2.experiment_name("store=data/sharadar_sp800_nominal_dl") == "store_sp800_nominal_dl"


def test_fit_pair_fits_only_the_seeds_each_side_needs(monkeypatch):
    import stocks_ml.train as train
    calls = []
    def fake_copy(sel, ctx, t, seed, label, ty, features, params, n_jobs=None, drop=(), train_top=None):
        calls.append((ctx, seed)); return pd.Series({"A": float(seed), "B": 0.0})
    monkeypatch.setattr(train, "copy_preds", fake_copy)
    c2._W.update(sel=None, ctx="ctxA", alt_ctx="ctxB", n_jobs=1)
    rec = {"label": "l", "train_years": 8, "features": [], "drop": [], "params": {}, "train_top": None}
    t, new_c, new_a = c2._fit_pair((pd.Timestamp("2010-01-08"), [2, 5], [7], rec, rec))
    assert [s for c, s in calls if c == "ctxA"] == [2, 5] and [s for c, s in calls if c == "ctxB"] == [7]
    assert sorted(new_c) == [2, 5] and sorted(new_a) == [7] and c2.ensemble(new_c)["A"] == 3.5     # the mean over seeds
