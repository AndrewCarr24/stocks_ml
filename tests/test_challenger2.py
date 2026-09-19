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
    assert len(d) == 200 and all(c2.SELECT[0] <= t <= c2.SELECT[1] for t, _ in d)
    assert all(c2.SEED_RANGE[0] <= s < c2.SEED_RANGE[1] for _, s in d) and d == c2.draws(ctx, 200, seed=1)


def test_running_mean_convergence_and_summary():
    rng = np.random.default_rng(0)
    v = list(rng.normal(1.0, 5.0, 300))
    m, se = running(v) if False else c2.running(v)
    assert np.isclose(m[-1], np.mean(v)) and np.isnan(se[0]) and np.isclose(se[-1], np.std(v, ddof=1) / np.sqrt(300))
    assert c2.converged(v[:30], min_iters=50, se_tol=1.0) is None            # too few
    assert c2.converged(v, min_iters=50, se_tol=0.5) == "decided"            # mean 1.0, SE ~0.3: the CI excludes zero
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
    assert r == {"top3": 3.0, "top6": 2.0, "top10": 1.0, "avg": 2.0, "universe": 0.5}
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


def test_run_pairs_the_week_and_seed_and_stops_when_converged(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", "2015-12-31", freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks, pan=pd.DataFrame(columns=["x_dollar_vol"]))
    sel = SimpleNamespace(slice_row=lambda c, t, h, p: {"top3": float(p["A"]), "top6": float(p["A"]), "top10": float(p["A"]),
                                                        "rand_mean": 0.0, "spy": 0.0})
    import stocks_ml.train as train
    monkeypatch.setattr(train, "context", lambda store: (sel, ctx, 64))
    monkeypatch.setattr(train, "fit_context", lambda *a, **k: ctx)
    monkeypatch.setattr(train, "thread_budget", lambda w: (1, 1))
    seen = []
    def fake_fit_pair(args):
        t, seed, champ, alt = args
        seen.append((t, seed, champ["params"], alt["params"]))
        rng = np.random.default_rng(seed)
        base = rng.normal(0.01, 0.002)
        return t, seed, pd.Series({"A": base}), pd.Series({"A": base + 0.003})     # the altered model is 0.3 pp better
    monkeypatch.setattr(c2, "_fit_pair", fake_fit_pair)
    monkeypatch.setattr("stocks_ml.models.trials.record_trials", lambda rows: None)
    spec = {"horizon": {"label": "label_4w_sector_rank"}, "training_window_years": 8, "features": ["x_dollar_vol"],
            "drop_features": ["f_dollar_vol"], "model": {"params": {}}}
    sp = tmp_path / "spec.json"; sp.write_text(json.dumps(spec))
    res = c2.run("params=max_depth:5", store="s", workers=1, min_iters=20, max_iters=200, se_tol=0.05,
                 out_dir=tmp_path / "c2", spec_path=sp, log=lambda m: None)
    s = res["summary"]
    assert s["converged"] and s["stop_reason"] == "decided" and 20 <= s["iterations"] < 200
    assert abs(s["diff_mean"] - 0.3) < 0.02 and s["diff_se"] <= 0.05          # the paired difference, exact to the noise
    assert all(champ == {} and alt == {"max_depth": 5} for _, _, champ, alt in seen)   # same week and seed, one change
    assert (tmp_path / "c2" / "max_depth_5.png").exists() and not list((tmp_path / "c2").glob("*.json"))   # the plot is the record


def test_altered_recipe_may_name_a_training_world():
    champ = {"label": "label_4w_sector_rank", "train_years": 8, "features": ["x_dollar_vol"], "drop": ["f_dollar_vol"], "params": {}, "train_top": None}
    alt = c2.altered_recipe("store=data/sharadar_sp800_nominal_dl", champ)
    assert alt["store"] == "data/sharadar_sp800_nominal_dl" and {k: v for k, v in alt.items() if k != "store"} == champ
    assert c2.experiment_name("store=data/sharadar_sp800_nominal_dl") == "store_sp800_nominal_dl"
