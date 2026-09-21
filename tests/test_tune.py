"""stocks-ml tune: the study's weeks, the search space, the statistics, the run on a fake fit."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("optuna")
import stocks_ml.tune as tu


def test_phase_weeks_are_non_overlapping_and_seeded():
    weeks = list(pd.date_range("2005-06-03", "2016-06-03", freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks)
    w = tu.phase_weeks(ctx, phase=0, seed=1)
    pos = {t: i for i, t in enumerate(weeks)}
    assert all(tu.SELECT[0] <= t <= tu.SELECT[1] for t in w) and len(set(w)) == len(w)
    assert all((pos[t] - pos[w[0]]) % 4 == 0 for t in w)                 # one phase: holds never overlap
    assert w == tu.phase_weeks(ctx, phase=0, seed=1) and w != tu.phase_weeks(ctx, phase=0, seed=2)
    assert not set(w) & set(tu.phase_weeks(ctx, phase=1, seed=1))


def test_stats_and_suggest_space():
    s = tu.stats([1.0, 2.0, 3.0])
    assert s["n"] == 3 and s["mean"] == 2.0 and abs(s["se"] - 1 / 3 ** 0.5) < 1e-9 and s["lo90"] < 2.0 < s["hi90"]
    import optuna
    import stocks_ml.selection as sel
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({k: v for k, v in sel.MODEL_PARAMS.items() if k in tu.SPACE})
    trial = study.ask()
    p = tu.suggest(trial, {})
    assert p == {k: sel.MODEL_PARAMS[k] for k in tu.SPACE}                # trial 0 is the champion's settings
    assert isinstance(p["max_depth"], int) and isinstance(p["learning_rate"], float)


def test_run_scores_trials_on_common_weeks_and_prunes_losers(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", "2015-12-31", freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks, pan=pd.DataFrame(columns=["x_dollar_vol"]))
    sel = SimpleNamespace(slice_row=lambda c, t, h, p: {"top3": float(p["A"]), "top6": float(p["A"]), "top10": float(p["A"]),
                                                        "rand_mean": 0.0, "spy": 0.0})
    import stocks_ml.train as train
    monkeypatch.setattr(train, "context", lambda store: (sel, ctx, 64))
    monkeypatch.setattr(train, "fit_context", lambda *a, **k: ctx)
    monkeypatch.setattr(train, "thread_budget", lambda w: (1, 1))
    import stocks_ml.challenger2 as c2
    monkeypatch.setattr(c2, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(tu, "champion_walks_for", lambda *a, **k: [])
    seen = []
    def fake_fit_pair(args):
        t, need_c, need_a, champ, alt = args
        seen.append((t, tuple(need_c), tuple(need_a)))
        depth = alt["params"].get("max_depth", 3)
        bump = 0.004 if depth == 3 else -0.01                          # only the champion's depth is any good
        def fit(seed, b):
            return pd.Series({"A": np.random.default_rng(seed * 7919 + int(t.value // 10 ** 9) % 997).normal(0.01, 0.002) + b})
        return t, {s: fit(s, 0.0) for s in need_c}, {s: fit(s, bump) for s in need_a}
    monkeypatch.setattr(tu, "_fit_pair", fake_fit_pair)
    monkeypatch.setattr("stocks_ml.models.trials.record_trials", lambda rows: None)
    spec = {"horizon": {"label": "label_4w_sector_rank"}, "training_window_years": 8, "features": ["x_dollar_vol"],
            "drop_features": ["f_dollar_vol"], "model": {"params": {}}, "procedure": {"preds": {"path": str(tmp_path / "none.parquet")}}}
    sp = tmp_path / "spec.json"; sp.write_text(json.dumps(spec))
    res = tu.run("toy", trials=6, copies=2, batch=20, workers=1, seed=0, store="s", out_dir=tmp_path / "out",
                 study_dir=tmp_path / "studies", spec_path=sp, log=lambda m: None)
    assert res["complete"] >= 1 and res["trials"] == 6 and (tmp_path / "out" / "tune_toy.png").exists()
    assert res["best"]["trial"] == 0 and abs(res["best"]["diff"] - 0.4) < 0.05        # the champion's own settings win, +0.4 pp/hold
    weeks_seen = {t for t, *_ in seen}
    assert all(tu.SELECT[0] <= t <= tu.SELECT[1] for t in weeks_seen)
    # the champion side was fitted once per week and then served from the cache
    first = [nc for t, nc, na in seen if t == seen[0][0]]
    assert len(first[0]) == 2 and all(nc == () for nc in first[1:])


def test_universe_trials_carve_a_world_per_n_and_score_the_trial_there(tmp_path, monkeypatch):
    weeks = list(pd.date_range("2006-01-06", "2015-12-31", freq="W-FRI"))
    def ctx_for(name):
        return SimpleNamespace(weeks=weeks, pan=pd.DataFrame(columns=["x_dollar_vol"]), name=name)
    champ_ctx = ctx_for("champ")
    sel = SimpleNamespace(slice_row=lambda c, t, h, p: {"top3": float(p["A"]), "top6": float(p["A"]), "top10": float(p["A"]),
                                                        "rand_mean": 0.0, "spy": 0.0})
    import stocks_ml.train as train
    monkeypatch.setattr(train, "context", lambda store: (sel, champ_ctx if store == "s" else ctx_for(store), 64))
    monkeypatch.setattr(train, "fit_context", lambda c, *a, **k: c)
    monkeypatch.setattr(train, "thread_budget", lambda w: (1, 1))
    import stocks_ml.challenger2 as c2
    monkeypatch.setattr(c2, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(tu, "champion_walks_for", lambda *a, **k: [])
    monkeypatch.setattr(tu, "WORLD_DIR", str(tmp_path / "uni{n}"))
    carved = []
    def fake_derive(root, parent, cfg, n, log=None, membership_from=None, sized=False, **k):
        carved.append((n, sized)); Path(root).mkdir(parents=True, exist_ok=True); (Path(root) / "panel_sf.parquet").write_bytes(b"x")
    monkeypatch.setattr("stocks_ml.data.world.derive_research_store", fake_derive)
    scored_on = []
    real_book = tu.book_returns
    def fake_book(sel_, c, t, p):
        scored_on.append(getattr(c, "name", "?")); return real_book(sel_, c, t, p)
    monkeypatch.setattr(tu, "book_returns", fake_book)
    def fake_fit_pair(args):
        t, need_c, need_a, champ, alt = args
        def fit(seed, b):
            return pd.Series({"A": np.random.default_rng(seed * 7919 + int(t.value // 10 ** 9) % 997).normal(0.01, 0.002) + b})
        return t, {s: fit(s, 0.0) for s in need_c}, {s: fit(s, 0.002) for s in need_a}
    monkeypatch.setattr(tu, "_fit_pair", fake_fit_pair)
    monkeypatch.setattr("stocks_ml.models.trials.record_trials", lambda rows: None)
    spec = {"horizon": {"label": "label_4w_sector_rank"}, "training_window_years": 8, "features": ["x_dollar_vol"],
            "drop_features": ["f_dollar_vol"], "model": {"params": {}}, "procedure": {"preds": {"path": str(tmp_path / "none.parquet")}}}
    sp = tmp_path / "spec.json"; sp.write_text(json.dumps(spec))
    # the champion cache must hold every seed the trials need (the champion is never fit on a trial world)
    cache = c2.PredCache("s", c2.champion_recipe(sp), cache_dir=tmp_path / "cache")
    for t in weeks:
        for s in range(1, 65):
            cache.put(t, s, pd.Series({"A": 0.01}))
    cache.save()
    res = tu.run("uni", trials=4, copies=2, batch=20, workers=1, seed=0, store="s", out_dir=tmp_path / "out",
                 study_dir=tmp_path / "studies", spec_path=sp, log=lambda m: None, universe=True, cfg=object())
    import optuna
    st = optuna.load_study(study_name="uni", storage=f"sqlite:///{tmp_path / 'studies' / 'uni.db'}")
    assert st.trials[0].params["universe_n"] == 500 and st.trials[0].params["train_years"] == 8   # the anchor: the champion at the index's size
    assert all("universe_n" in t.params for t in st.trials)
    assert carved and all(sized for _, sized in carved) and all(n % 100 == 0 and 300 <= n <= 2000 for n, _ in carved)
    assert any(name.startswith(str(tmp_path / "uni")) for name in scored_on)   # the trial's picks scored on its own world
