"""stocks-ml tune: hyperparameter search inside the challenger2 framework
(owner's request, 2026-09-20), starting from the champion's settings.

Objective: a trial's paired difference over the champion in the three-book
excess per hold (challenger2's statistic: the mean 4-week return of the
top-3/6/10 minus the week's average member, mean of the three) on ONE fixed
list of weeks and seeds shared by every trial — common random numbers, so
trials are paired with each other as well as with the champion. The weeks
come from one 4-week phase of the selection window (no overlapping holds:
an honest SE), in a seeded order; each trial fits `copies` seeded copies per
week (the week's seeds from challenger2's pool) and the champion's side is
read from its cache (its K=64 walks). Trials are scored in batches; Optuna's
median pruner stops the ones that fall behind, and a trial whose 90% upper
bound on the difference is below zero after two batches is stopped as a
clear loser. Trial 0 is the champion's own settings.

The search space is XGBoost's settings the champion fixes by decision
(selection.MODEL_PARAMS): depth, learning rate, leaf floor, row and column
sampling, the regularizers; the number of trees stays at early stopping.
With --universe the space also carries the universe (owner 2026-09-20:
"start at the companies in the S&P 500 and tune from there"): `universe_n`
names each quarter — the index's largest n members below the index's size,
the index topped up with the largest non-members above it — each n its own
world carved once from the top-2000 store (data/sharadar_uni<n>_nominal_dl,
the champion's recipe fit, picked and scored there), plus the training
window and the label family. The champion's side stays the deployed model on
its own data (its picks are what the trial's picks must beat). Nothing is
adopted here: the best trial is a candidate for the challenge.

Outputs: challenger2_convergence/tune_<name>.png (every trial's difference
with its 90% interval), the study in data/tune/<name>.db (resumable), a
ledger row for the best trial.
"""
from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.challenger2 import (COLORS, PredCache, SELECT, Z, _fit_pair, _init, book_returns, champion_recipe,
                                   champion_walks_for, ensemble, week_seeds)

OUT_DIR = Path("challenger2_convergence")
STUDY_DIR = Path("data/tune")
PHASE = 0                 # weeks of the window in one 4-week phase: non-overlapping holds
BATCH = 26                # weeks per batch (a report to the pruner after each)
COPIES = 4
SPACE = {                 # name: (low, high, log)
    "max_depth": (2, 6, False), "learning_rate": (0.005, 0.1, True), "min_child_weight": (5, 200, True),
    "subsample": (0.5, 1.0, False), "colsample_bytree": (0.4, 1.0, False), "reg_lambda": (0.1, 10.0, True),
    "reg_alpha": (0.0, 2.0, False), "gamma": (0.0, 0.1, False)}
UNIVERSE = {"universe_n": (300, 2000, 100), "train_years": (5, 12, 1), "label": ("label_4w_sector_rank", "label_4w_sector_log")}
PARENT = "data/sharadar_top2000_nominal_dl"     # the worlds are carved from here ...
INDEX_STORE = "data/sharadar_world2000_nominal_dl"   # ... around this store's index membership
WORLD_DIR = "data/sharadar_uni{n}_nominal_dl"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def phase_weeks(ctx, phase: int = PHASE, seed: int = 0, lo=SELECT[0], hi=SELECT[1]) -> list[pd.Timestamp]:
    """The window's rank weeks in one 4-week phase (non-overlapping holds),
    in a seeded random order: the study's common list."""
    weeks = [t for t in ctx.weeks if lo <= t <= hi][phase::4]
    rng = np.random.default_rng(seed)
    return [weeks[i] for i in rng.permutation(len(weeks))]


def suggest(trial, base: dict) -> dict:
    """A trial's params: the search space around the champion's settings;
    trial 0 (enqueued) is the champion's own."""
    import stocks_ml.selection as sel
    out = {}
    for name, (lo, hi, is_log) in SPACE.items():
        if isinstance(sel.MODEL_PARAMS[name], int):
            out[name] = trial.suggest_int(name, int(lo), int(hi), log=is_log)
        else:
            out[name] = trial.suggest_float(name, lo, hi, log=is_log)
    return out


def suggest_universe(trial) -> dict:
    """The universe dial and the recipe's other searched fields."""
    lo, hi, step = UNIVERSE["universe_n"]
    return {"universe_n": trial.suggest_int("universe_n", lo, hi, step=step),
            "train_years": trial.suggest_int("train_years", *UNIVERSE["train_years"][:2]),
            "label": trial.suggest_categorical("label", list(UNIVERSE["label"]))}


def ensure_world(n: int, cfg, log=log) -> str:
    """The sized world for n, carved once from the top-2000 store."""
    from stocks_ml.data.world import derive_research_store
    root = WORLD_DIR.format(n=n)
    if not Path(root, "panel_sf.parquet").exists():
        log(f"carving the universe of {n} names -> {root}")
        derive_research_store(root, PARENT, cfg, n=n, log=log, membership_from=INDEX_STORE, sized=True)
    return root


def stats(diffs: list[float]) -> dict:
    d = np.asarray(diffs, float)
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "mean": float(d.mean()), "se": se, "lo90": float(d.mean() - Z * se), "hi90": float(d.mean() + Z * se)}


def render(study, name: str, png: Path, champ_score: float | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for t in study.trials:
        s = t.user_attrs.get("stats")
        if s:
            rows.append((t.number, s["mean"], s["se"], s["n"], t.state.name))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for num, m, se, n, state in rows:
        color = COLORS["altered"] if state == "COMPLETE" else "#c3c2b7"
        ax.errorbar(num, m, yerr=Z * se if se == se else 0, fmt="o", color=color, ms=5, capsize=2, lw=1)
    best = max((r for r in rows if r[4] == "COMPLETE"), key=lambda r: r[1], default=None)
    if best:
        ax.annotate(f"best {best[1]:+.2f} (trial {best[0]}, {best[3]} weeks)", (best[0], best[1]), xytext=(8, 10),
                    textcoords="offset points", fontsize=9, color="#52514e")
    ax.axhline(0, color=COLORS["champion"], lw=2, label="champion (trial 0's settings)")
    ax.set_xlabel("trial (pruned trials in grey)")
    ax.set_ylabel("paired difference over the champion\n(three-book excess, pp per hold; 90% interval)")
    ax.set_ylim(-6, 6)
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    done = sum(1 for r in rows if r[4] == "COMPLETE")
    ax.set_title(f"tune — {name}: {len(rows)} trials ({done} complete), objective = paired three-book excess vs the champion "
                 f"on one 4-week phase, {COPIES} copies", fontsize=10)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    tmp = png.with_suffix(".tmp.png"); fig.savefig(tmp, dpi=140); plt.close(fig); tmp.replace(png)


def run(name: str, trials: int = 40, copies: int = COPIES, batch: int = BATCH, workers: int = 6, seed: int = 0,
        phase: int = PHASE, store: str | None = None, out_dir: Path = OUT_DIR, study_dir: Path = STUDY_DIR,
        spec_path: Path | None = None, log=log, universe: bool = False, cfg=None) -> dict:
    import gc
    import optuna
    from stocks_ml.train import STORE, context, fit_context, thread_budget
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    store = store or STORE
    champ = champion_recipe(spec_path)
    sel, ctx, _ = context(store)
    weeks = phase_weeks(ctx, phase, seed)
    cache = PredCache(store, champ)
    if not cache.frames:
        for walk in champion_walks_for(store, champ, spec_path):
            cache.seed_from_walk(walk)
    caches = {}                                   # recipe hash -> PredCache, for trials that repeat a setting
    workers, n_jobs = thread_budget(workers)
    if universe and cfg is None:
        from stocks_ml.config import load_config
        cfg = load_config()
    pool = {"store": None, "ex": None, "ctx": None}   # the trial world's pool and scoring context (per world)

    def use_world(trial_store: str, span: int):
        """The pool (and scoring context) for a trial world; rebuilt when the world changes."""
        if pool["store"] == trial_store:
            return pool["ex"], pool["ctx"]
        if pool["ex"] is not None:
            pool["ex"].shutdown(cancel_futures=True)
        pool["ex"] = pool["ctx"] = None; gc.collect()
        if trial_store == store:
            t_ctx = ctx
        else:
            _, t_ctx, _ = context(trial_store)
        t_fit = fit_context(t_ctx, tuple(champ["features"]), SELECT[0], SELECT[1], span)
        ex = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init, initargs=(t_fit, n_jobs, t_fit)) if workers > 1 else None
        if ex is None:
            _init(t_fit, n_jobs, t_fit)
        pool.update(store=trial_store, ex=ex, ctx=t_ctx)
        return ex, t_ctx
    study_dir = Path(study_dir); study_dir.mkdir(parents=True, exist_ok=True)
    png = Path(out_dir) / f"tune_{name}.png"
    study = optuna.create_study(study_name=name, storage=f"sqlite:///{study_dir / name}.db", load_if_exists=True,
                                direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    import stocks_ml.selection as sel_mod
    champ_params = {k: v for k, v in sel_mod.MODEL_PARAMS.items() if k in SPACE}
    if not study.trials and not universe:
        study.enqueue_trial(champ_params)
    if universe and not any("universe_n" in t.params for t in study.trials):
        # the anchors: the champion's settings and the study's best so far, each at the index's size (n=500)
        done = [t for t in study.trials if t.state.name == "COMPLETE" and t.user_attrs.get("stats")]
        best = max(done, key=lambda t: t.user_attrs["stats"]["mean"]) if done else None
        anchor = {"universe_n": 500, "train_years": champ["train_years"], "label": champ["label"]}
        study.enqueue_trial({**champ_params, **anchor})
        if best is not None and best.params != champ_params:
            study.enqueue_trial({**{k: best.params[k] for k in SPACE}, **anchor})
    log(f"tune {name}: {trials} trials, {len(weeks)} weeks (phase {phase} of the window, seeded order), {copies} copies per week, "
        f"batches of {batch}; champion cache {len(cache.frames)} weeks; {workers} workers; study {study_dir / name}.db")
    fits_total = 0

    def champion_side(t, seeds):
        have = cache.get(t, seeds)
        need = [s for s in seeds if s not in have]
        return have, need

    def objective(trial):
        nonlocal fits_total
        params = suggest(trial, champ)
        alt = {**champ, "params": params}
        trial_store = store
        if universe:
            u = suggest_universe(trial)
            alt = {**alt, "train_years": int(u["train_years"]), "label": u["label"]}
            trial_store = ensure_world(int(u["universe_n"]), cfg, log)
            trial.set_user_attr("store", trial_store)
        ex, t_ctx = use_world(trial_store, alt["train_years"])
        from stocks_ml.challenger2 import recipe_hash
        h = recipe_hash({**alt, "store": trial_store})
        if h not in caches:
            caches[h] = PredCache(trial_store, alt)
        tc = caches[h]
        diffs, t0 = [], time.time()
        for b in range(0, len(weeks), batch):
            chunk = weeks[b:b + batch]
            tasks = []
            for t in chunk:
                seeds = week_seeds(t, copies)
                have_c, need_c = champion_side(t, seeds)
                if need_c and trial_store != store:
                    raise SystemExit(f"the champion's cache lacks seeds {need_c} at {t.date()}; the champion is fit on its "
                                     f"own world only — walk them first (stocks-ml train --copies)")
                have_a = tc.get(t, seeds); need_a = [s for s in seeds if s not in have_a]
                tasks.append((t, need_c, need_a, champ, alt, have_c, have_a))
            results = list(ex.map(_fit_pair, [(t, nc, na, c, a) for t, nc, na, c, a, _, _ in tasks])) if ex else \
                [_fit_pair((t, nc, na, c, a)) for t, nc, na, c, a, _, _ in tasks]
            for (t, nc, na, c, a, have_c, have_a), (_, new_c, new_a) in zip(tasks, results):
                fits_total += len(new_c) + len(new_a)
                for s_, p in new_c.items():
                    cache.put(t, s_, p)
                for s_, p in new_a.items():
                    tc.put(t, s_, p)
                pc, pa = ensemble({**have_c, **new_c}), ensemble({**have_a, **new_a})
                bc = book_returns(sel, ctx, t, pc) if pc is not None else None
                ba = book_returns(sel, t_ctx, t, pa) if pa is not None else None      # the trial's picks on its own world
                if bc is not None and ba is not None:
                    diffs.append(ba["avg"] - bc["avg"])
            s = stats(diffs)
            trial.set_user_attr("stats", s)
            step = b // batch
            trial.report(s["mean"], step)
            log(f"  trial {trial.number} {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in trial.params.items()})}: "
                f"{s['n']} weeks, diff {s['mean']:+.2f} (90% {s['lo90']:+.2f} to {s['hi90']:+.2f}), {(time.time() - t0) / 60:.1f} min")
            tc.save(); cache.save()
            render(study, name, png)
            if step >= 1 and s["hi90"] < 0:
                log(f"  trial {trial.number}: a clear loser (90% upper bound {s['hi90']:+.2f} < 0); stopped")
                raise optuna.TrialPruned()
            if trial.should_prune():
                log(f"  trial {trial.number}: pruned by the median rule at step {step}")
                raise optuna.TrialPruned()
        return s["mean"]

    try:
        study.optimize(objective, n_trials=trials)
    finally:
        if pool["ex"] is not None:
            pool["ex"].shutdown(cancel_futures=True)
        cache.save()
        for c in caches.values():
            c.save()
    render(study, name, png)
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    best = max(done, key=lambda t: t.value) if done else None
    res = {"study": str(study_dir / f"{name}.db"), "trials": len(study.trials), "complete": len(done), "fits": fits_total,
           "weeks": len(weeks), "copies": copies, "png": str(png)}
    if best is not None:
        s = best.user_attrs["stats"]
        res["best"] = {"trial": best.number, "params": best.params, "diff": round(s["mean"], 3), "se": round(s["se"], 3),
                       "ci90": [round(s["lo90"], 3), round(s["hi90"], 3)], "weeks": s["n"]}
        champ_trial = next((t for t in study.trials if t.number == 0), None)
        from stocks_ml.models.trials import record_trials
        record_trials([{"kind": "tune", "name": f"tune_{name}_best", "cv_metric": res["best"]["diff"],
                        "config": {**champ, "params": best.params},
                        "notes": json.dumps({**res["best"], "objective": "paired three-book excess vs the champion, pp/hold, "
                                             "one 4-week phase, common weeks and seeds"}, default=str)}])
        log(f"tune {name}: best trial {best.number} diff {s['mean']:+.2f} pp/hold (90% {s['lo90']:+.2f} to {s['hi90']:+.2f}) "
            f"with {best.params}; champion's own settings (trial 0): "
            + (f"{champ_trial.user_attrs['stats']['mean']:+.2f}" if champ_trial and champ_trial.user_attrs.get('stats') else "n/a")
            + f"; {fits_total} fits -> {png}")
    return res
