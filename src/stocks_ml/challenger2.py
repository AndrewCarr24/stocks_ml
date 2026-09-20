"""challenger2 (owner's design, 2026-09-19): the champion against one altered
recipe, paired and sequential until the two converge.

Each iteration: draw a random rank week of 2006-2015 and a random seed; fit
the champion's recipe at that week with that seed; fit the altered recipe at
the same week with the same seed (`copies` > 1: that many consecutive seeds
per model, predictions averaged — the deployed model is a 16-copy mean, and
a single fit's early-stopping round is fragile); score each model's top-3,
top-6 and top-10 by their mean 4-week forward return MINUS the week's
average member (the edge over an average stock; the market's own move,
which the two models share, drops out — owner 2026-09-19) and keep the mean
of the three books (the "three-book average", pp per hold). Repeat until
the running means converge, redrawing the experiment's plot after every
iteration. The paired difference is the same with or without the
subtraction; the lines settle about twice as fast.

Paired (same week, same seed), so the common market move and the seed luck
cancel and the difference between the two models is what accumulates.

Convergence: at least MIN_ITERS iterations, then stop when the standard
error of the paired difference (altered minus champion) is at or below
SE_TOL percentage points per hold, or at MAX_ITERS; the 90% interval is
read once, there. No early stop on the interval: looking after every
iteration, a 90% boundary fires on a true tie 44% of the time (the sp500up
sanity check of 2026-09-19 caught it; owner: keep the SE rule only). The plot is the record (plus a summary row in the ledger); no
per-iteration file is kept (owner 2026-09-19: "those will add up"). The plot is the running mean of each model with a
90% band; the JSON beside it holds every iteration. Ranks only: nothing is
decided by this test (the adjudication rules stand).

Every experiment's plot lives in challenger2_convergence/<experiment>.png.

Seeds and the cache (owner 2026-09-19): weeks are drawn WITHOUT replacement
and each week's model seeds are a week-determined subset of 1..16 (all of
them at copies=16), the same in every experiment. So a recipe's copies at a
week are reusable: the champion's come from its K=16 walk when the store
is the walk's, and every fit is cached under data/challenger2_cache/<store>/
<recipe hash>.parquet (week, ticker, c1..c16) for the next experiment.
"""
from __future__ import annotations

import json
import multiprocessing
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("challenger2_convergence")
SELECT = (pd.Timestamp("2006-01-01"), pd.Timestamp("2015-12-31"))
BOOKS = (3, 6, 10)
MIN_ITERS = 50
MAX_ITERS = 500
SE_TOL = 0.35            # pp per hold (owner 2026-09-19: 0.25 took too many iterations): the standard error of the paired difference at which the test stops
SEED_POOL = 64           # model seeds per week come from 1..64 (the champion's walks: select + twin + seeds 33-64), week-determined
CACHE_DIR = Path("data/challenger2_cache")
Z = 1.645                # the 90% interval (owner 2026-09-19)
YLIM = (-20, 20)         # the plot's y-axis, fixed so the lines are readable and the frame does not jump
COLORS = {"champion": "#2a78d6", "altered": "#eb6834"}   # the reference categorical slots 1 and 2 (validated)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def champion_recipe(spec_path: Path | None = None) -> dict:
    """The champion's recipe as the spec records it (label, window, features,
    drop, params, train_top)."""
    from stocks_ml.procedure import SPEC_PATH
    spec = json.loads(Path(spec_path or SPEC_PATH).read_text())
    return {"label": spec["horizon"]["label"], "train_years": int(spec["training_window_years"]),
            "features": list(spec.get("features") or []), "drop": list(spec.get("drop_features") or []),
            "params": {}, "train_top": spec.get("train_top")}


def altered_recipe(text: str, base: dict) -> dict:
    """The altered recipe: the champion's with the named fields changed
    (challenge.parse_candidate syntax: params=max_depth:5, label=..., ...).
    `store=` names another world (universe) the altered model trains on
    and picks from; it is scored there, the champion on its own."""
    from stocks_ml.challenge import parse_candidate
    r = parse_candidate(text, base)
    out = {"label": r["label"], "train_years": r["train_years"], "features": list(r.get("features") or []),
           "drop": list(r.get("drop") or []), "params": dict(r.get("params") or {}), "train_top": r.get("train_top")}
    if r.get("store"):
        out["store"] = r["store"]
    return out


def experiment_name(text: str) -> str:
    """A file name from the candidate text: 'params=max_depth:5' -> 'max_depth_5'."""
    s = re.sub(r"^params=", "", text.strip())
    s = re.sub(r"store=data/sharadar_", "store_", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s or "experiment"


def week_seeds(t, copies: int) -> list[int]:
    """The model seeds for week t: a week-determined subset of 1..SEED_POOL
    (all of them at copies = SEED_POOL), the same in every experiment, so a
    recipe's fits at a week are reusable across experiments."""
    rng = np.random.default_rng(int(pd.Timestamp(t).value // 10 ** 9) % (2 ** 32))
    return sorted(int(s) for s in rng.permutation(np.arange(1, SEED_POOL + 1))[:min(copies, SEED_POOL)])


def draws(ctx, n: int, seed: int = 0, lo=SELECT[0], hi=SELECT[1]) -> list[pd.Timestamp]:
    """The weeks, in order: a seeded random order of the window's rank weeks,
    without replacement (a repeated week would repeat its seeds too), at
    most n of them."""
    weeks = [t for t in ctx.weeks if lo <= t <= hi]
    rng = np.random.default_rng(seed)
    return [weeks[i] for i in rng.permutation(len(weeks))[:min(n, len(weeks))]]


def recipe_hash(recipe: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]


class PredCache:
    """A recipe's single-copy predictions per (week, seed) on one store:
    week, ticker, c1..c16 in one parquet; seeded from the champion's K=16
    walk when the store and recipe are the walk's."""

    def __init__(self, store: str, recipe: dict, cache_dir: Path | None = None):
        self.path = Path(cache_dir or CACHE_DIR) / Path(store).name / f"{recipe_hash(recipe)}.parquet"
        self.recipe = recipe
        self.frames: dict = {}                        # week -> DataFrame(index ticker, columns c<seed>)
        self.dirty = False
        if self.path.exists():
            self._absorb(pd.read_parquet(self.path))
        meta = self.path.with_suffix(".json")
        if not meta.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text(json.dumps({"store": store, "recipe": recipe}, indent=1))

    def _absorb(self, df: pd.DataFrame) -> None:
        df = df.copy(); df["week"] = pd.to_datetime(df["week"])
        for t, g in df.groupby("week"):
            cols = [c for c in g.columns if c.startswith("c") and c[1:].isdigit()]
            new = g.set_index("ticker")[cols]
            self.frames[t] = self.frames[t].combine_first(new) if t in self.frames else new     # what is cached stays

    def seed_from_walk(self, preds_path) -> int:
        """Absorb a walk's copies (copy c = seed c: c1..c16, c17..c32, c33..c64)."""
        df = pd.read_parquet(preds_path)
        self._absorb(df); self.dirty = True
        return int(df["week"].nunique())

    def get(self, t, seeds) -> dict:
        g = self.frames.get(pd.Timestamp(t))
        if g is None:
            return {}
        return {s: g[f"c{s}"].dropna() for s in seeds if f"c{s}" in g.columns and g[f"c{s}"].notna().any()}

    def put(self, t, seed: int, p: pd.Series) -> None:
        t = pd.Timestamp(t)
        g = self.frames.get(t, pd.DataFrame())
        g = g.join(p.rename(f"c{seed}"), how="outer") if len(g) else p.rename(f"c{seed}").to_frame()
        self.frames[t] = g; self.dirty = True

    def save(self) -> None:
        if not self.dirty:
            return
        rows = []
        for t, g in self.frames.items():
            d = g.reset_index().rename(columns={"index": "ticker"}); d.insert(0, "week", t); rows.append(d)
        df = pd.concat(rows, ignore_index=True)
        cols = ["week", "ticker"] + sorted([c for c in df.columns if c.startswith("c")], key=lambda c: int(c[1:]))
        df[cols].to_parquet(self.path, index=False); self.dirty = False


def champion_walks_for(store: str, recipe: dict, spec_path: Path | None = None) -> list[Path]:
    """The champion's weekly walks on `store` with `recipe`, every seed set
    beside the spec's select walk (select: copies 1-16; twin: 17-32; any
    seeds_<a>_<b> directory), each checked against its record."""
    from stocks_ml.procedure import SPEC_PATH
    spec = json.loads(Path(spec_path or SPEC_PATH).read_text())
    p = Path(spec["procedure"]["preds"]["path"])
    out = []
    for d in sorted(p.parent.parent.glob("*")):
        preds, rec_path = d / "preds.parquet", d / "spec.json"
        if not (d.is_dir() and preds.exists() and rec_path.exists()):
            continue
        rec = json.loads(rec_path.read_text())
        r = rec.get("recipe", {})
        same = (rec.get("store") == store and r.get("label") == recipe["label"] and int(r.get("train_years", 0)) == recipe["train_years"]
                and list(r.get("features") or []) == recipe["features"] and list(r.get("drop") or []) == recipe["drop"]
                and dict(r.get("params") or {}) == recipe["params"] and r.get("train_top") == recipe["train_top"]
                and "refit_every" not in rec and str(rec.get("weeks", "")).startswith("every week of 2006-01-01"))
        if same:
            out.append(preds)
    return out


def book_returns(sel, ctx, t, preds: pd.Series) -> dict | None:
    """The mean 4-week forward return of the top-3/6/10 minus the week's
    average member (pp per hold), their mean, and the average member's own
    return, from selection.slice_row (the live universe rule)."""
    row = sel.slice_row(ctx, t, "4w", preds)
    if row is None:
        return None
    uni = 100 * float(row["rand_mean"])
    out = {f"top{b}": 100 * float(row[f"top{b}"]) - uni for b in BOOKS}
    out["avg"] = float(np.mean([out[f"top{b}"] for b in BOOKS]))
    out["universe"] = uni
    return out


# ---- the worker: both fits at one (week, seed) ----
_W: dict = {}


def _init(fit_ctx, n_jobs: int, alt_ctx=None) -> None:
    import stocks_ml.selection as sel
    _W.update(sel=sel, ctx=fit_ctx, alt_ctx=alt_ctx or fit_ctx, n_jobs=n_jobs)


def _fit_pair(args):
    """Fit the seeds each side still needs at week t: (t, {seed: preds} for
    the champion, {seed: preds} for the altered)."""
    t, need_c, need_a, champ, alt = args
    from stocks_ml.train import copy_preds
    out = []
    for r, c, need in ((champ, _W["ctx"], need_c), (alt, _W["alt_ctx"], need_a)):
        got = {}
        for s in need:
            p = copy_preds(_W["sel"], c, t, s, r["label"], r["train_years"], r["features"],
                           r["params"] or None, n_jobs=_W["n_jobs"], drop=r["drop"], train_top=r["train_top"])
            if p is not None:
                got[s] = p
        out.append(got)
    return t, out[0], out[1]


def ensemble(parts: dict) -> pd.Series | None:
    """The mean over seeds of single-copy predictions, as the ensemble ranks."""
    ps = [p for p in parts.values() if p is not None]
    return pd.concat(ps, axis=1).mean(axis=1) if ps else None


def running(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """The running mean and its standard error after each iteration."""
    v = np.asarray(values, float)
    n = np.arange(1, len(v) + 1)
    mean = np.cumsum(v) / n
    sq = np.cumsum(v ** 2)
    var = np.where(n > 1, (sq - n * mean ** 2) / np.maximum(n - 1, 1), np.nan)
    se = np.sqrt(np.maximum(var, 0) / n)
    return mean, se


def converged(diffs: list[float], min_iters: int = MIN_ITERS, se_tol: float = SE_TOL) -> str | None:
    """Why the loop may stop now: "measured" when the paired difference's
    standard error is at or below se_tol (the one look at the interval),
    else None."""
    n = len(diffs)
    if n < min_iters:
        return None
    se = float(np.std(np.asarray(diffs, float), ddof=1) / np.sqrt(n))
    return "measured" if se <= se_tol else None


def summary(rows: list[dict]) -> dict:
    c = np.array([r["champion"]["avg"] for r in rows]); a = np.array([r["altered"]["avg"] for r in rows])
    d = a - c
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    per_book = {}
    for b in BOOKS:
        db = np.array([r["altered"][f"top{b}"] - r["champion"][f"top{b}"] for r in rows])
        per_book[f"top{b}"] = {"champion": round(float(np.mean([r["champion"][f"top{b}"] for r in rows])), 3),
                               "altered": round(float(np.mean([r["altered"][f"top{b}"] for r in rows])), 3),
                               "diff": round(float(db.mean()), 3),
                               "se": round(float(db.std(ddof=1) / np.sqrt(n)), 3) if n > 1 else None}
    return {"iterations": n, "champion_mean": round(float(c.mean()), 3), "altered_mean": round(float(a.mean()), 3),
            "diff_mean": round(float(d.mean()), 3), "diff_se": round(se, 3),
            "diff_ci90": [round(float(d.mean() - Z * se), 3), round(float(d.mean() + Z * se), 3)],
            "diff_t": round(float(d.mean() / se), 2) if se and se == se and se > 0 else None,
            "share_altered_ahead": round(float((d > 0).mean()), 3), "per_book": per_book}


def render(rows: list[dict], name: str, alt_text: str, png: Path, final: bool = False,
           se_tol: float = SE_TOL, copies: int = 1) -> None:
    """The experiment's plot: the running mean of each model's three-book
    average (% per hold) with a 90% band, iteration on the x-axis; the
    end labels are pushed apart so they never overlap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = [r["champion"]["avg"] for r in rows]; a = [r["altered"]["avg"] for r in rows]
    mc, sc = running(c); ma, sa = running(a)
    x = np.arange(1, len(rows) + 1)
    s = summary(rows) if len(rows) > 1 else None
    fig, ax = plt.subplots(figsize=(10, 5.2))
    series = (("champion", mc, sc, COLORS["champion"]), (f"altered: {alt_text}", ma, sa, COLORS["altered"]))
    upper = max(series, key=lambda s: s[1][-1])[0]          # the higher line's label goes up, the other down
    for label, m, se, color in series:
        ax.plot(x, m, color=color, lw=2, label=label)
        ok = ~np.isnan(se)
        ax.fill_between(x[ok], (m - Z * se)[ok], (m + Z * se)[ok], color=color, alpha=0.12, lw=0)
        dy = 10 if label == upper else -10
        ax.annotate(f"{m[-1]:+.2f}%", (x[-1], m[-1]), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=9, color="#52514e")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("iteration (one random week of 2006-2015 and one random seed, shared by both models"
                  + (f"; {copies} copies averaged)" if copies > 1 else ")"))
    ax.set_ylabel("running mean of the three-book average\n(mean 4-week return of the top-3/6/10 minus the average member, pp per hold)")
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    status = "final" if final else "running"
    if s:
        ax.set_title(f"challenger2 — {name} ({status}, {s['iterations']} iterations)\n"
                     f"paired difference (altered − champion) {s['diff_mean']:+.2f} pp per hold, "
                     f"90% CI {s['diff_ci90'][0]:+.2f} to {s['diff_ci90'][1]:+.2f}, t {s['diff_t']}; "
                     f"stops when the SE ≤ {se_tol} (now {s['diff_se']})", fontsize=10)
    else:
        ax.set_title(f"challenger2 — {name} ({status}, {len(rows)} iteration)", fontsize=10)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    tmp = png.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    tmp.replace(png)                      # never a half-written png


def run(candidate: str, name: str | None = None, store: str | None = None, seed: int = 0,
        min_iters: int = MIN_ITERS, max_iters: int = MAX_ITERS, se_tol: float = SE_TOL,
        workers: int = 4, out_dir: Path = OUT_DIR, spec_path: Path | None = None, log=log,
        copies: int = 1) -> dict:
    from stocks_ml.train import STORE, context, fit_context, thread_budget
    store = store or STORE
    champ = champion_recipe(spec_path)
    alt = altered_recipe(candidate, champ)
    if alt == champ:
        raise SystemExit("the altered recipe is the champion's own")
    name = name or experiment_name(candidate)
    out_dir = Path(out_dir)
    png = out_dir / f"{name}.png"
    sel, ctx, _ = context(store)
    alt_store = alt.pop("store", None)
    alt_ctx = None
    if alt_store and alt_store != store:
        _, alt_ctx, _ = context(alt_store)        # the altered model trains, picks and is scored here
        if [t for t in alt_ctx.weeks if SELECT[0] <= t <= SELECT[1]] != [t for t in ctx.weeks if SELECT[0] <= t <= SELECT[1]]:
            raise SystemExit(f"{alt_store} and {store} do not share the window's rank weeks")
    for c in set(champ["features"]) | set(alt["features"]):
        if c not in ctx.pan.columns or c not in (alt_ctx or ctx).pan.columns:
            raise SystemExit(f"a panel lacks the feature column {c}")
    weeks = draws(ctx, max_iters, seed)
    caches = {"champion": PredCache(store, champ), "altered": PredCache(alt_store or store, alt)}
    if not caches["champion"].frames:
        for walk in champion_walks_for(store, champ, spec_path):
            n_w = caches["champion"].seed_from_walk(walk)
            log(f"champion cache seeded from {walk} ({n_w} weeks)")
    log(f"challenger2 {name}: champion {champ} on {store} vs altered {alt}"
        + (f" on {alt_store} (trained, picked and scored there)" if alt_ctx is not None else "")
        + f"; {copies} seeded copies per model per iteration (seeds from 1..{SEED_POOL}, week-determined)"
        + f"; up to {len(weeks)} paired iterations (weeks without replacement; min {min_iters}, stop when the paired "
        f"difference's SE <= {se_tol} pp/hold); draw seed {seed}; {workers} workers; caches hold "
        f"{len(caches['champion'].frames)} / {len(caches['altered'].frames)} weeks")
    workers, n_jobs = thread_budget(workers)
    feats = tuple(set(champ["features"]) | set(alt["features"]))
    span = max(champ["train_years"], alt["train_years"])
    fit_ctx = fit_context(ctx, feats, SELECT[0], SELECT[1], span)
    alt_fit = fit_context(alt_ctx, feats, SELECT[0], SELECT[1], span) if alt_ctx is not None else None
    score_ctx = alt_ctx if alt_ctx is not None else ctx          # the altered model's picks graded on its own world
    rows, diffs, t0, fitted = [], [], time.time(), 0
    ex = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init, initargs=(fit_ctx, n_jobs, alt_fit)) if workers > 1 else None
    try:
        if ex is None:
            _init(fit_ctx, n_jobs, alt_fit)
        pending, i = {}, 0
        def submit():
            nonlocal i
            if i < len(weeks):
                t = weeks[i]; i += 1
                seeds = week_seeds(t, copies)
                have_c, have_a = caches["champion"].get(t, seeds), caches["altered"].get(t, seeds)
                need_c, need_a = [s for s in seeds if s not in have_c], [s for s in seeds if s not in have_a]
                task = (t, need_c, need_a, champ, alt)
                key = ex.submit(_fit_pair, task) if ex else (t, len(pending))
                pending[key] = (task, have_c, have_a)
        for _ in range(2 * workers if ex else 1):
            submit()
        while pending and not converged(diffs, min_iters, se_tol):
            if ex:
                fut = next(as_completed(list(pending)))
                task, have_c, have_a = pending.pop(fut)
                t, new_c, new_a = fut.result()
            else:
                key = next(iter(pending)); task, have_c, have_a = pending.pop(key)
                t, new_c, new_a = _fit_pair(task)
            fitted += len(new_c) + len(new_a)
            for s_, p in new_c.items():
                caches["champion"].put(t, s_, p)
            for s_, p in new_a.items():
                caches["altered"].put(t, s_, p)
            pc, pa = ensemble({**have_c, **new_c}), ensemble({**have_a, **new_a})
            bc = book_returns(sel, ctx, t, pc) if pc is not None else None
            ba = book_returns(sel, score_ctx, t, pa) if pa is not None else None
            if bc is None or ba is None:
                log(f"  {t.date()}: no scorable week; skipped")
            else:
                rows.append({"iteration": len(rows) + 1, "week": str(t.date()), "seeds": week_seeds(t, copies), "champion": bc, "altered": ba})
                diffs.append(ba["avg"] - bc["avg"])
                render(rows, name, candidate, png, se_tol=se_tol, copies=copies)
                if len(rows) % 10 == 0 or len(rows) == 1:
                    s = summary(rows) if len(rows) > 1 else None
                    log(f"  {len(rows)} iterations, {(time.time() - t0) / len(rows):.1f} s each, {fitted} fits so far"
                        + (f": champion {s['champion_mean']:+.2f}, altered {s['altered_mean']:+.2f}, diff {s['diff_mean']:+.2f} "
                           f"(SE {s['diff_se']:.2f})" if s else ""))
                if len(rows) % 20 == 0:
                    for c in caches.values():
                        c.save()
            submit()
    finally:
        if ex is not None:
            ex.shutdown(cancel_futures=True)
        for c in caches.values():
            c.save()
    s = summary(rows)
    why = converged(diffs, min_iters, se_tol)
    s["converged"] = bool(why)
    s["stop_reason"] = why or "max_iters"
    render(rows, name, candidate, png, final=True, se_tol=se_tol, copies=copies)
    res = {"experiment": name, "candidate": candidate, "champion": champ, "altered": {**alt, "store": alt_store or store}, "store": store,
           "window": [str(SELECT[0].date()), str(SELECT[1].date())], "seed": seed, "fits": fitted,
           "rules": {"min_iters": min_iters, "max_iters": max_iters, "se_tol_pp": se_tol, "books": list(BOOKS), "copies": copies,
                     "value": "mean over top-3/6/10 of the book's mean 4-week forward return minus the week's average "
                              "member, pp per hold; one week and one seed per iteration, shared by both models"},
           "summary": s, "png": str(png), "finished_at": str(pd.Timestamp.now().floor("s"))}
    from stocks_ml.models.trials import record_trials
    record_trials([{"kind": "challenger2", "name": f"challenger2_{name}", "cv_metric": s["diff_mean"],
                    "config": alt, "notes": json.dumps({**s, "copies": copies, "seed": seed}, default=str)}])
    log(f"challenger2 {name}: stopped ({s['stop_reason']}: "
        f"{'SE at the tolerance' if why == 'measured' else 'the iteration ceiling'}) "
        f"after {s['iterations']} iterations "
        f"({(time.time() - t0) / 60:.1f} min): champion {s['champion_mean']:+.2f}% vs altered {s['altered_mean']:+.2f}% per hold; "
        f"paired difference {s['diff_mean']:+.2f} (90% CI {s['diff_ci90'][0]:+.2f} to {s['diff_ci90'][1]:+.2f}, t {s['diff_t']}); "
        f"altered ahead in {s['share_altered_ahead']:.0%} of weeks; {fitted} new fits, the rest from the caches -> {png}")
    return res
