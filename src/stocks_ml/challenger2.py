"""challenger2 (owner's design, 2026-09-19): the champion against one altered
recipe, paired and sequential until the two converge.

Each iteration: draw a random rank week of 2006-2015 and a random seed; fit
the champion's recipe at that week with that seed; fit the altered recipe at
the same week with the same seed; score each model's top-3, top-6 and top-10
by their mean 4-week forward return and keep the mean of the three books
(the "three-book average", % per hold). Repeat until the running means
converge, redrawing the experiment's plot after every iteration.

Paired (same week, same seed), so the common market move and the seed luck
cancel and the difference between the two models is what accumulates.

Convergence: at least MIN_ITERS iterations, then stop as soon as the 90%
interval on the paired difference (altered minus champion) excludes zero —
a clear winner or loser — or when the difference's standard error is at or
below SE_TOL percentage points per hold (a near-tie, measured), or at
MAX_ITERS. The plot is the record (plus a summary row in the ledger); no
per-iteration file is kept (owner 2026-09-19: "those will add up"). The plot is the running mean of each model with a
90% band; the JSON beside it holds every iteration. Ranks only: nothing is
decided by this test (the adjudication rules stand).

Every experiment's plot lives in challenger2_convergence/<experiment>.png.
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
SEED_RANGE = (1, 10 ** 6)
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


def draws(ctx, n: int, seed: int = 0, lo=SELECT[0], hi=SELECT[1]) -> list[tuple[pd.Timestamp, int]]:
    """The (week, seed) pairs, in order: uniform over the rank weeks of the
    window with replacement, seeds uniform over SEED_RANGE; one RNG so an
    experiment is reproducible from its seed."""
    weeks = [t for t in ctx.weeks if lo <= t <= hi]
    rng = np.random.default_rng(seed)
    return [(weeks[int(rng.integers(len(weeks)))], int(rng.integers(*SEED_RANGE))) for _ in range(n)]


def book_returns(sel, ctx, t, preds: pd.Series) -> dict | None:
    """The mean 4-week forward return of the top-3/6/10 (% per hold) and
    their mean, from selection.slice_row (the live universe rule)."""
    row = sel.slice_row(ctx, t, "4w", preds)
    if row is None:
        return None
    out = {f"top{b}": 100 * float(row[f"top{b}"]) for b in BOOKS}
    out["avg"] = float(np.mean([out[f"top{b}"] for b in BOOKS]))
    out["universe"] = 100 * float(row["rand_mean"])
    return out


# ---- the worker: both fits at one (week, seed) ----
_W: dict = {}


def _init(fit_ctx, n_jobs: int, alt_ctx=None) -> None:
    import stocks_ml.selection as sel
    _W.update(sel=sel, ctx=fit_ctx, alt_ctx=alt_ctx or fit_ctx, n_jobs=n_jobs)


def _fit_pair(args):
    t, seed, champ, alt = args
    from stocks_ml.train import copy_preds
    out = []
    for r, c in ((champ, _W["ctx"]), (alt, _W["alt_ctx"])):
        p = copy_preds(_W["sel"], c, t, seed, r["label"], r["train_years"], r["features"],
                       r["params"] or None, n_jobs=_W["n_jobs"], drop=r["drop"], train_top=r["train_top"])
        out.append(p)
    return t, seed, out[0], out[1]


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
    """Why the loop may stop now: "decided" when the 90% interval on the
    paired difference excludes zero, "measured" when its standard error is
    at or below se_tol, else None (owner 2026-09-19: a clear result should
    not run to the SE rule)."""
    n = len(diffs)
    if n < min_iters:
        return None
    d = np.asarray(diffs, float)
    se = float(d.std(ddof=1) / np.sqrt(n))
    if se > 0 and abs(d.mean()) > Z * se:
        return "decided"
    if se <= se_tol:
        return "measured"
    return None


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
           se_tol: float = SE_TOL) -> None:
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
    ax.set_xlabel("iteration (one random week of 2006-2015 and one random seed, shared by both models)")
    ax.set_ylabel("running mean of the three-book average\n(mean 4-week return of the top-3/6/10, % per hold)")
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    status = "final" if final else "running"
    if s:
        ax.set_title(f"challenger2 — {name} ({status}, {s['iterations']} iterations)\n"
                     f"paired difference (altered − champion) {s['diff_mean']:+.2f} pp per hold, "
                     f"90% CI {s['diff_ci90'][0]:+.2f} to {s['diff_ci90'][1]:+.2f}, t {s['diff_t']}; "
                     f"stops when the 90% CI excludes zero or the SE ≤ {se_tol} (now {s['diff_se']})", fontsize=10)
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
        workers: int = 4, out_dir: Path = OUT_DIR, spec_path: Path | None = None, log=log) -> dict:
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
    pairs = draws(ctx, max_iters, seed)
    log(f"challenger2 {name}: champion {champ} on {store} vs altered {alt}"
        + (f" on {alt_store} (trained, picked and scored there)" if alt_ctx is not None else "")
        + f"; up to {max_iters} paired iterations "
        f"(min {min_iters}, stop when the paired difference's SE <= {se_tol} pp/hold); seed {seed}; {workers} workers")
    workers, n_jobs = thread_budget(workers)
    feats = tuple(set(champ["features"]) | set(alt["features"]))
    span = max(champ["train_years"], alt["train_years"])
    fit_ctx = fit_context(ctx, feats, SELECT[0], SELECT[1], span)
    alt_fit = fit_context(alt_ctx, feats, SELECT[0], SELECT[1], span) if alt_ctx is not None else None
    score_ctx = alt_ctx if alt_ctx is not None else ctx          # the altered model's picks graded on its own world
    rows, diffs, t0 = [], [], time.time()
    ex = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init, initargs=(fit_ctx, n_jobs, alt_fit)) if workers > 1 else None
    try:
        if ex is None:
            _init(fit_ctx, n_jobs, alt_fit)
        pending, i = {}, 0
        def submit():
            nonlocal i
            if i < len(pairs):
                t, sd = pairs[i]; i += 1
                task = (t, sd, champ, alt)
                fut = ex.submit(_fit_pair, task) if ex else None
                pending[fut if ex else i] = task
        for _ in range(2 * workers if ex else 1):
            submit()
        while pending and not converged(diffs, min_iters, se_tol):
            if ex:
                fut = next(as_completed(list(pending)))
                pending.pop(fut)
                t, sd, pc, pa = fut.result()
            else:
                task = pending.pop(next(iter(pending)))
                t, sd, pc, pa = _fit_pair(task)
            bc = book_returns(sel, ctx, t, pc) if pc is not None else None
            ba = book_returns(sel, score_ctx, t, pa) if pa is not None else None
            if bc is None or ba is None:
                log(f"  {t.date()} seed {sd}: no scorable week; skipped")
            else:
                rows.append({"iteration": len(rows) + 1, "week": str(t.date()), "seed": sd, "champion": bc, "altered": ba})
                diffs.append(ba["avg"] - bc["avg"])
                render(rows, name, candidate, png, se_tol=se_tol)
                if len(rows) % 10 == 0 or len(rows) == 1:
                    s = summary(rows) if len(rows) > 1 else None
                    log(f"  {len(rows)} iterations, {(time.time() - t0) / len(rows):.1f} s each"
                        + (f": champion {s['champion_mean']:+.2f}, altered {s['altered_mean']:+.2f}, diff {s['diff_mean']:+.2f} "
                           f"(SE {s['diff_se']:.2f})" if s else ""))
            submit()
    finally:
        if ex is not None:
            ex.shutdown(cancel_futures=True)
    s = summary(rows)
    why = converged(diffs, min_iters, se_tol)
    s["converged"] = bool(why)
    s["stop_reason"] = why or "max_iters"
    render(rows, name, candidate, png, final=True, se_tol=se_tol)
    res = {"experiment": name, "candidate": candidate, "champion": champ, "altered": {**alt, "store": alt_store or store}, "store": store,
           "window": [str(SELECT[0].date()), str(SELECT[1].date())], "seed": seed,
           "rules": {"min_iters": min_iters, "max_iters": max_iters, "se_tol_pp": se_tol, "books": list(BOOKS),
                     "value": "mean over top-3/6/10 of the book's mean 4-week forward return, % per hold; "
                              "one week and one seed per iteration, shared by both models"},
           "summary": s, "png": str(png), "finished_at": str(pd.Timestamp.now().floor("s"))}
    from stocks_ml.models.trials import record_trials
    record_trials([{"kind": "challenger2", "name": f"challenger2_{name}", "cv_metric": s["diff_mean"],
                    "config": alt, "notes": json.dumps({k: v for k, v in s.items()}, default=str)}])
    log(f"challenger2 {name}: stopped ({s['stop_reason']}: "
        f"{'the 90% CI excludes zero' if why == 'decided' else 'SE at the tolerance' if why == 'measured' else 'the iteration ceiling'}) "
        f"after {s['iterations']} iterations "
        f"({(time.time() - t0) / 60:.1f} min): champion {s['champion_mean']:+.2f}% vs altered {s['altered_mean']:+.2f}% per hold; "
        f"paired difference {s['diff_mean']:+.2f} (90% CI {s['diff_ci90'][0]:+.2f} to {s['diff_ci90'][1]:+.2f}, t {s['diff_t']}); "
        f"altered ahead in {s['share_altered_ahead']:.0%} of weeks -> {png}")
    return res
