"""`stocks-ml train`: the walk — the champion model refit every week, K copies.

Training here is walk-forward. At each rank week t the model (selection.
MODEL_PARAMS, a depth-3 XGBoost with time-tail early stopping) is refit on
the trailing `train_years` of panel rows, labels purged, and scores that
week's members. Copy c differs only by its whole-week bootstrap seed
(models/replication.py), so the live job's sixteen copies at one week
(selection.ensemble_preds) and a walk's sixteen copies at that week are the
same fits. The walk writes

    <out>/preds.parquet    week, ticker, c1..cK
    <out>/spec.json        the record: the recipe (label, training window, extra
                           features, param overrides), K, store, weeks

and resumes by week under a guard on that record. This file is what
`stocks-ml procedure`, `backtest`, `eval` and `app` read; a walk that lacks
a record is refused by all of them.

    stocks-ml train --out data/experiments/<name>/select --start 2006-01-01 --end 2015-12-31
    stocks-ml train --out data/experiments/<name>/extend --start 2016-01-01 --end 2024-07-18
    stocks-ml train ... --every 4 --k 4      # a sample, for exploration only

Rules: nothing at or past the holdout (selection.HOLDOUT_START) is walked;
`--every` > 1 marks the record as a sample, which the procedure refuses.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

STORE = "data/sharadar_world2000_nominal_dl"    # the research world (nominal basis, last-print labels)
CHECKPOINT = 10                                 # weeks between writes of preds.parquet
WORKERS = 7                                     # fits in parallel processes (each gets cpu_count // WORKERS threads): 2.7x on 14 cores
CORES_ENV = "STOCKS_ML_CORES"                   # a cap on the cores a walk may use (workers x threads); unset = all


def thread_budget(workers: int) -> tuple[int, int | None]:
    """(workers, threads per fit) for a walk: all cores shared out between
    the workers, or, under STOCKS_ML_CORES=N, at most N cores in total so
    the machine stays usable (None = XGBoost's default, every core)."""
    cap = int(os.environ.get(CORES_ENV) or 0)
    cores = cap or (os.cpu_count() or 1)
    workers = max(1, min(int(workers), cores))
    if workers == 1:
        return 1, (cap or None)
    return workers, max(1, cores // workers)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def context(store: str = STORE):
    """selection.Ctx on the store, checked: the model's inputs are the
    panel's admitted f_ columns and nothing that names a label."""
    import stocks_ml.selection as sel
    from stocks_ml.features.panel import feature_cols
    ctx = sel.Ctx(store)
    ctx.extra = []
    fc = feature_cols(ctx.pan)
    bad = [c for c in fc if c.startswith(("label", "fwd_ret"))]
    if bad:
        raise RuntimeError(f"label columns among the features: {bad}")
    return sel, ctx, len(fc)


def cap_rank_asof(store: str, pan: pd.DataFrame) -> pd.Series:
    """Each panel row's market-cap rank at the last quarter-end snapshot
    STRICTLY before its date (the store's daily_snapshots: the universe's own
    size measure; a top-N world only). NaN when the name has no cap there."""
    p = Path(store) / "daily_snapshots.parquet"
    if not p.exists():
        raise SystemExit(f"{store} has no daily_snapshots.parquet: train_top needs a top-N world")
    s = pd.read_parquet(p).dropna(subset=["marketcap"])
    s["date"] = pd.to_datetime(s["date"])
    s["cap_rank"] = s.groupby("date")["marketcap"].rank(ascending=False, method="first")
    left = pan[["date", "ticker"]].reset_index().sort_values("date")
    left["date"] = pd.to_datetime(left["date"])
    m = pd.merge_asof(left, s[["date", "ticker", "cap_rank"]].sort_values("date"), on="date", by="ticker",
                      allow_exact_matches=False)
    return m.set_index("index")["cap_rank"].reindex(pan.index)


class FitContext:
    """What a fit needs of selection.Ctx and nothing more: the panel cut to
    the columns the recipe can see (date, ticker, labels, the admitted f_
    columns and the recipe's extras: ~80 of ~220), no prices. The fit is
    bit-identical; a worker process holds a third of the memory and skips
    the context's build."""

    def __init__(self, ctx, features=(), lo=None, hi=None, train_years=None):
        from stocks_ml.features.panel import feature_cols
        admitted = set(feature_cols(ctx.pan)) | set(features)
        keep = [c for c in ctx.pan.columns if not c.startswith(("f_", "x_", "g_")) or c in admitted]
        pan = ctx.pan
        if lo is not None and hi is not None and train_years is not None:
            # only the rows the walk's fits can see: the trailing window before the first
            # rank week (a year of slack for the label's end dates) up to the last one
            d = pd.to_datetime(pan["date"])
            first = pd.Timestamp(lo) - pd.DateOffset(years=int(train_years) + 1)
            pan = pan[(d >= first) & (d <= pd.Timestamp(hi))]
        self.pan = pan[keep].copy()
        self.extra = list(ctx.extra)
        self.cfg = ctx.cfg

    def world_cfg(self, train_years):
        import stocks_ml.selection as sel
        return sel.Ctx.world_cfg(self, train_years)


def fit_context(ctx, features=(), lo=None, hi=None, train_years=None):
    """A real context is slimmed for the fits; anything else (a test's
    stand-in) passes through."""
    import stocks_ml.selection as sel
    return FitContext(ctx, features, lo, hi, train_years) if isinstance(ctx, sel.Ctx) else ctx


def copy_preds(sel, ctx, t, c: int, label: str, train_years: int, features=(),
               params: dict | None = None, n_jobs: int | None = None, drop=(),
               train_top: int | None = None, block=None):
    """Copy c's scores at rank week t: the champion model (MODEL_PARAMS with
    `params` overriding) on the trailing `train_years`, bootstrap seed c,
    trained on `label`, on the panel's f_ columns plus `features`. `n_jobs`
    caps the fit's threads (a worker's share of the cores); the fit is the
    same either way."""
    from stocks_ml.models.replication import WeekBootstrapEstimator
    from stocks_ml.models.walk import walk_forward_predictions
    from stocks_ml.models.xgb import estimator_for
    purge = sel.label_purge(label, "4w")
    fixed = sel.fixed(purge)
    if n_jobs is not None:
        fixed = {**fixed, "n_jobs": int(n_jobs)}
    est = WeekBootstrapEstimator(estimator_for({**sel.MODEL_PARAMS, **(params or {})}, fixed), bootstrap_seed=c)
    if block:                       # one fit at block[0] scores every week of the block
        wf = walk_forward_predictions(ctx.pan, est, ctx.world_cfg(train_years), start=block[0], end=block[-1],
                                      label_col=label, purge_days=purge,
                                      extra_features=tuple(features or ctx.extra),
                                      drop_features=tuple(drop), train_top=train_top, refit_every=len(block))
        return {w: wf.preds.get(w) for w in block}
    wf = walk_forward_predictions(ctx.pan, est, ctx.world_cfg(train_years), start=t, end=t,
                                  label_col=label, purge_days=purge,
                                  extra_features=tuple(features or ctx.extra),
                                  drop_features=tuple(drop), train_top=train_top)
    return wf.preds.get(t)


def recipe(label: str, train_years: int, features=(), params: dict | None = None, drop=(),
           train_top: int | None = None) -> dict:
    """A candidate model as `train` walks it: the label, the training window,
    extra panel columns the model gets beyond the panel's f_ columns, and
    overrides of selection.MODEL_PARAMS. Empty features/params are left out,
    so the champion's recipe reads {label, train_years} as its records do."""
    import stocks_ml.selection as sel
    r = {"label": label, "train_years": int(train_years)}
    if features:
        r["features"] = list(features)
    if drop:
        r["drop"] = list(drop)                  # admitted f_ columns withheld from the model
    if train_top:
        r["train_top"] = int(train_top)         # fit on the largest N names only; score every member
    if params:
        from stocks_ml.models.xgb import RANK_PARAMS
        allowed = {**sel.MODEL_PARAMS, **RANK_PARAMS}
        bad = [k for k in params if k not in allowed]
        if bad:
            raise ValueError(f"params {bad} are not in selection.MODEL_PARAMS {tuple(sel.MODEL_PARAMS)} "
                             f"or the rank params {tuple(RANK_PARAMS)}")
        r["params"] = {k: type(allowed[k])(v) for k, v in params.items()}
    return r


def record(store: str, lo, hi, label: str, train_years: int, k: int, every: int,
           delist: str, price_basis: str, features=(), params: dict | None = None,
           sample: str | None = None, copies=None, drop=(), train_top: int | None = None,
           refit_every: int = 1) -> dict:
    """The walk's record (spec.json): what made it, on what, over which weeks.
    `sample` describes an explicit week list (challenge-fast's stratified
    random sample); such a record is marked a sample like `every` > 1."""
    import stocks_ml.selection as sel
    span = f"{pd.Timestamp(lo).date()} -> {pd.Timestamp(hi).date()}"
    rec = {"code": "stocks_ml.train.walk",
           "recipe": recipe(label, train_years, features, params, drop, train_top),
           "k": int(k), "store": store, "delist": delist, "price_basis": price_basis,
           "base_params": {p: str(v) for p, v in sel.MODEL_PARAMS.items()},
           "every": int(every),
           "weeks": (f"{sample} of {span} (a sample)" if sample else
                     f"every week of {span}" if every == 1
                     else f"every {every}th week of {span} (a sample)")}
    if sample:
        rec["sample"] = sample
    if int(refit_every) > 1:
        rec["refit_every"] = int(refit_every)      # the screening cadence: a fit serves N weeks
        rec["weeks"] += f", refit every {int(refit_every)} weeks (screening cadence)"
    copies = list(copies) if copies is not None else list(range(1, int(k) + 1))
    if copies != list(range(1, int(k) + 1)):
        rec["copies"] = [int(copies[0]), int(copies[-1])]      # e.g. a seed twin: 17..32
    return rec


def guard(out: Path, rec: dict) -> None:
    """A walk resumes only under the record it was started with."""
    path = out / "spec.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != rec:
            raise RuntimeError(f"{out} was walked under {old}; this run is {rec}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1, sort_keys=True))


_W: dict = {}


def _init_worker(fit_ctx, n_jobs: int) -> None:
    import stocks_ml.selection as sel
    _W.update(sel=sel, ctx=fit_ctx, n_jobs=n_jobs)


def _fit_task(args):
    t, c, label, train_years, features, params, drop, train_top, block = args
    p = copy_preds(_W["sel"], _W["ctx"], t, c, label, train_years, features, params,
                   n_jobs=_W["n_jobs"], drop=drop, train_top=train_top, block=block)
    return t, c, p


def walk(store: str, lo, hi, label: str, train_years: int, k: int, out: Path,
         every: int = 1, checkpoint: int = CHECKPOINT, log=log, features=(),
         params: dict | None = None, weeks=None, sample: str | None = None,
         workers: int = 1, copies=None, drop=(), train_top: int | None = None,
         refit_every: int = 1) -> Path:
    """Copies 1..k (or the explicit `copies`, e.g. seeds 17..32 for a seed
    twin) at every rank week of [lo, hi] (every `every`th week for a sample;
    or the explicit `weeks`, described by `sample`), checkpointed and
    resumed by week under the record's guard. `workers` > 1 fits the copies
    in that many processes, each with cpu_count // workers threads — the
    same fits, assembled in the same order; only the wall time changes."""
    from stocks_ml.selection import HOLDOUT_START, LABELS_4W
    if label not in LABELS_4W:
        raise SystemExit(f"label must be one of {tuple(LABELS_4W)}, got {label!r}")
    lo, hi, out = pd.Timestamp(lo), pd.Timestamp(hi), Path(out)
    if hi >= HOLDOUT_START:
        raise SystemExit(f"--end {hi.date()} reaches the holdout ({HOLDOUT_START.date()}); "
                         "the holdout is never walked without the owner's go")
    sel, ctx, n_feat = context(store)
    missing = [c for c in features if c not in ctx.pan.columns]
    if missing:
        raise SystemExit(f"the panel lacks the recipe's feature columns {missing[:5]}")
    if weeks is None:
        weeks = [t for t in ctx.weeks if lo <= t <= hi][::every]
    else:
        weeks = sorted(pd.Timestamp(w) for w in weeks)
        if not sample:
            raise ValueError("an explicit week list must be described by `sample`")
    copies = list(copies) if copies is not None else list(range(1, k + 1))
    if len(copies) != k:
        raise ValueError(f"k={k} but {len(copies)} copies given")
    refit_every = max(1, int(refit_every))
    if refit_every > 1 and every != 1:
        raise ValueError("refit_every applies to a walk of every week or a block sample, not an every-Nth sample")
    rec = record(store, lo, hi, label, train_years, k, every, ctx.delist_labels,
                 getattr(ctx.cfg, "price_basis", "closeadj"), features, params, sample, copies, drop, train_top,
                 refit_every)
    guard(out, rec)
    path = out / "preds.parquet"
    frames, done = [], set()
    if path.exists():
        old = pd.read_parquet(path)
        old["week"] = pd.to_datetime(old["week"])
        frames.append(old)
        done = set(old["week"].unique())
    # the units of work: single weeks, or blocks of `refit_every` consecutive weeks
    # sharing one fit (a block with any week missing is redone whole)
    if refit_every > 1 and sample:
        # a block sample: runs of `refit_every` consecutive rank weeks (challenge.stratified_blocks)
        pos = {t: i for i, t in enumerate(ctx.weeks)}
        blocks, run = [], []
        for t in weeks:
            if run and (pos[t] != pos[run[-1]] + 1 or len(run) == refit_every):
                blocks.append(tuple(run)); run = []
            run.append(t)
        if run:
            blocks.append(tuple(run))
        if any(len(b) != refit_every for b in blocks):
            raise ValueError(f"a block sample at refit_every={refit_every} must be runs of {refit_every} consecutive rank weeks")
    else:
        blocks = [tuple(weeks[i:i + refit_every]) for i in range(0, len(weeks), refit_every)]
    todo = [t for t in weeks if t not in done]
    todo_blocks = [b for b in blocks if any(t not in done for t in b)]
    log(f"train: {rec['weeks']}, {len(todo)} of {len(weeks)} weeks to do, K={k}, "
        f"{label} / {train_years}y on {n_feat} features"
        + (f" + {len(features)} extra" if features else "")
        + (f", params {rec['recipe']['params']}" if params else "")
        + (f", minus {len(drop)} dropped" if drop else "")
        + (f", trained on the largest {train_top}" if train_top else "")
        + (f", one fit per {refit_every} weeks ({len(todo_blocks)} blocks)" if refit_every > 1 else "")
        + f", store {store}")
    t0 = time.time()
    ex = None
    workers, n_jobs = thread_budget(workers)
    if todo and train_top:
        from stocks_ml.models.walk import CAP_RANK_COL
        ctx.pan[CAP_RANK_COL] = cap_rank_asof(store, ctx.pan)
    if todo:                       # the slim panel the fits see; the full context is released
        ctx = fit_context(ctx, features, weeks[0], weeks[-1], train_years)
    if workers > 1 and todo:
        ex = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(ctx, n_jobs))
        log(f"  {workers} worker processes x {n_jobs} threads")
    try:
        done_n = 0
        units = todo_blocks if refit_every > 1 else [(t,) for t in todo]
        step = max(1, checkpoint // refit_every)
        for b in range(0, len(units), step):
            batch_units = units[b:b + step]
            batch = [t for u in batch_units for t in u]
            tasks = [(u[0], c, label, train_years, tuple(features), params, tuple(drop), train_top,
                      u if refit_every > 1 else None) for u in batch_units for c in copies]
            if ex is not None:
                raw = list(ex.map(_fit_task, tasks))              # in task order
            else:
                raw = [(t, c, copy_preds(sel, ctx, t, c, label, train_years, features, params,
                                         n_jobs=n_jobs, drop=drop, train_top=train_top, block=blk))
                       for t, c, *_, blk in tasks]
            results = []                                           # (week, copy, preds) per week
            for tf, c, p in raw:                                  # tf: the unit's first (fit) week
                if isinstance(p, dict):
                    results += [(w, c, s) for w, s in p.items()]
                else:
                    results.append((tf, c, p))
            for t in batch:
                cols = {f"c{c}": p for tt, c, p in results if tt == t and p is not None}
                df = pd.DataFrame(cols)
                df.index.name = "ticker"
                df = df.reset_index()
                df.insert(0, "week", t)
                frames.append(df)
            done_n += len(batch)
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
            el = time.time() - t0
            log(f"  {done_n}/{len(todo)} weeks, {el / done_n:.0f} s/week, "
                f"~{el / done_n * (len(todo) - done_n) / 60:.0f} min left")
    finally:
        if ex is not None:
            ex.shutdown()
    log(f"train: done in {(time.time() - t0) / 60:.1f} min -> {path}")
    return path


def check_reproduces(store: str, preds_path: Path, weeks: int = 2, copies=(1,), log=log) -> dict:
    """Refit the first `weeks` rank weeks of a saved walk for the given copies
    and compare with what it stored: the record is only worth its name if
    the code still reproduces the file. Cheap; used by `stocks-ml train
    --check`."""
    preds_path = Path(preds_path)
    rec = json.loads((preds_path.parent / "spec.json").read_text())
    sel, ctx, _ = context(store)
    if rec["recipe"].get("train_top"):
        from stocks_ml.models.walk import CAP_RANK_COL
        ctx.pan[CAP_RANK_COL] = cap_rank_asof(store, ctx.pan)
    saved = pd.read_parquet(preds_path)
    saved["week"] = pd.to_datetime(saved["week"])
    out = {}
    for t in sorted(saved["week"].unique())[:weeks]:
        g = saved[saved.week == t].set_index("ticker")
        for c in copies:
            r = rec["recipe"]
            p = copy_preds(sel, ctx, t, c, r["label"], r["train_years"], r.get("features", ()),
                           r.get("params"), drop=r.get("drop", ()), train_top=r.get("train_top"))
            a, b = p.align(g[f"c{c}"], join="inner")
            same = len(a) == len(g) and np.allclose(a.values, b.values, atol=1e-6)
            out[f"{pd.Timestamp(t).date()} c{c}"] = bool(same)
            log(f"  {pd.Timestamp(t).date()} copy {c}: {'reproduces' if same else 'DIFFERS'} "
                f"({len(a)} names)")
    return out
