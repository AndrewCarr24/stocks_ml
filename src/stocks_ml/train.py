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


def copy_preds(sel, ctx, t, c: int, label: str, train_years: int, features=(),
               params: dict | None = None, n_jobs: int | None = None) -> pd.Series | None:
    """Copy c's scores at rank week t: the champion model (MODEL_PARAMS with
    `params` overriding) on the trailing `train_years`, bootstrap seed c,
    trained on `label`, on the panel's f_ columns plus `features`. `n_jobs`
    caps the fit's threads (a worker's share of the cores); the fit is the
    same either way."""
    from stocks_ml.models.replication import WeekBootstrapEstimator
    from stocks_ml.models.walk import walk_forward_predictions
    from stocks_ml.models.xgb import TimeTailEarlyStopXGB
    purge = sel.label_purge(label, "4w")
    fixed = sel.fixed(purge)
    if n_jobs is not None:
        fixed = {**fixed, "n_jobs": int(n_jobs)}
    est = WeekBootstrapEstimator(TimeTailEarlyStopXGB(**{**sel.MODEL_PARAMS, **(params or {})}, **fixed),
                                 bootstrap_seed=c)
    wf = walk_forward_predictions(ctx.pan, est, ctx.world_cfg(train_years), start=t, end=t,
                                  label_col=label, purge_days=purge,
                                  extra_features=tuple(features or ctx.extra))
    return wf.preds.get(t)


def recipe(label: str, train_years: int, features=(), params: dict | None = None) -> dict:
    """A candidate model as `train` walks it: the label, the training window,
    extra panel columns the model gets beyond the panel's f_ columns, and
    overrides of selection.MODEL_PARAMS. Empty features/params are left out,
    so the champion's recipe reads {label, train_years} as its records do."""
    import stocks_ml.selection as sel
    r = {"label": label, "train_years": int(train_years)}
    if features:
        r["features"] = list(features)
    if params:
        bad = [k for k in params if k not in sel.MODEL_PARAMS]
        if bad:
            raise ValueError(f"params {bad} are not in selection.MODEL_PARAMS {tuple(sel.MODEL_PARAMS)}")
        r["params"] = {k: type(sel.MODEL_PARAMS[k])(v) for k, v in params.items()}
    return r


def record(store: str, lo, hi, label: str, train_years: int, k: int, every: int,
           delist: str, price_basis: str, features=(), params: dict | None = None,
           sample: str | None = None, copies=None) -> dict:
    """The walk's record (spec.json): what made it, on what, over which weeks.
    `sample` describes an explicit week list (challenge-fast's stratified
    random sample); such a record is marked a sample like `every` > 1."""
    import stocks_ml.selection as sel
    span = f"{pd.Timestamp(lo).date()} -> {pd.Timestamp(hi).date()}"
    rec = {"code": "stocks_ml.train.walk",
           "recipe": recipe(label, train_years, features, params),
           "k": int(k), "store": store, "delist": delist, "price_basis": price_basis,
           "base_params": {p: str(v) for p, v in sel.MODEL_PARAMS.items()},
           "every": int(every),
           "weeks": (f"{sample} of {span} (a sample)" if sample else
                     f"every week of {span}" if every == 1
                     else f"every {every}th week of {span} (a sample)")}
    if sample:
        rec["sample"] = sample
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


def _init_worker(store: str, n_jobs: int) -> None:
    sel, ctx, _ = context(store)
    _W.update(sel=sel, ctx=ctx, n_jobs=n_jobs)


def _fit_task(args):
    t, c, label, train_years, features, params = args
    p = copy_preds(_W["sel"], _W["ctx"], t, c, label, train_years, features, params,
                   n_jobs=_W["n_jobs"])
    return t, c, p


def walk(store: str, lo, hi, label: str, train_years: int, k: int, out: Path,
         every: int = 1, checkpoint: int = CHECKPOINT, log=log, features=(),
         params: dict | None = None, weeks=None, sample: str | None = None,
         workers: int = 1, copies=None) -> Path:
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
    rec = record(store, lo, hi, label, train_years, k, every, ctx.delist_labels,
                 getattr(ctx.cfg, "price_basis", "closeadj"), features, params, sample, copies)
    guard(out, rec)
    path = out / "preds.parquet"
    frames, done = [], set()
    if path.exists():
        old = pd.read_parquet(path)
        old["week"] = pd.to_datetime(old["week"])
        frames.append(old)
        done = set(old["week"].unique())
    todo = [t for t in weeks if t not in done]
    log(f"train: {rec['weeks']}, {len(todo)} of {len(weeks)} weeks to do, K={k}, "
        f"{label} / {train_years}y on {n_feat} features"
        + (f" + {len(features)} extra" if features else "")
        + (f", params {rec['recipe']['params']}" if params else "") + f", store {store}")
    t0 = time.time()
    ex = None
    if workers > 1 and todo:
        n_jobs = max(1, (os.cpu_count() or workers) // workers)
        ex = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(store, n_jobs))
        log(f"  {workers} worker processes x {n_jobs} threads")
    try:
        done_n = 0
        for b in range(0, len(todo), checkpoint):
            batch = todo[b:b + checkpoint]
            tasks = [(t, c, label, train_years, tuple(features), params) for t in batch
                     for c in copies]
            if ex is not None:
                results = list(ex.map(_fit_task, tasks))          # in task order
            else:
                results = [(t, c, copy_preds(sel, ctx, t, c, label, train_years, features, params))
                           for t, c, *_ in tasks]
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
    saved = pd.read_parquet(preds_path)
    saved["week"] = pd.to_datetime(saved["week"])
    out = {}
    for t in sorted(saved["week"].unique())[:weeks]:
        g = saved[saved.week == t].set_index("ticker")
        for c in copies:
            r = rec["recipe"]
            p = copy_preds(sel, ctx, t, c, r["label"], r["train_years"], r.get("features", ()),
                           r.get("params"))
            a, b = p.align(g[f"c{c}"], join="inner")
            same = len(a) == len(g) and np.allclose(a.values, b.values, atol=1e-6)
            out[f"{pd.Timestamp(t).date()} c{c}"] = bool(same)
            log(f"  {pd.Timestamp(t).date()} copy {c}: {'reproduces' if same else 'DIFFERS'} "
                f"({len(a)} names)")
    return out
