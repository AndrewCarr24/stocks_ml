"""`stocks-ml train`: the walk — the champion model refit every week, K copies.

Training here is walk-forward. At each rank week t the model (selection.
MODEL_PARAMS, a depth-3 XGBoost with time-tail early stopping) is refit on
the trailing `train_years` of panel rows, labels purged, and scores that
week's members. Copy c differs only by its whole-week bootstrap seed
(models/replication.py), so the live job's sixteen copies at one week
(selection.ensemble_preds) and a walk's sixteen copies at that week are the
same fits. The walk writes

    <out>/preds.parquet    week, ticker, c1..cK
    <out>/spec.json        the record: label, training window, K, store, weeks

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
import time
from pathlib import Path

import numpy as np
import pandas as pd

STORE = "data/sharadar_world2000_nominal_dl"    # the research world (nominal basis, last-print labels)
CHECKPOINT = 10                                 # weeks between writes of preds.parquet


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


def copy_preds(sel, ctx, t, c: int, label: str, train_years: int) -> pd.Series | None:
    """Copy c's scores at rank week t: the champion model on the trailing
    `train_years`, bootstrap seed c, trained on `label`."""
    from stocks_ml.models.replication import WeekBootstrapEstimator
    from stocks_ml.models.walk import walk_forward_predictions
    from stocks_ml.models.xgb import TimeTailEarlyStopXGB
    h = sel.HORIZONS["4w"]
    est = WeekBootstrapEstimator(TimeTailEarlyStopXGB(**sel.MODEL_PARAMS, **sel.fixed(h["purge"])),
                                 bootstrap_seed=c)
    wf = walk_forward_predictions(ctx.pan, est, ctx.world_cfg(train_years), start=t, end=t,
                                  label_col=label, purge_days=h["purge"],
                                  extra_features=tuple(ctx.extra))
    return wf.preds.get(t)


def record(store: str, lo, hi, label: str, train_years: int, k: int, every: int,
           delist: str, price_basis: str) -> dict:
    """The walk's record (spec.json): what made it, on what, over which weeks."""
    import stocks_ml.selection as sel
    span = f"{pd.Timestamp(lo).date()} -> {pd.Timestamp(hi).date()}"
    return {"code": "stocks_ml.train.walk",
            "recipe": {"label": label, "train_years": int(train_years)},
            "k": int(k), "store": store, "delist": delist, "price_basis": price_basis,
            "base_params": {p: str(v) for p, v in sel.MODEL_PARAMS.items()},
            "every": int(every),
            "weeks": (f"every week of {span}" if every == 1
                      else f"every {every}th week of {span} (a sample)")}


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


def walk(store: str, lo, hi, label: str, train_years: int, k: int, out: Path,
         every: int = 1, checkpoint: int = CHECKPOINT, log=log) -> Path:
    """Copies 1..k at every rank week of [lo, hi] (every `every`th week for a
    sample), checkpointed and resumed by week under the record's guard."""
    from stocks_ml.selection import HOLDOUT_START, LABELS_4W
    if label not in LABELS_4W:
        raise SystemExit(f"label must be one of {tuple(LABELS_4W)}, got {label!r}")
    lo, hi, out = pd.Timestamp(lo), pd.Timestamp(hi), Path(out)
    if hi >= HOLDOUT_START:
        raise SystemExit(f"--end {hi.date()} reaches the holdout ({HOLDOUT_START.date()}); "
                         "the holdout is never walked without the owner's go")
    sel, ctx, n_feat = context(store)
    weeks = [t for t in ctx.weeks if lo <= t <= hi][::every]
    rec = record(store, lo, hi, label, train_years, k, every, ctx.delist_labels,
                 getattr(ctx.cfg, "price_basis", "closeadj"))
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
        f"{label} / {train_years}y on {n_feat} features, store {store}")
    t0, pending = time.time(), []
    for i, t in enumerate(todo, 1):
        cols = {}
        for c in range(1, k + 1):
            p = copy_preds(sel, ctx, t, c, label, train_years)
            if p is not None:
                cols[f"c{c}"] = p
        df = pd.DataFrame(cols)
        df.index.name = "ticker"
        df = df.reset_index()
        df.insert(0, "week", t)
        pending.append(df)
        if i % checkpoint == 0 or i == len(todo):
            frames.extend(pending)
            pending = []
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
            el = time.time() - t0
            log(f"  {i}/{len(todo)} weeks, {el / i:.0f} s/week, ~{el / i * (len(todo) - i) / 60:.0f} min left")
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
            p = copy_preds(sel, ctx, t, c, rec["recipe"]["label"], rec["recipe"]["train_years"])
            a, b = p.align(g[f"c{c}"], join="inner")
            same = len(a) == len(g) and np.allclose(a.values, b.values, atol=1e-6)
            out[f"{pd.Timestamp(t).date()} c{c}"] = bool(same)
            log(f"  {pd.Timestamp(t).date()} copy {c}: {'reproduces' if same else 'DIFFERS'} "
                f"({len(a)} names)")
    return out
