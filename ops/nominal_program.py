"""The nominal-basis rebuild program (reports/nominal_basis_registration.md).

Registered 2026-09-09 before any number. Everything here runs on the nominal
research world (data/sharadar_world2000_nominal: levels from closeunadj /
close_split; returns unchanged) with K=16 copies, replication.py seeding, and
every grade window ending at selection.HOLDOUT_START exclusive.

    walk --line clean  --seg select   sixteen copies, base features, 2006-2015
    screen                            the generated-bundle screen inside 2006-2015
    walk --line bundle --seg select   sixteen copies with the screened bundle
    walk --line ...    --seg extend   2016-01 -> 2024-07-12, the one look
    grade                             the board: clean vs bundle vs SPY

Usage: PYTHONPATH=src:. STOCKS_ML_PRICE_BASIS=nominal .venv/bin/python \
       ops/nominal_program.py <cmd> [...]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("STOCKS_ML_PRICE_BASIS", "nominal")

from stocks_ml.selection import HOLDOUT_START  # noqa: E402

# STOCKS_ML_NOMINAL_STORE / STOCKS_ML_RUN_SUFFIX parameterize variant runs
# (the delisting-honest world "_dl", 2026-09-10) without new driver code.
STORE = os.environ.get("STOCKS_ML_NOMINAL_STORE", "data/sharadar_world2000_nominal")
SUFFIX = os.environ.get("STOCKS_ML_RUN_SUFFIX", "")
EXP = Path("data/experiments")
SCREEN_OUT = EXP / "nominal_screen_2006_2015"
OUTS = {("clean", "select"): EXP / f"nominal_clean_2006_2015{SUFFIX}",
        ("clean", "extend"): EXP / f"nominal_clean_2016_2024{SUFFIX}",
        ("bundle", "select"): EXP / f"nominal_bundle_2006_2015{SUFFIX}",
        ("bundle", "extend"): EXP / f"nominal_bundle_2016_2024{SUFFIX}"}
SEGS = {"select": (pd.Timestamp("2006-01-01"), pd.Timestamp("2015-12-31")),
        "extend": (pd.Timestamp("2016-01-01"), HOLDOUT_START - pd.Timedelta(days=1))}
WINDOWS = {"2006_2015": (pd.Timestamp("2006"), pd.Timestamp("2016")),
           "2016_2024": (pd.Timestamp("2016"), HOLDOUT_START),
           "pre_holdout": (pd.Timestamp("2006"), HOLDOUT_START)}
CHAMPION = dict(horizon="4w", book=6, cap=2, stop=None, floor="70/30")
K = 16
CHECKPOINT = 10
REPORT = Path(f"reports/nominal_board{os.environ.get('STOCKS_ML_RUN_SUFFIX', '')}.md")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def context(line):
    import stocks_ml.selection as sel
    if line == "clean":
        ctx = sel.Ctx(STORE)
        ctx.extra = []
        return sel, ctx, {}
    from stocks_ml.features import generated as gen
    formulas = json.loads((SCREEN_OUT / "formulas.json").read_text())
    ctx = sel.Ctx(STORE)
    g = pd.read_parquet(SCREEN_OUT / "generated.parquet")
    g = g.set_index(pd.MultiIndex.from_frame(g[["date", "ticker"]])).drop(columns=["date", "ticker"])
    g = g.reindex(pd.MultiIndex.from_frame(ctx.pan[["date", "ticker"]]))
    g.index = ctx.pan.index
    ctx.pan = pd.concat(
        [ctx.pan.drop(columns=[c for c in ctx.pan.columns if c.startswith(gen.PREFIX)]),
         g.fillna(0.0)], axis=1)
    ctx.extra = list(formulas)
    return sel, ctx, formulas


def _guard_spec(out: Path, line, formulas):
    """preds.parquet resumes by week: refuse to resume under another recipe."""
    from stocks_ml.selection import K_COPIES, MODEL_PARAMS
    spec = {"k_copies": K_COPIES, "model_params": {k: str(v) for k, v in MODEL_PARAMS.items()},
            "line": line, "features": sorted(formulas), "basis": "nominal", "store": STORE,
            "delist": os.environ.get("STOCKS_ML_DELIST_LABELS", "drop")}
    sp = out / "spec.json"
    if sp.exists():
        old = json.loads(sp.read_text())
        if old != spec:
            raise RuntimeError(f"{out} was written under {old}; current recipe {spec}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(spec, indent=1, sort_keys=True))


def walk(line, seg):
    from ops.k16_seed_spread import copy_preds
    sel, ctx, formulas = context(line)
    lo, hi = SEGS[seg]
    out = OUTS[(line, seg)]
    _guard_spec(out, line, formulas)
    h = sel.HORIZONS["4w"]
    cfg2 = ctx.world_cfg(5)
    weeks = [t for t in ctx.weeks if lo <= t <= hi]
    path = out / "preds.parquet"
    frames, done = [], set()
    if path.exists():
        old = pd.read_parquet(path)
        old["week"] = pd.to_datetime(old["week"])
        frames.append(old)
        done = set(old["week"].unique())
    todo = [t for t in weeks if t not in done]
    log(f"walk {line}/{seg}: {len(weeks)} rank weeks {weeks[0].date()} -> {weeks[-1].date()}, "
        f"{len(todo)} to do, K={K}; panel {ctx.pan.shape}, extra={len(ctx.extra)}")
    t0, pending = time.time(), []
    for i, t in enumerate(todo, 1):
        cols = {}
        for c in range(1, K + 1):
            p = copy_preds(sel, ctx, t, c, h, cfg2)
            if p is not None:
                cols[f"c{c}"] = p
        if cols:
            df = pd.DataFrame(cols)
            df.index.name = "ticker"
            df = df.reset_index()
            df.insert(0, "week", t)
            pending.append(df)
        if i % CHECKPOINT == 0 or i == len(todo):
            frames.extend(pending)
            pending = []
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
            el = time.time() - t0
            log(f"  {i}/{len(todo)} weeks, {el / i:.1f} s/week, "
                f"~{el / i * (len(todo) - i) / 3600:.2f} h left")
    log(f"walk {line}/{seg}: done in {(time.time() - t0) / 3600:.2f} h -> {path}")


def screen():
    import ops.openfe_arm as oa
    oa.STORE = STORE
    oa.OUT = SCREEN_OUT
    SCREEN_OUT.mkdir(parents=True, exist_ok=True)
    if not (SCREEN_OUT / "inputs.parquet").exists():
        oa.inputs()
    else:
        log("screen: inputs cached")
    oa.fit()
    oa.features()


def load_preds(line):
    frames = [pd.read_parquet(OUTS[(line, seg)] / "preds.parquet") for seg in SEGS
              if (OUTS[(line, seg)] / "preds.parquet").exists()]
    df = pd.concat(frames, ignore_index=True)
    df["week"] = pd.to_datetime(df["week"])
    return df.sort_values(["week", "ticker"])


def grade():
    from ops.k16_seed_spread import ensemble_rows
    from ops.k16_program import cascade_at
    from stocks_ml.models.trials import record_trials
    res, led, series = {}, [], {}
    sp500 = None
    for line in ("clean", "bundle"):
        try:
            sel, ctx, _ = context(line)
        except FileNotFoundError:
            log(f"grade: no {line} artifacts yet, skipping")
            continue
        preds = load_preds(line)
        entry = {"rank_weeks": int(preds.week.nunique()),
                 "first": str(preds.week.min().date()), "last": str(preds.week.max().date())}
        ens = {"K=16": range(1, K + 1)}
        ens.update({f"K=4 copies {r.start}-{r.stop - 1}": r
                    for r in (range(1, 5), range(5, 9), range(9, 13), range(13, 17))})
        for name, copies in ens.items():
            rows, _ = ensemble_rows(sel, ctx, preds, copies)
            rows = rows.sort_values("week").reset_index(drop=True)
            s = sel.simulate(ctx, rows, **CHAMPION)
            spy = ctx.wret["SPY"].reindex(s.index)
            entry[name] = {w: {"config": sel.metrics(s, *b), "sp500": sel.metrics(spy, *b)}
                           for w, b in WINDOWS.items()
                           if (s.index >= b[0]).any() and (s.index < b[1]).any()}
            if name == "K=16":
                series[line] = s
                sp500 = spy
                entry["cascade_2006_2015"] = cascade_at(sel, ctx, rows)
        res[line] = entry
        led.append({"kind": "r4w_strategy", "name": f"nominal_{line}_k16",
                    "n_weeks": entry["rank_weeks"], "windows": {
                        w: entry["K=16"][w]["config"] for w in entry["K=16"]},
                    "notes": f"Nominal-basis {line} line, K=16 at the champion's settings "
                             f"(reports/nominal_basis_registration.md); windows end "
                             f"{HOLDOUT_START.date()} exclusive."})
    out = EXP / f"nominal_grades{SUFFIX}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    if led:
        record_trials(led)
    log(f"grade -> {out}")
    _report(res, series, sp500)


def gl(m):
    return (f"${m['terminal_100']:,.0f} ({m['cagr_pct']:+.1f}%/yr, SR {m['sharpe']:.2f}, "
            f"DD {m['max_dd']:.0%})")


def _report(res, series, sp500):
    md = ["# The board on the nominal basis", "",
          f"Generated by `ops/nominal_program.py grade` on {pd.Timestamp.today().date()}; "
          "preregistered in reports/nominal_basis_registration.md before any number. "
          "K=16 at the champion's settings (top-6 / cap 2 / no stop / 70-30), fills at the "
          "next open, 5 bp a side; every window ends at the holdout exclusive. The old "
          "record ($4,256 / $900 / $473 vs SPY $608 / $310 / $197) is contaminated-basis "
          "history, quoted for scale only.", ""]
    for w in WINDOWS:
        md += [f"## {w}", "", "| line | terminal $100 | CAGR | Sharpe | DD | weeks |", "|---|---|---|---|---|---|"]
        for line, entry in res.items():
            m = entry.get("K=16", {}).get(w)
            if m:
                c = m["config"]
                md.append(f"| {line} (K=16) | ${c['terminal_100']:,.0f} | {c['cagr_pct']:+.1f}% | "
                          f"{c['sharpe']:.2f} | {c['max_dd']:.0%} | {c['n_weeks']} |")
        any_line = next(iter(res.values()), {})
        m = any_line.get("K=16", {}).get(w)
        if m:
            c = m["sp500"]
            md.append(f"| sp500 | ${c['terminal_100']:,.0f} | {c['cagr_pct']:+.1f}% | "
                      f"{c['sharpe']:.2f} | {c['max_dd']:.0%} | {c['n_weeks']} |")
        md.append("")
    REPORT.write_text("\n".join(md))
    log(f"report -> {REPORT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("walk")
    w.add_argument("--line", choices=("clean", "bundle"), required=True)
    w.add_argument("--seg", choices=("select", "extend"), required=True)
    sub.add_parser("screen")
    sub.add_parser("grade")
    a = ap.parse_args()
    if a.cmd == "walk":
        walk(a.line, a.seg)
    elif a.cmd == "screen":
        screen()
    else:
        grade()
