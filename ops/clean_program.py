"""Improving the clean line (reports/clean_improvement_registration.md).

Registered 2026-09-11 before any number. Everything runs on the
delisting-honest research world (data/sharadar_world2000_nominal_dl: nominal
basis, last-print labels); the incumbent is the deployed champion's saved
K=16 predictions there. Selection reads 2006-2015 only; 2016-2024 is read
once, in stage E, for the frozen package.

    stage_a                 strategy layers on the incumbent's K=16 preds (free)
    sweep --variant <id>    one model variant, every 4th week of 2006-2015, K=4
    sweep --all             all fourteen registered variants, sequentially
    stage_b                 the sample table: variants vs the incumbent's copies 1-4
    walk --candidate <id>   one stage C candidate, every week of 2006-2015, K=4
    walk --all              the six stage C candidates (owner's go 2026-09-11)
    stage_c                 the frozen-model table + leak audit per candidate
    stage_d                 feature families on the frozen model (owner's go
                            2026-09-11): build -> probe -> dedup -> walk -> exam;
                            resumes at every step
    extend --arm <id>       a stage D arm at K=4 on every week of 2016-2024 (owner's
                            ask 2026-09-11: the pre-holdout record before stage E)
    assess                  the arms, the incumbent and the S&P 500 at the deployed
                            settings: 2006-2015 | 2016-2024 | 2006-2024
    stage_e --arm <id>      the package (owner's go 2026-09-11): K=16 on every week of
                            2006-2015 -> the procedure decides its layers (spec not
                            written) -> K=16 on 2016-2024 -> falsification test, leak
                            audit, CIs, ledger row; resumes at every step
    stage_e_amend --arm <id>
                            refresh an existing stage_e.json: the package-at-deployed-
                            settings row (one simulation) and the leak audit (seconds);
                            walks and CIs untouched; ledger row upserted
    champion_chart          reports/champion_vs_sp500_{2006,2016}_2024.png: the
                            champion the spec names (its stage E walk, its written
                            settings) against the S&P 500, pre-holdout; one simulation,
                            cached beside the walk; needs matplotlib
                            (`uv run --with matplotlib python ops/clean_program.py champion_chart`)

Usage: PYTHONPATH=src:. .venv/bin/python ops/clean_program.py <cmd> [...]
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
os.environ.setdefault("STOCKS_ML_DELIST_LABELS", "last_print")

from stocks_ml.selection import HOLDOUT_START  # noqa: E402

STORE = "data/sharadar_world2000_nominal_dl"
EXP = Path("data/experiments")
INCUMBENT = {"select": EXP / "nominal_clean_2006_2015_dl" / "preds.parquet",
             "extend": EXP / "nominal_clean_2016_2024_dl" / "preds.parquet"}
OUT = EXP / "clean_program"
SELECT = (pd.Timestamp("2006-01-01"), pd.Timestamp("2015-12-31"))
EXTEND = (pd.Timestamp("2016-01-01"), HOLDOUT_START - pd.Timedelta(days=1))
K_SAMPLE = 4
SAMPLE_EVERY = 4
CHECKPOINT = 10
REPORT = Path("reports/clean_improvement.md")

# Fourteen single-knob variants of the incumbent (registration, stage B).
# params: merged into selection.MODEL_PARAMS; early_stop False: plain
# XGBRegressor on every row of the window (no 10% time tail held out);
# train_years: the walk's window; label: a column the driver adds to the panel.
VARIANTS: dict[str, dict] = {
    "depth2": dict(params=dict(max_depth=2)),
    "depth4": dict(params=dict(max_depth=4)),
    "depth6": dict(params=dict(max_depth=6)),
    "lr01": dict(params=dict(learning_rate=0.01)),
    "lr05": dict(params=dict(learning_rate=0.05)),
    "mcw5": dict(params=dict(min_child_weight=5)),
    "mcw100": dict(params=dict(min_child_weight=100)),
    "col50": dict(params=dict(colsample_bytree=0.5)),
    "col30": dict(params=dict(colsample_bytree=0.3)),
    "fixed300": dict(params=dict(n_estimators=300), early_stop=False),
    "win3": dict(train_years=3),
    "win8": dict(train_years=8),
    "lab_gauss": dict(label="label_4w_gauss"),
    "lab_sector": dict(label="label_4w_sector"),
}
INCUMBENT_VARIANT: dict = {}
# Stage C (owner's go 2026-09-11): the four highest stage B sample metrics
# plus two combinations of the top two knobs, with and without the repaired
# stop (reports/clean_improvement.md).
CANDIDATES: dict[str, dict] = {
    "lab_sector": VARIANTS["lab_sector"],
    "win8": VARIANTS["win8"],
    "lr01": VARIANTS["lr01"],
    "fixed300": VARIANTS["fixed300"],
    "ls_w8": dict(label="label_4w_sector", train_years=8),
    "ls_w8_f300": dict(label="label_4w_sector", train_years=8,
                       params=dict(n_estimators=300), early_stop=False),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- context
def add_labels(pan: pd.DataFrame, smap: dict) -> pd.DataFrame:
    """The two label variants, from fwd_ret_4w only (never a feature)."""
    from scipy.stats import norm
    fwd = pan["fwd_ret_4w"]
    r = fwd.groupby(pan["date"]).rank(method="average")
    n = fwd.notna().groupby(pan["date"]).transform("sum")
    pan["label_4w_gauss"] = norm.ppf((r - 0.5) / n)
    # label_4w_sector: the package rule (features/panel.sector_label), which
    # Ctx already applies when the stored panel lacks the column.
    from stocks_ml.features.panel import sector_label
    pan["label_4w_sector"] = sector_label(fwd, pan["date"], pan["ticker"].map(smap))
    return pan


def deployed_settings():
    """The strategy layers as models/champion_spec.json carries them — the
    procedure's decision (stocks_ml.procedure), read, never typed here."""
    d = json.loads(Path("models/champion_spec.json").read_text())["procedure"]["decision"]
    return dict(book=d["book_size"], floor=d["floor"], stop=d["stop_loss"], cap=d["sector_cap"])


def context():
    import stocks_ml.selection as sel
    from stocks_ml.features.panel import feature_cols
    ctx = sel.Ctx(STORE)
    ctx.extra = []
    assert ctx.delist_labels == "last_print", ctx.delist_labels
    ctx.pan = add_labels(ctx.pan, ctx.smap)
    fc = feature_cols(ctx.pan)
    bad = [c for c in fc if c.startswith(("label", "fwd_ret", "x_", "g_"))]
    assert not bad and len(fc) == 64, (len(fc), bad)
    return sel, ctx


def sample_weeks(ctx):
    lo, hi = SELECT
    return [t for t in ctx.weeks if lo <= t <= hi][::SAMPLE_EVERY]


# ----------------------------------------------------------------------------- one copy
def estimator(sel, variant: dict, c: int, purge: int):
    from xgboost import XGBRegressor
    from stocks_ml.models.replication import WeekBootstrapEstimator
    from stocks_ml.models.xgb import TimeTailEarlyStopXGB
    params = {**sel.MODEL_PARAMS, **variant.get("params", {})}
    if variant.get("early_stop", True):
        base = TimeTailEarlyStopXGB(**params, **sel.fixed(purge))
    else:
        base = XGBRegressor(**params, n_jobs=-1, random_state=0)
    return WeekBootstrapEstimator(base, bootstrap_seed=c)


def variant_preds(sel, ctx, t, c, variant: dict):
    """One copy at rank week t under the variant, otherwise exactly
    ops.k16_seed_spread.copy_preds."""
    from stocks_ml.models.walk import walk_forward_predictions
    h = sel.HORIZONS["4w"]
    cfg2 = ctx.world_cfg(variant.get("train_years", 5))
    est = estimator(sel, variant, c, h["purge"])
    wf = walk_forward_predictions(ctx.pan, est, cfg2, start=t, end=t,
                                  label_col=variant.get("label", h["label"]),
                                  purge_days=h["purge"], extra_features=tuple(ctx.extra))
    return wf.preds.get(t)


def _guard_spec(out: Path, spec: dict):
    sp = out / "spec.json"
    if sp.exists():
        old = json.loads(sp.read_text())
        if old != spec:
            raise RuntimeError(f"{out} was written under {old}; current recipe {spec}")
    else:
        out.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(spec, indent=1, sort_keys=True))


def sweep(vid: str, full: bool = False, variant: dict | None = None, sel_ctx=None,
          out: Path | None = None, window=None):
    """K=4 copies of a variant at each week: the stage B sample (every 4th
    week of 2006-2015) or, for a stage C candidate, every week of it. Stage D
    passes its own variant (with `extra`: derived panel columns the model
    gets by name), context and output directory."""
    sel, ctx = sel_ctx or context()
    variant = variant or (CANDIDATES if full else VARIANTS)[vid]
    ctx.extra = list(variant.get("extra", []))
    out = out or OUT / ("full" if full else "sweep") / vid
    lo, hi = window or SELECT
    weeks = ([t for t in ctx.weeks if lo <= t <= hi] if full else sample_weeks(ctx))
    span = f"{lo.date()} -> {hi.date()}"
    _guard_spec(out, {"variant": vid, "recipe": json.loads(json.dumps(variant)),
                      "base_params": {k: str(v) for k, v in sel.MODEL_PARAMS.items()},
                      "k": K_SAMPLE, "weeks": (f"every week of {span}" if full else
                                               f"every {SAMPLE_EVERY}th of {span}"),
                      "store": STORE, "delist": ctx.delist_labels})
    path = out / "preds.parquet"
    frames, done = [], set()
    if path.exists():
        old = pd.read_parquet(path)
        old["week"] = pd.to_datetime(old["week"])
        frames.append(old)
        done = set(old["week"].unique())
    todo = [t for t in weeks if t not in done]
    log(f"sweep {vid}: {len(weeks)} sample weeks, {len(todo)} to do, K={K_SAMPLE}; {variant}")
    t0, pending = time.time(), []
    for i, t in enumerate(todo, 1):
        cols = {}
        for c in range(1, K_SAMPLE + 1):
            p = variant_preds(sel, ctx, t, c, variant)
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
            log(f"  {vid} {i}/{len(todo)} weeks, {el / i:.1f} s/week, "
                f"~{el / i * (len(todo) - i) / 60:.0f} min left")
    log(f"sweep {vid}: done in {(time.time() - t0) / 60:.1f} min -> {path}")


# ----------------------------------------------------------------------------- stage A
def load_incumbent(seg="select"):
    df = pd.read_parquet(INCUMBENT[seg])
    df["week"] = pd.to_datetime(df["week"])
    return df.sort_values(["week", "ticker"])


def stage_a():
    """Strategy layers on the incumbent's K=16 predictions over 2006-2015:
    the cascade as run_cascade decides it, plus the full book x floor grid."""
    from ops.k16_program import cascade_at
    from ops.k16_seed_spread import ensemble_rows
    sel, ctx = context()
    preds = load_incumbent("select")
    rows, _ = ensemble_rows(sel, ctx, preds, range(1, 17))
    rows = rows.sort_values("week").reset_index(drop=True)
    lo, hi = SELECT
    hold = rows[(rows.week >= lo) & (rows.week <= hi)]
    cas = cascade_at(sel, ctx, hold)
    grid = {}
    for book in sel.BOOKS:
        for floor in sel.FLOORS:
            s = sel.simulate(ctx, hold, "4w", book, None, None, floor)
            grid[f"{book}/{floor}"] = {"sharpe": round(sel.sharpe(s, lo, hi), 3),
                                       **{k: v for k, v in sel.metrics(s, lo, hi).items()
                                          if k in ("terminal_100", "cagr_pct", "max_dd")}}
    res = {"weeks": int(hold.week.nunique()), "cascade": cas, "grid": grid,
           "deployed": deployed_settings()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage_a.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"stage A -> {OUT / 'stage_a.json'}: cascade {cas['book']}/{cas['floor']}/"
        f"{cas['stop']}/{cas['cap']}")
    return res


# ----------------------------------------------------------------------------- stage B
def excess_series(sel, ctx, preds: pd.DataFrame, copies, weeks) -> pd.Series:
    """top6 - rand_mean at each week for the mean of the given copies."""
    cols = [f"c{c}" for c in copies]
    out = {}
    for t, g in preds[preds.week.isin(weeks)].groupby("week"):
        p = g.set_index("ticker")[[c for c in cols if c in g.columns]].mean(axis=1)
        if p.nunique() < 20:
            continue
        row = sel.slice_row(ctx, t, "4w", p)
        if row is not None:
            out[t] = row["top6"] - row["rand_mean"]
    return pd.Series(out).sort_index()


def paired_t(x: pd.Series):
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 else float("nan")


def stage_b():
    sel, ctx = context()
    weeks = sample_weeks(ctx)
    inc = load_incumbent("select")
    inc = inc[inc.week.isin(weeks)]
    rows = {}

    def entry(name, preds):
        cols = [f"c{c}" for c in range(1, K_SAMPLE + 1)]
        copies = {c: excess_series(sel, ctx, preds, [c], weeks) for c in range(1, K_SAMPLE + 1)}
        ens = excess_series(sel, ctx, preds, range(1, K_SAMPLE + 1), weeks)
        # a copy whose early stop kept 0-7 rounds scores ~490 names with <20
        # distinct values: a collapsed fit (finding of 2026-09-11, stage B)
        nun = preds[preds.week.isin(weeks)].groupby("week")[cols].nunique()
        return {"weeks": int(len(ens)), "ens": ens, "pred_weeks": int(preds.week.nunique()),
                "collapsed": round(float((nun < 20).values.mean()), 3),
                "copy_means": [round(float(s.mean()) * 100, 3) for s in copies.values()],
                "k4_mean": round(float(ens.mean()) * 100, 3)}

    rows["incumbent"] = entry("incumbent", inc)
    base = rows["incumbent"]["ens"]
    for vid in VARIANTS:
        path = OUT / "sweep" / vid / "preds.parquet"
        if not path.exists():
            continue
        p = pd.read_parquet(path)
        p["week"] = pd.to_datetime(p["week"])
        e = entry(vid, p)
        if e["pred_weeks"] < len(weeks):
            e["partial"] = True
        d = (e["ens"] - base).dropna()
        e["diff_mean"] = round(float(d.mean()) * 100, 3)
        e["t"] = round(paired_t(d), 2)
        e["win_rate"] = round(float((d > 0).mean()), 2)
        rows[vid] = e
    table = {k: {kk: vv for kk, vv in v.items() if kk != "ens"} for k, v in rows.items()}
    (OUT / "stage_b.json").write_text(json.dumps(table, indent=1, default=str))
    order = sorted(table, key=lambda k: -table[k]["k4_mean"])
    md = ["| variant | K=4 mean excess %/4w | copies 1-4 | vs incumbent (pp) | paired t | win rate | collapsed fits | weeks |",
          "|---|---|---|---|---|---|---|---|"]
    for k in order:
        v = table[k]
        md.append(f"| {k}{' (partial)' if v.get('partial') else ''} | {v['k4_mean']:+.3f} | "
                  f"{', '.join(f'{c:+.2f}' for c in v['copy_means'])} | "
                  f"{v.get('diff_mean', 0):+.3f} | {v.get('t', float('nan')):+.2f} | "
                  f"{v.get('win_rate', float('nan')):.2f} | {v['collapsed']:.0%} | {v['weeks']} |")
    print("\n".join(md))
    log(f"stage B -> {OUT / 'stage_b.json'}")
    return table, md


# ----------------------------------------------------------------------------- stage C
def single_rows(sel, ctx, preds, c):
    """One copy's slice_row at every week it scored, ranking ties as they
    fall (a collapsed copy ranks a handful of distinct values): what the copy
    alone would have done, on every week, so the four values share the
    ensemble's weeks instead of skipping its collapsed (crisis-heavy) ones."""
    rows = []
    for t, g in preds.groupby("week"):
        p = g.set_index("ticker")[f"c{c}"].dropna()
        row = sel.slice_row(ctx, t, "4w", p)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)


def stage_c():
    """The frozen model: the registered selection metric (top-6 cost-adjusted
    compounded %/yr, K=4 copies 1-4) over {incumbent} + candidates on the
    weeks of 2006-2015 every complete walk ranked (same basis), with the
    single-copy values on those weeks, the paired weekly t of top-6 excess vs
    the incumbent, the deployed-settings simulation for information, and the
    leak audit per candidate walk."""
    from ops.k16_seed_spread import ensemble_rows
    from ops.leak_audit import audit_segments
    sel, ctx = context()
    lo, hi = SELECT
    last = sel.label_end(hi, 4)
    copies = range(1, K_SAMPLE + 1)

    def load(vid):
        if vid == "incumbent":
            return load_incumbent("select")
        path = OUT / "full" / vid / "preds.parquet"
        if not path.exists():
            return None
        p = pd.read_parquet(path)
        p["week"] = pd.to_datetime(p["week"])
        return p

    ents = {}
    for vid in ["incumbent", *CANDIDATES]:
        preds = load(vid)
        if preds is None:
            continue
        preds = preds[(preds.week >= lo) & (preds.week <= hi)]
        rows, _ = ensemble_rows(sel, ctx, preds, copies)
        ents[vid] = {"k4": rows.sort_values("week").reset_index(drop=True),
                     "single": [single_rows(sel, ctx, preds, c) for c in copies],
                     "pred_weeks": int(preds.week.nunique()),
                     "partial": int(preds.week.nunique()) < len([t for t in ctx.weeks if lo <= t <= hi])}
    complete = [v for v, e in ents.items() if not e["partial"]]
    common = None
    for v in complete:
        w = set(ents[v]["k4"].week)
        common = w if common is None else common & w
    common = sorted(common)
    log(f"stage C: {len(common)} common weeks across {complete}")
    base = ents["incumbent"]["k4"].set_index("week")
    base_ex = (base["top6"] - base["rand_mean"]).reindex(common)
    table = {}
    for vid, e in ents.items():
        wk = common if not e["partial"] else sorted(set(e["k4"].week))
        hold = e["k4"][e["k4"].week.isin(wk)]
        h = hold.set_index("week")
        ex = (h["top6"] - h["rand_mean"]).reindex(wk)
        dep = deployed_settings()
        s = sel.simulate(ctx, hold, "4w", dep["book"], dep["cap"], dep["stop"], dep["floor"])
        m = sel.metrics(s, lo, pd.Timestamp("2016-01-01"))
        d = (ex - base_ex).dropna()
        row = {"weeks": len(wk), "partial": e["partial"],
               "metric": round(sel.compounded_pct(hold, "top6", 4, lo, last), 2),
               "copies": [round(sel.compounded_pct(r[r.week.isin(wk)], "top6", 4, lo, last), 2)
                          for r in e["single"]],
               "diff_pp_4w": round(float(d.mean()) * 100, 3), "t": round(paired_t(d), 2),
               "deployed_settings": {k: m[k] for k in ("terminal_100", "cagr_pct", "sharpe", "max_dd")}}
        if vid != "incumbent":
            row["leak_audit"] = audit_segments(STORE, [str(OUT / "full" / vid / "preds.parquet")])
        table[vid] = row
    table["frozen"] = max((k for k in table if not table[k]["partial"]),
                          key=lambda k: table[k]["metric"])
    table["common_weeks"] = len(common)
    (OUT / "stage_c.json").write_text(json.dumps(table, indent=1, default=str))
    dep = deployed_settings()
    dep_label = f"{dep['book']}/{dep['floor']}/cap{dep['cap']}/stop {dep['stop']}"
    md = [f"| candidate | top-6 compounded %/yr | copies 1-4 | vs incumbent (pp/4w) | paired t | {dep_label}: $100, SR, DD | leak audit | weeks |",
          "|---|---|---|---|---|---|---|---|"]
    names = [k for k in table if k not in ("frozen", "common_weeks")]
    for k in sorted(names, key=lambda k: -table[k]["metric"]):
        v = table[k]
        ds = v["deployed_settings"]
        la = v.get("leak_audit")
        md.append(f"| {k}{' (partial)' if v['partial'] else ''} | {v['metric']:+.2f} | "
                  f"{', '.join(f'{c:+.2f}' for c in v['copies'])} | {v['diff_pp_4w']:+.3f} | "
                  f"{v['t']:+.2f} | ${ds['terminal_100']:,.0f}, {ds['sharpe']:.2f}, {ds['max_dd']:.0%} | "
                  f"{(la['VERDICT'] + (f' (retention {la['worst_retention']:.2f})' if la.get('worst_retention') is not None else '')) if la else '—'} | "
                  f"{v['weeks']} |")
    md.append(f"\nfrozen model (argmax of the metric among complete walks, "
              f"{len(common)} common weeks): **{table['frozen']}**")
    print("\n".join(md))
    log(f"stage C -> {OUT / 'stage_c.json'}")
    return table, md


# ----------------------------------------------------------------------------- stage D
DEDUP_CORR = 0.7
STAGE_D = OUT / "stage_d"


def frozen_model() -> str:
    """Stage C's argmax, read from its output — never typed here."""
    return json.loads((OUT / "stage_c.json").read_text())["frozen"]


def context_d():
    """context() with stage D's two derived families appended to the panel
    (features/derived.py: sector-relative and rank momentum of the 50
    per-stock ranked features)."""
    from stocks_ml.features.derived import add_derived
    sel, ctx = context()
    ctx.pan, names = add_derived(ctx.pan, ctx.smap)
    return sel, ctx, names


def dedup(res: pd.DataFrame, win: pd.DataFrame, bar: float = DEDUP_CORR) -> list[str]:
    """The keepers in |t| order, each admitted only if its |corr| with every
    feature already in the bundle is <= bar (pooled over the window's rows)."""
    keep = res[res["keep"]].copy()
    keep = keep.reindex(keep["t"].abs().sort_values(ascending=False).index)
    order = keep["feature"].tolist()
    if not order:
        return []
    corr = win[order].corr().abs()
    bundle = []
    for f in order:
        if all(corr.loc[f, g] <= bar for g in bundle):
            bundle.append(f)
    return bundle


def stage_d_screen(sel, ctx, names: list[str], label: str) -> dict:
    """The probe (feature_screen.probe: weekly Spearman IC vs the frozen
    model's label on 2006-2015, NW t, same sign in both halves, keep iff
    |t| >= 2) and the dedup. Writes stage_d/probe.csv and screen.json."""
    import stocks_ml.feature_screen as fs
    from stocks_ml.features.panel import feature_cols
    lo, hi = SELECT
    STAGE_D.mkdir(parents=True, exist_ok=True)
    sp = STAGE_D / "screen.json"
    if sp.exists():
        return json.loads(sp.read_text())
    t0 = time.time()
    res = fs.probe(ctx.pan, lo, hi, kweeks=4, label=label, candidates=names)
    res.to_csv(STAGE_D / "probe.csv", index=False)
    win = fs.probe_window(ctx.pan, lo, hi, 4)
    bundle = dedup(res, win)
    base = feature_cols(ctx.pan)
    with_base = {f: round(float(win[base].corrwith(win[f]).abs().max()), 3) for f in bundle}
    out = {"label": label, "window": [str(lo.date()), str(hi.date())], "candidates": len(names),
           "families": {"sector_relative": sum(n.startswith("d_sec_") for n in names),
                        "rank_momentum": sum(n.startswith("d_mom") for n in names)},
           "probe_rule": f"|NW t| >= {fs.T_BAR} and same IC sign in both halves of the window",
           "keepers": fs.keepers(res), "dedup_rule": f"greedy by |t|, pairwise |corr| <= {DEDUP_CORR}",
           "bundle": bundle, "max_abs_corr_with_base": with_base,
           "probe": {r.feature: {"ic": round(r.ic, 4), "t": round(r.t, 2), "ic_a": round(r.ic_a, 4),
                                 "ic_b": round(r.ic_b, 4), "weeks": int(r.weeks)}
                     for r in res[res["keep"]].itertuples()}}
    sp.write_text(json.dumps(out, indent=1, default=str))
    log(f"stage D screen: {len(names)} candidates, {len(out['keepers'])} keepers, "
        f"bundle of {len(bundle)} after dedup, {(time.time() - t0) / 60:.1f} min -> {sp}")
    return out


def stage_d_exam(sel, ctx, fid: str, bundle: list[str]) -> tuple[dict, list[str]]:
    """The registered bar: the selection metric at K=4 on every week of
    2006-2015, frozen model with the bundle vs without, on the weeks both
    ranked with a real ensemble; admitted iff with > without."""
    from ops.k16_seed_spread import ensemble_rows
    from ops.leak_audit import audit_segments
    lo, hi = SELECT
    last = sel.label_end(hi, 4)
    copies = range(1, K_SAMPLE + 1)
    paths = {"without": OUT / "full" / fid / "preds.parquet",
             "with": STAGE_D / "walk" / "preds.parquet"}
    ents = {}
    for arm, path in paths.items():
        p = pd.read_parquet(path)
        p["week"] = pd.to_datetime(p["week"])
        p = p[(p.week >= lo) & (p.week <= hi)]
        n_weeks = len([t for t in ctx.weeks if lo <= t <= hi])
        if p.week.nunique() < n_weeks:
            raise RuntimeError(f"{path}: {p.week.nunique()} of {n_weeks} weeks — finish the walk")
        rows, _ = ensemble_rows(sel, ctx, p, copies)
        ents[arm] = {"k4": rows.sort_values("week").reset_index(drop=True),
                     "single": [single_rows(sel, ctx, p, c) for c in copies]}
    common = sorted(set(ents["without"]["k4"].week) & set(ents["with"]["k4"].week))
    ex = {}
    table = {}
    for arm, e in ents.items():
        hold = e["k4"][e["k4"].week.isin(common)]
        h = hold.set_index("week")
        ex[arm] = (h["top6"] - h["rand_mean"]).reindex(common)
        dep = deployed_settings()
        s = sel.simulate(ctx, hold, "4w", dep["book"], dep["cap"], dep["stop"], dep["floor"])
        m = sel.metrics(s, lo, pd.Timestamp("2016-01-01"))
        table[arm] = {"metric": round(sel.compounded_pct(hold, "top6", 4, lo, last), 2),
                      "copies": [round(sel.compounded_pct(r[r.week.isin(common)], "top6", 4, lo, last), 2)
                                 for r in e["single"]],
                      "deployed_settings": {k: m[k] for k in ("terminal_100", "cagr_pct", "sharpe", "max_dd")}}
    d = (ex["with"] - ex["without"]).dropna()
    table["with"]["diff_pp_4w"] = round(float(d.mean()) * 100, 3)
    table["with"]["t"] = round(paired_t(d), 2)
    table["with"]["leak_audit"] = audit_segments(STORE, [str(paths["with"])])
    table["frozen_model"] = fid
    table["bundle"] = bundle
    table["common_weeks"] = len(common)
    table["rule"] = "admitted iff the selection metric with the bundle > without (same weeks, K=4)"
    table["admitted"] = bool(table["with"]["metric"] > table["without"]["metric"])
    (OUT / "stage_d.json").write_text(json.dumps(table, indent=1, default=str))
    dep_label = f"{dep['book']}/{dep['floor']}/cap{dep['cap']}/stop {dep['stop']}"
    md = [f"| arm | top-6 compounded %/yr | copies 1-4 | vs without (pp/4w) | paired t | {dep_label}: $100, SR, DD | leak audit | weeks |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in ("without", "with"):
        v = table[arm]
        ds = v["deployed_settings"]
        la = v.get("leak_audit")
        name = f"{fid}" if arm == "without" else f"{fid} + engineered features*"
        md.append(f"| {name} | {v['metric']:+.2f} | {', '.join(f'{c:+.2f}' for c in v['copies'])} | "
                  f"{v.get('diff_pp_4w', 0):+.3f} | {v.get('t', float('nan')):+.2f} | "
                  f"${ds['terminal_100']:,.0f}, {ds['sharpe']:.2f}, {ds['max_dd']:.0%} | "
                  f"{(la['VERDICT'] + (f' (retention {la['worst_retention']:.2f})' if la.get('worst_retention') is not None else '')) if la else '—'} | "
                  f"{len(common)} |")
    md.append(f"\nbundle ({len(bundle)}): {', '.join(bundle) or 'none'}")
    md.append(f"\n**{'ADMITTED' if table['admitted'] else 'NOT ADMITTED'}** — {table['rule']}")
    print("\n".join(md))
    log(f"stage D -> {OUT / 'stage_d.json'}: {'admitted' if table['admitted'] else 'not admitted'}")
    return table, md


def stage_d():
    """Build the derived families, probe and dedup them on 2006-2015 against
    the frozen model's label, walk the frozen model WITH the bundle at K=4 on
    every week of 2006-2015, and apply the registered bar. Every step
    checkpoints; a rerun resumes."""
    fid = frozen_model()
    variant = CANDIDATES[fid]
    label = variant.get("label", "label_4w")
    sel, ctx, names = context_d()
    log(f"stage D: frozen model {fid} {variant}; label {label}; {len(names)} derived candidates")
    screen = stage_d_screen(sel, ctx, names, label)
    bundle = screen["bundle"]
    if not bundle:
        table = {"frozen_model": fid, "bundle": [], "admitted": False,
                 "rule": "no keepers survived the probe and dedup: nothing to exam"}
        (OUT / "stage_d.json").write_text(json.dumps(table, indent=1))
        log("stage D: no bundle; the frozen model stands on the base features")
        return table, []
    # the walk carries only the bundle's derived columns (memory, and the
    # model must not see the rest)
    ctx.pan = ctx.pan[[c for c in ctx.pan.columns if c in bundle or not c.startswith("d_")]]
    sweep(f"{fid}_x", full=True, variant={**variant, "extra": bundle}, sel_ctx=(sel, ctx),
          out=STAGE_D / "walk")
    return stage_d_exam(sel, ctx, fid, bundle)


# ----------------------------------------------------------------------------- 2016-2024 (owner's ask 2026-09-11)
def stage_d_arms() -> dict:
    """The two arms stage D leaves: the frozen model alone and with the
    admitted bundle (both read from the stage outputs, never typed)."""
    fid = frozen_model()
    d = json.loads((OUT / "stage_d.json").read_text())
    arms = {fid: dict(CANDIDATES[fid])}
    if d.get("admitted"):
        arms[f"{fid}_x"] = {**CANDIDATES[fid], "extra": list(d["bundle"])}
    return arms


def extend(vid: str):
    """One arm at K=4 on every rank week of 2016-01-01 -> the day before the
    holdout (the registration's grading window; the holdout is never read).
    Owner's ask 2026-09-11: assess the champion before stage E."""
    arms = stage_d_arms()
    variant = arms[vid]
    if variant.get("extra"):
        sel, ctx, _ = context_d()
        ctx.pan = ctx.pan[[c for c in ctx.pan.columns
                           if c in variant["extra"] or not c.startswith("d_")]]
    else:
        sel, ctx = context()
    sweep(vid, full=True, variant=variant, sel_ctx=(sel, ctx), out=OUT / "extend" / vid,
          window=EXTEND)


def assess():
    """The pre-holdout record at the deployed settings, 2006-2015 | 2016-2024
    | 2006-2024, for the stage D arms (K=4: their 2006-2015 walks joined to
    the 2016-2024 walks), the incumbent at K=4 (copies 1-4, same basis) and
    as deployed (K=16), and the S&P 500. Reads nothing after 2024-07-18."""
    from ops.k16_seed_spread import ensemble_rows
    sel, ctx = context()
    dep = deployed_settings()
    lo, mid, end = SELECT[0], EXTEND[0], HOLDOUT_START
    spans = {"2006-2015": (lo, mid), "2016-2024": (mid, end), "2006-2024": (lo, end)}
    arms = stage_d_arms()

    def walk(paths):
        frames = []
        for p in paths:
            df = pd.read_parquet(p)
            df["week"] = pd.to_datetime(df["week"])
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        if (df.week >= HOLDOUT_START).any():
            raise RuntimeError(f"{paths} reach the holdout")
        return df

    def record(preds, copies):
        rows, _ = ensemble_rows(sel, ctx, preds, copies)
        hold = rows.sort_values("week").reset_index(drop=True)
        s = sel.simulate(ctx, hold, "4w", dep["book"], dep["cap"], dep["stop"], dep["floor"])
        return {k: sel.metrics(s, a, b) for k, (a, b) in spans.items()}, s

    out, series = {}, {}
    for vid in arms:
        sel_path = (OUT / "full" / vid / "preds.parquet") if vid in CANDIDATES \
            else STAGE_D / "walk" / "preds.parquet"
        ext_path = OUT / "extend" / vid / "preds.parquet"
        if not ext_path.exists():
            log(f"assess: {vid} has no 2016-2024 walk yet ({ext_path})")
            continue
        preds = walk([sel_path, ext_path])
        n = preds.week.nunique()
        want = len([t for t in ctx.weeks if lo <= t < end])
        if n < want:
            raise RuntimeError(f"{vid}: {n} of {want} rank weeks — finish the walk")
        out[vid], series[vid] = record(preds, range(1, K_SAMPLE + 1))
    inc = walk([INCUMBENT["select"], INCUMBENT["extend"]])
    out["incumbent K=4"], series["incumbent K=4"] = record(inc, range(1, K_SAMPLE + 1))
    out["incumbent K=16 (deployed)"], series["incumbent K=16 (deployed)"] = record(inc, range(1, 17))
    spy = ctx.wret["SPY"]
    out["sp500"] = {k: sel.metrics(spy, a, b) for k, (a, b) in spans.items()}
    series["sp500"] = spy
    for vid in list(out):
        if vid == "sp500":
            continue
        d = (series[vid] - spy.reindex(series[vid].index)).dropna()
        out[vid]["paired_t_vs_sp500_2016_2024"] = round(paired_t(d[(d.index >= mid) & (d.index < end)]), 2)
        out[vid]["paired_t_vs_sp500_2006_2024"] = round(paired_t(d[(d.index >= lo) & (d.index < end)]), 2)
    out["settings"] = dep
    (OUT / "assess.json").write_text(json.dumps(out, indent=1, default=str))
    md = [f"Deployed settings {dep} at K=4 (the incumbent also at K=16 as deployed); $100 at the span's start.",
          "",
          "| model | 2006-2015: $100, %/yr, SR, DD | 2016-2024: $100, %/yr, SR, DD | 2006-2024: $100, %/yr, SR, DD | weekly t vs sp500 (2016-24 / 2006-24) |",
          "|---|---|---|---|---|"]
    for vid, r in out.items():
        if vid == "settings":
            continue
        cells = [f"${r[k]['terminal_100']:,.0f}, {r[k]['cagr_pct']:+.1f}%, {r[k]['sharpe']:.2f}, {r[k]['max_dd']:.0%}"
                 for k in spans]
        t = (f"{r['paired_t_vs_sp500_2016_2024']:+.2f} / {r['paired_t_vs_sp500_2006_2024']:+.2f}"
             if vid != "sp500" else "—")
        name = f"{vid} + engineered features*" if vid.endswith("_x") else vid
        md.append(f"| {name} | " + " | ".join(cells) + f" | {t} |")
    print("\n".join(md))
    log(f"assess -> {OUT / 'assess.json'}")
    return out, md


# ----------------------------------------------------------------------------- stage E (owner's go 2026-09-11)
K_FULL = 16
STAGE_E = OUT / "stage_e"
CI_SEED_DRAWS = 200
CI_HISTORY_DRAWS = 4000
CI_NESTED_PER_SEED = 20
CI_BLOCK_WEEKS = 8
FALSIFY_T = -2.0


def stage_e_arm(arm: str) -> dict:
    """The package's recipe, read from the stage outputs. The registration's
    default is the frozen model with the admitted bundle (`<frozen>_x`); the
    owner chose the frozen model alone ('Ok do stage e', 2026-09-11, to the
    recommendation made after the K=4 2016-2024 read) — a discretionary call,
    recorded as theirs in stage_e.json."""
    arms = stage_d_arms()
    if arm not in arms:
        raise SystemExit(f"stage E arm must be one of {list(arms)}, got {arm!r}")
    return arms[arm]


def walk_k16(vid: str, variant: dict, sel_ctx, src: Path, out: Path, window):
    """Copies 5..16 of a K=4 walk, every rank week of the window, merged with
    its copies 1-4 (same seeds, same recipe: copy c is bootstrap_seed=c, so
    the saved copies are the first four of the sixteen; the first week
    recomputes copy 1 and refuses if it differs). Checkpoints and resumes
    by week under a spec guard, like sweep."""
    sel, ctx = sel_ctx
    ctx.extra = list(variant.get("extra", []))
    lo, hi = window
    weeks = [t for t in ctx.weeks if lo <= t <= hi]
    bspec = json.loads((src / "spec.json").read_text())
    if bspec["recipe"] != json.loads(json.dumps(variant)) or bspec["k"] != K_SAMPLE:
        raise RuntimeError(f"{src} was walked under {bspec['recipe']} at K={bspec['k']}, "
                           f"not {variant} at K={K_SAMPLE}")
    base = pd.read_parquet(src / "preds.parquet")
    base["week"] = pd.to_datetime(base["week"])
    absent = [t for t in weeks if t not in set(base.week.unique())]
    if absent:
        raise RuntimeError(f"{src} lacks {len(absent)} weeks of {lo.date()} -> {hi.date()}")
    span = f"{lo.date()} -> {hi.date()}"
    _guard_spec(out, {"variant": vid, "recipe": json.loads(json.dumps(variant)),
                      "base_params": {k: str(v) for k, v in sel.MODEL_PARAMS.items()},
                      "k": K_FULL, "weeks": f"every week of {span}", "store": STORE,
                      "delist": ctx.delist_labels, "copies_1_4_from": str(src)})
    path = out / "preds.parquet"
    base_cols = [f"c{c}" for c in range(1, K_SAMPLE + 1)]
    frames, done = [], set()
    if path.exists():
        old = pd.read_parquet(path)
        old["week"] = pd.to_datetime(old["week"])
        frames.append(old)
        done = set(old["week"].unique())
    todo = [t for t in weeks if t not in done]
    log(f"walk_k16 {vid}: {len(weeks)} weeks of {span}, {len(todo)} to do, copies "
        f"{K_SAMPLE + 1}..{K_FULL} (1-4 from {src}); {variant}")
    t0, pending = time.time(), []
    for i, t in enumerate(todo, 1):
        g = base[base.week == t].set_index("ticker")[base_cols]
        if i == 1:
            p1 = variant_preds(sel, ctx, t, 1, variant)
            a, b = p1.align(g["c1"], join="inner")
            if len(a) != len(g) or not np.allclose(a.values, b.values, atol=1e-6):
                raise RuntimeError(f"copy 1 at {t.date()} does not reproduce {src}: "
                                   "the saved copies cannot be reused")
            log(f"  copy 1 at {t.date()} reproduces the saved walk ({len(a)} names)")
        cols = {}
        for c in range(K_SAMPLE + 1, K_FULL + 1):
            p = variant_preds(sel, ctx, t, c, variant)
            if p is not None:
                cols[f"c{c}"] = p
        df = g.join(pd.DataFrame(cols), how="outer")
        df.index.name = "ticker"
        df = df.reset_index()
        df.insert(0, "week", t)
        pending.append(df)
        if i % CHECKPOINT == 0 or i == len(todo):
            frames.extend(pending)
            pending = []
            pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
            el = time.time() - t0
            log(f"  {vid} {i}/{len(todo)} weeks, {el / i:.1f} s/week, "
                f"~{el / i * (len(todo) - i) / 60:.0f} min left")
    log(f"walk_k16 {vid}: done in {(time.time() - t0) / 60:.1f} min -> {path}")
    return path


def stage_e_context(variant: dict):
    if variant.get("extra"):
        sel, ctx, _ = context_d()
        ctx.pan = ctx.pan[[c for c in ctx.pan.columns
                           if c in variant["extra"] or not c.startswith("d_")]]
        return sel, ctx
    return context()


def stage_e_walks(arm: str):
    """The package at K=16: every week of 2006-2015 (from the stage C/D walk),
    then the strategy layers decided by the procedure's own code on that walk
    (stocks_ml.procedure.decide — the spec is NOT written; adoption is the
    owner's separate go), then every week of 2016-2024 (from the extend walk)."""
    from stocks_ml.procedure import decide
    variant = stage_e_arm(arm)
    sel, ctx = stage_e_context(variant)
    src_sel = (OUT / "full" / arm) if arm in CANDIDATES else STAGE_D / "walk"
    out = STAGE_E / arm
    walk_k16(arm, variant, (sel, ctx), src_sel, out / "select", SELECT)
    proc_path = out / "procedure.json"
    if not proc_path.exists():
        proc = decide(out / "select" / "preds.parquet", STORE, K_FULL, *SELECT, log=log)
        proc_path.write_text(json.dumps(proc, indent=1, default=str))
        log(f"stage E procedure on {arm}: decision {proc['decision']}; evidence "
            f"{json.dumps(proc['evidence'])}")
    walk_k16(arm, variant, (sel, ctx), OUT / "extend" / arm, out / "extend", EXTEND)


def _load(paths):
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df["week"] = pd.to_datetime(df["week"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if (df.week >= HOLDOUT_START).any():
        raise RuntimeError(f"{paths} reach the holdout")
    return df.sort_values(["week", "ticker"]).reset_index(drop=True)


def _sim(sel, ctx, preds, copies, st: dict) -> pd.Series:
    hold, _ = sel.ensemble_holdings(ctx, preds, copies)
    hold = hold.sort_values("week").reset_index(drop=True)
    return sel.simulate(ctx, hold, "4w", st["book"], st["cap"], st["stop"], st["floor"])


def _ci_stats(r: np.ndarray, spy: np.ndarray):
    """terminal $100, CAGR %, excess CAGR vs SPY % — compounded, same weeks."""
    yrs = len(r) / 52.18
    w, s = np.prod(1 + r), np.prod(1 + spy)
    return 100 * w, 100 * (w ** (1 / yrs) - 1), 100 * ((w / s) ** (1 / yrs) - 1)


def confidence(sel, ctx, preds, st: dict, spans: dict, seed_draws=CI_SEED_DRAWS) -> dict:
    """95% intervals as reports/clean_line_confidence.md: seed noise (K=16
    ensembles drawn by resampling the 16 copies with replacement, each
    re-simulated), history noise (circular block bootstrap of the weeks,
    strategy and SPY resampled on the same weeks) and the two nested."""
    rng = np.random.default_rng(20260911)
    point = _sim(sel, ctx, preds, range(1, K_FULL + 1), st)
    spy_all = ctx.wret["SPY"].reindex(point.index)
    draws, t0 = [], time.time()
    for b in range(seed_draws):
        copies = list(rng.integers(1, K_FULL + 1, size=K_FULL))
        draws.append(_sim(sel, ctx, preds, copies, st).reindex(point.index))
        if (b + 1) % 25 == 0:
            log(f"  ci: {b + 1}/{seed_draws} seed draws, {(time.time() - t0) / (b + 1):.1f} s/draw")
    A_all = pd.concat(draws, axis=1)

    def block_idx(n, B):
        nb = int(np.ceil(n / CI_BLOCK_WEEKS))
        starts = rng.integers(0, n, size=(B, nb))
        idx = (starts[:, :, None] + np.arange(CI_BLOCK_WEEKS)[None, None, :]).reshape(B, -1) % n
        return idx[:, :n]

    out = {}
    names = ("terminal_100", "cagr_pct", "excess_cagr_vs_sp500_pct")
    for w, (lo, hi) in spans.items():
        m = (point.index >= lo) & (point.index < hi)
        pt, spy, A = point[m].to_numpy(), spy_all[m].to_numpy(), A_all[m].to_numpy()
        ok = np.isfinite(pt) & np.isfinite(spy) & np.isfinite(A).all(axis=1)
        pt, spy, A = pt[ok], spy[ok], A[ok]
        n = len(pt)
        p = _ci_stats(pt, spy)
        seed = np.array([_ci_stats(A[:, j], spy) for j in range(A.shape[1])])
        hist = np.array([_ci_stats(pt[i], spy[i]) for i in block_idx(n, CI_HISTORY_DRAWS)])
        idx2 = block_idx(n, CI_NESTED_PER_SEED * A.shape[1])
        nest = np.array([_ci_stats(A[i, j % A.shape[1]], spy[i]) for j, i in enumerate(idx2)])
        ex = np.log1p(pt) - np.log1p(spy)
        q = lambda a, k: [round(float(v), 2) for v in np.percentile(a[:, k], [2.5, 97.5])]
        out[w] = {"weeks": int(n),
                  **{nm: {"point": round(float(p[k]), 2), "seed_95": q(seed, k),
                          "history_95": q(hist, k), "nested_95": q(nest, k)}
                     for k, nm in enumerate(names)},
                  "p_excess_positive_nested": round(float((nest[:, 2] > 0).mean()), 3),
                  "paired_weekly_log_excess_t_vs_sp500": round(float(ex.mean() / ex.std(ddof=1) * np.sqrt(n)), 2)}
    out["method"] = {"seed_draws": seed_draws, "history_draws": CI_HISTORY_DRAWS,
                     "nested_per_seed": CI_NESTED_PER_SEED, "block_weeks": CI_BLOCK_WEEKS}
    return out


def _row_vs_spy(sel, series: pd.Series, spy: pd.Series, spans: dict) -> dict:
    """metrics per span plus the paired weekly t vs sp500 — one table row."""
    row = {w: sel.metrics(series, a, b) for w, (a, b) in spans.items()}
    d = (series - spy.reindex(series.index)).dropna()
    row["paired_t_vs_sp500"] = {w: round(paired_t(d[(d.index >= a) & (d.index < b)]), 2)
                                for w, (a, b) in spans.items()}
    return row


def stage_e_deployed_row(arm: str, sel=None, ctx=None, preds=None) -> dict:
    """The package's own predictions run through the deployed spec's settings
    (the incumbent's book/floor/stop/cap). Next to the package at its
    procedure-decided settings, this separates what the model change is worth
    from what the settings change is worth. One simulation; no walk."""
    if sel is None:
        sel, ctx = context()
        out_dir = STAGE_E / arm
        preds = _load([out_dir / "select" / "preds.parquet", out_dir / "extend" / "preds.parquet"])
    lo, mid, end = SELECT[0], EXTEND[0], HOLDOUT_START
    spans = {"2006-2015": (lo, mid), "2016-2024": (mid, end), "2006-2024": (lo, end)}
    st = deployed_settings()
    series = _sim(sel, ctx, preds, range(1, K_FULL + 1), st)
    return _row_vs_spy(sel, series, ctx.wret["SPY"], spans)


def _stage_e_ledger_row(res: dict) -> dict:
    arm, table = res["arm"], res["table"]
    return {"kind": "clean_program_stage_e", "name": f"stage_e_{arm}_k{res['k']}",
            "pre_holdout_sharpe": table[arm]["2006-2024"]["sharpe"],
            "notes": json.dumps({"recipe": res["recipe"], "settings": res["package_settings"],
                                 "2016_2024": table[arm]["2016-2024"],
                                 "pre_holdout": table[arm]["2006-2024"],
                                 "falsification_t": res["falsification"]["2016-2024"]["t"],
                                 "verdict": res["falsification"]["verdict"],
                                 "leak": res["leak_audit"]["VERDICT"],
                                 "inflation": res["selection_inflation_cagr_pp"]})}


def stage_e_amend(arm: str):
    """Refresh the cheap parts of an existing stage_e.json without re-running
    the walks or the confidence draws: the package-at-deployed-settings row
    (one simulation) and the leak audit (seconds; re-run after the owner's
    2026-09-12 ruling made the identity check the only gate). The ledger
    row is upserted from the refreshed result."""
    from ops.leak_audit import audit_segments
    from stocks_ml.models.trials import record_trials
    path = OUT / "stage_e.json"
    res = json.loads(path.read_text())
    if res["arm"] != arm:
        raise SystemExit(f"{path} holds arm {res['arm']}, not {arm}")
    out_dir = STAGE_E / arm
    res["table"][f"{arm} at deployed settings"] = stage_e_deployed_row(arm)
    res["leak_audit"] = audit_segments(STORE, [str(out_dir / "select" / "preds.parquet"),
                                               str(out_dir / "extend" / "preds.parquet")])
    path.write_text(json.dumps(res, indent=1, default=str))
    record_trials([_stage_e_ledger_row(res)])
    md = stage_e_md(res)
    print("\n".join(md))
    log(f"stage E {arm}: amended (deployed-settings row, leak audit {res['leak_audit']['VERDICT']}) -> {path}")
    return res, md


def stage_e(arm: str, seed_draws: int = CI_SEED_DRAWS):
    """The one look. The package (K=16, its own procedure-decided settings)
    and the incumbent (K=16, the deployed spec's settings) on 2006-2015,
    2016-2024 and pre-holdout 2006-2024; the leak audit per segment; the
    pre-registered falsification test (paired weekly excess vs the incumbent
    on 2016-2024, t < -2 rejects); the selection inflation (the package's
    edge over the incumbent on 2006-2015, where it was chosen, next to its
    edge on 2016-2024, where it was not); 95% CIs; the ledger row. Nothing
    at or past the holdout is read; the deployed spec is not written."""
    from ops.leak_audit import audit_segments
    from stocks_ml.models.trials import record_trials
    variant = stage_e_arm(arm)
    out_dir = STAGE_E / arm
    paths = [out_dir / "select" / "preds.parquet", out_dir / "extend" / "preds.parquet"]
    proc = json.loads((out_dir / "procedure.json").read_text())
    dec = proc["decision"]
    pkg_st = dict(book=dec["book_size"], floor=dec["floor"], stop=dec["stop_loss"], cap=dec["sector_cap"])
    inc_st = deployed_settings()
    sel, ctx = context()
    lo, mid, end = SELECT[0], EXTEND[0], HOLDOUT_START
    spans = {"2006-2015": (lo, mid), "2016-2024": (mid, end), "2006-2024": (lo, end)}
    preds = _load(paths)
    want = [t for t in ctx.weeks if lo <= t < end]
    have = set(preds.week.unique())
    if any(t not in have for t in want):
        raise RuntimeError(f"stage E walk has {len(have)} of {len(want)} rank weeks — finish it")
    if not all(f"c{c}" in preds.columns for c in range(1, K_FULL + 1)):
        raise RuntimeError("stage E walk lacks copies")
    log(f"stage E {arm}: {len(have)} rank weeks at K={K_FULL}; package settings {pkg_st}; "
        f"incumbent settings {inc_st}")
    pkg = _sim(sel, ctx, preds, range(1, K_FULL + 1), pkg_st)
    inc = _sim(sel, ctx, _load([INCUMBENT["select"], INCUMBENT["extend"]]), range(1, K_FULL + 1), inc_st)
    spy = ctx.wret["SPY"]
    rows = {arm: pkg, "incumbent": inc, "sp500": spy}
    table = {name: {w: sel.metrics(s, a, b) for w, (a, b) in spans.items()} for name, s in rows.items()}
    for name in (arm, "incumbent"):
        d = (rows[name] - spy.reindex(rows[name].index)).dropna()
        table[name]["paired_t_vs_sp500"] = {w: round(paired_t(d[(d.index >= a) & (d.index < b)]), 2)
                                            for w, (a, b) in spans.items()}
    if pkg_st != inc_st:
        table[f"{arm} at deployed settings"] = stage_e_deployed_row(arm, sel, ctx, preds)
    d = (pkg - inc.reindex(pkg.index)).dropna()
    fals = {w: {"weeks": int(((d.index >= a) & (d.index < b)).sum()),
                "mean_weekly_excess_pct": round(float(d[(d.index >= a) & (d.index < b)].mean() * 100), 4),
                "t": round(paired_t(d[(d.index >= a) & (d.index < b)]), 2)}
            for w, (a, b) in spans.items()}
    t_2016 = fals["2016-2024"]["t"]
    verdict = "REJECTED" if t_2016 < FALSIFY_T else "not rejected"
    infl = {w: round(table[arm][w]["cagr_pct"] - table["incumbent"][w]["cagr_pct"], 2) for w in spans}
    log(f"stage E {arm}: falsification t {t_2016:+.2f} on 2016-2024 -> {verdict}; "
        f"edge over the incumbent %/yr: {infl}")
    leak = audit_segments(STORE, [str(p) for p in paths])
    log(f"stage E {arm}: leak audit {leak['VERDICT']} (worst retention {leak['worst_retention']})")
    ci = confidence(sel, ctx, preds, pkg_st, spans, seed_draws)
    res = {"arm": arm, "recipe": variant, "k": K_FULL,
           "owner_choice": "frozen model alone (owner, 2026-09-11, after the K=4 2016-2024 read; "
                           "the registration's default package carried the admitted bundle)"
                           if not variant.get("extra") else "registration default: frozen model + admitted bundle",
           "package_settings": pkg_st, "incumbent_settings": inc_st,
           "procedure": {k: proc[k] for k in ("decision", "evidence", "preds", "selection_window")},
           "rank_weeks": len(have), "table": table,
           "falsification": {"rule": f"paired weekly excess vs the incumbent on 2016-2024, t < {FALSIFY_T} rejects",
                             **fals, "verdict": verdict},
           "selection_inflation_cagr_pp": infl, "leak_audit": leak, "confidence": ci,
           "post_tax": "not computed: the account is a Roth IRA (post-tax = pre-tax); the taxable-account "
                       "engine is not in the repo"}
    STAGE_E.mkdir(parents=True, exist_ok=True)
    (OUT / "stage_e.json").write_text(json.dumps(res, indent=1, default=str))
    record_trials([_stage_e_ledger_row(res)])
    md = stage_e_md(res)
    print("\n".join(md))
    log(f"stage E -> {OUT / 'stage_e.json'}")
    return res, md


def leak_line(la: dict) -> str:
    """verdict (identity gate) plus the report-only factor numbers per segment."""
    segs = la.get("segments", {})
    parts = [f"{name}: identity {s.get('IDENTITY')}, score-vs-split-factor {s['spearman_score_vs_factor']:+.3f} "
             f"(t {s['spearman_t']:+.1f}), IC {s['ic']:+.4f} (t {s['ic_t']:+.1f}), retention {s['ic_retention']}"
             for name, s in segs.items()]
    return f"{la['VERDICT']} — " + "; ".join(parts) + "." if parts else f"{la['VERDICT']}."


def stage_e_md(res: dict) -> list[str]:
    arm, table, ci = res["arm"], res["table"], res["confidence"]
    ps, ins = res["package_settings"], res["incumbent_settings"]
    lab = lambda s: f"{s['book']}/{s['floor']}/cap{s['cap']}/stop {s['stop']}"
    md = [f"Package {arm} at K={res['k']} ({lab(ps)}, decided by the procedure on its 2006-2015 walk); "
          f"incumbent at K=16 ({lab(ins)}, the deployed spec). $100 at each span's start; "
          f"pre-tax (Roth).",
          "",
          "| model | pre-holdout 2006-2024: $100, %/yr | SR, DD | 2016-2024: $100, %/yr | SR, DD | 2006-2015: $100, %/yr | SR, DD | weekly t vs sp500 (06-24 / 16-24) |",
          "|---|---|---|---|---|---|---|---|"]
    names = [arm] + [n for n in table if n.endswith("at deployed settings")] + ["incumbent", "sp500"]
    for name in names:
        r = table[name]
        cells = []
        for w in ("2006-2024", "2016-2024", "2006-2015"):
            cells += [f"${r[w]['terminal_100']:,.0f}, {r[w]['cagr_pct']:+.1f}%",
                      f"{r[w]['sharpe']:.2f}, {r[w]['max_dd']:.0%}"]
        t = (f"{r['paired_t_vs_sp500']['2006-2024']:+.2f} / {r['paired_t_vs_sp500']['2016-2024']:+.2f}"
             if name != "sp500" else "—")
        label = name if not name.endswith("at deployed settings") else f"{arm} at the deployed settings ({lab(ins)})"
        md.append(f"| {label} | " + " | ".join(cells) + f" | {t} |")
    f = res["falsification"]
    md += ["", f"Falsification ({f['rule']}): 2016-2024 t {f['2016-2024']['t']:+.2f} on "
               f"{f['2016-2024']['weeks']} weeks -> **{f['verdict']}**. Edge over the incumbent, %/yr: "
               f"2006-2015 {res['selection_inflation_cagr_pp']['2006-2015']:+.2f} (selection window), "
               f"2016-2024 {res['selection_inflation_cagr_pp']['2016-2024']:+.2f} (the one look).",
           "Leak audit: " + leak_line(res["leak_audit"]),
           "", "| window | metric | point | seed-only 95% | history-only 95% | nested 95% |", "|---|---|---|---|---|---|"]
    for w in ("2016-2024", "2006-2024", "2006-2015"):
        for nm, label in (("excess_cagr_vs_sp500_pct", "excess CAGR vs sp500 %/yr"),
                          ("cagr_pct", "CAGR %/yr"), ("terminal_100", "terminal $100")):
            c = ci[w][nm]
            fmt = (lambda v: f"${v:,.0f}") if nm == "terminal_100" else (lambda v: f"{v:+.1f}")
            md.append(f"| {w} | {label} | {fmt(c['point'])} | {fmt(c['seed_95'][0])} … {fmt(c['seed_95'][1])} | "
                      f"{fmt(c['history_95'][0])} … {fmt(c['history_95'][1])} | "
                      f"{fmt(c['nested_95'][0])} … {fmt(c['nested_95'][1])} |")
        md.append(f"| {w} | P(excess > 0), nested | {ci[w]['p_excess_positive_nested']:.2f} | | | |")
    return md


def champion_series() -> pd.DataFrame:
    """The champion's weekly returns as the spec names it — its stage E walk
    (`procedure.preds.path` must be one) at its written settings, K copies —
    beside SPY, pre-holdout. One simulation, cached beside the walk as
    champion_weekly.parquet; delete the cache after any change to the spec."""
    spec = json.loads(Path("models/champion_spec.json").read_text())
    proc = spec["procedure"]
    walk = Path(proc["preds"]["path"]).parent.parent
    if walk.parent != STAGE_E:
        raise SystemExit(f"the spec's walk {walk} is not a stage E walk of this program")
    cache = walk / "champion_weekly.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        if df.attrs.get("decided_at") == proc["decided_at"]:
            return df
    st = deployed_settings()
    sel, ctx = context()
    preds = _load([walk / "select" / "preds.parquet", walk / "extend" / "preds.parquet"])
    log(f"champion series: {walk.name} at {st}, K={proc['k_copies']}, {preds.week.nunique()} rank weeks")
    r = _sim(sel, ctx, preds, range(1, proc["k_copies"] + 1), st)
    df = pd.DataFrame({"champion": r, "sp500": ctx.wret["SPY"].reindex(r.index)})
    df = df[(df.index >= SELECT[0]) & (df.index < HOLDOUT_START)]
    df.index.name = "week"
    df.attrs["decided_at"] = proc["decided_at"]
    df.to_parquet(cache)
    return df


def champion_chart():
    """Growth of $100, champion vs the S&P 500, on 2006-01 -> 2024-07 and on
    2016-01 -> 2024-07 (the years the label and window were not chosen on):
    NAV on a log axis, the ratio to the S&P, and both drawdowns."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, AQUA, SURF, INK, INK2 = "#2a78d6", "#1baf7a", "#fcfcfb", "#0b0b0b", "#52514e"
    spec = json.loads(Path("models/champion_spec.json").read_text())
    d, st = spec["procedure"]["decision"], spec["strategy"]
    who = (f"{LABELS_4W_SHORT[spec['horizon']['label']]} / {spec['training_window_years']}y / "
           f"top-{d['book_size']} / {d['floor']} / cap {d['sector_cap'] or 'none'}")
    df = champion_series()
    end = df.index[-1].strftime("%Y-%m")
    for lo, out in (("2006-01-01", "reports/champion_vs_sp500_2006_2024.png"),
                    ("2016-01-01", "reports/champion_vs_sp500_2016_2024.png")):
        w = df[df.index >= lo]
        navs = {k: (1 + w[k].fillna(0)).cumprod() * 100 for k in ("champion", "sp500")}
        names = {"champion": f"champion: {who}", "sp500": "sp500"}
        colors = {"champion": BLUE, "sp500": AQUA}
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(11.5, 9), dpi=150, sharex=True,
            gridspec_kw={"height_ratios": [3, 1.2, 1.0], "hspace": 0.12})
        fig.patch.set_facecolor(SURF)
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor(SURF)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=INK2, labelsize=9)
            ax.grid(True, axis="y", color="#e8e7e3", lw=0.8)
        for k, nav in navs.items():
            ax1.plot(nav.index, nav.values, color=colors[k], lw=2, label=names[k])
        ends = sorted(((np.log10(n.iloc[-1]), k) for k, n in navs.items()), reverse=True)
        ys = [y for y, _ in ends]
        for i in range(1, len(ys)):
            ys[i] = min(ys[i], ys[i - 1] - 0.07)
        for y, (_, k) in zip(ys, ends):
            ax1.annotate(f" ${navs[k].iloc[-1]:,.0f}", (navs[k].index[-1], 10 ** y),
                         color=colors[k], fontsize=9, fontweight="bold", va="center")
        ax1.set_yscale("log")
        lo_y = min(n.min() for n in navs.values()) * 0.9
        hi_y = max(n.max() for n in navs.values()) * 1.1
        ax1.set_ylim(lo_y, hi_y)
        ax1.set_yticks([t for t in (25, 50, 75, 100, 150, 200, 400, 800, 1600, 3200)
                        if lo_y <= t <= hi_y])
        ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
        ax1.set_title(f"r5 champion — growth of $100, {lo[:7]} -> {end} (pre-holdout, K={spec['procedure']['k_copies']}, "
                      f"next-open fills, 5 bp a side)", color=INK, fontsize=12, loc="left", pad=10)
        idx = w.index
        ax1.set_xlim(idx[0], idx[-1] + (idx[-1] - idx[0]) * 0.07)
        ratio = navs["champion"] / navs["sp500"]
        ax2.plot(ratio.index, ratio.values, color=BLUE, lw=2)
        ax2.axhline(1.0, color=INK2, lw=1, ls=":")
        ax2.set_ylabel("champion / sp500", color=INK2, fontsize=9)
        ax2.annotate(f" {ratio.iloc[-1]:.2f}x", (ratio.index[-1], ratio.iloc[-1]),
                     color=BLUE, fontsize=9, fontweight="bold", va="center")
        worst = {}
        for k, nav in navs.items():
            dd = nav / nav.cummax() - 1
            ax3.fill_between(dd.index, dd.values, 0, color=colors[k], alpha=0.25, lw=0)
            ax3.plot(dd.index, dd.values, color=colors[k], lw=1)
            worst[k] = dd.min()
        ax3.text(0.99, 0.06, "   ".join(f"worst {k} {v:.0%}" for k, v in worst.items()),
                 transform=ax3.transAxes, color=INK2, fontsize=8, ha="right", va="bottom")
        ax3.set_ylim(min(worst.values()) * 1.15, 0.02)
        ax3.set_ylabel("drawdown", color=INK2, fontsize=9)
        ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        fig.text(0.01, 0.005, f"walk {spec['procedure']['preds']['path']} (sha256 {spec['procedure']['preds']['sha256'][:12]}), "
                 f"settings written by stocks-ml procedure {spec['procedure']['decided_at']}; "
                 "the label and window were chosen on 2006-2015, 2016-2024 read once (stage E)",
                 color=INK2, fontsize=7.5)
        fig.savefig(out, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)
        log(f"chart -> {out}: champion ${navs['champion'].iloc[-1]:,.0f} vs sp500 ${navs['sp500'].iloc[-1]:,.0f}")


LABELS_4W_SHORT = {"label_4w": "4w label", "label_4w_sector": "sector-relative 4w label"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage_a")
    s = sub.add_parser("sweep")
    s.add_argument("--variant", choices=list(VARIANTS))
    s.add_argument("--all", action="store_true")
    sub.add_parser("stage_b")
    w = sub.add_parser("walk")
    w.add_argument("--candidate", choices=list(CANDIDATES))
    w.add_argument("--all", action="store_true")
    sub.add_parser("stage_c")
    sub.add_parser("stage_d")
    e = sub.add_parser("extend", help="one stage D arm at K=4 on every week of 2016-2024 (pre-holdout)")
    e.add_argument("--arm", required=True)
    sub.add_parser("assess")
    se = sub.add_parser("stage_e", help="the package at K=16: walks, procedure, the one look")
    se.add_argument("--arm", required=True)
    se.add_argument("--seed-draws", type=int, default=CI_SEED_DRAWS)
    sa = sub.add_parser("stage_e_amend", help="refresh stage_e.json: deployed-settings row + leak audit; no walks, no CIs")
    sa.add_argument("--arm", required=True)
    sub.add_parser("champion_chart", help="reports/champion_vs_sp500_*.png from the spec's stage E walk (needs matplotlib)")
    a = ap.parse_args()
    if a.cmd == "stage_a":
        stage_a()
    elif a.cmd == "sweep":
        for vid in (list(VARIANTS) if a.all else [a.variant]):
            sweep(vid)
    elif a.cmd == "walk":
        for vid in (list(CANDIDATES) if a.all else [a.candidate]):
            sweep(vid, full=True)
    elif a.cmd == "stage_b":
        stage_b()
    elif a.cmd == "stage_d":
        stage_d()
    elif a.cmd == "extend":
        extend(a.arm)
    elif a.cmd == "assess":
        assess()
    elif a.cmd == "stage_e":
        stage_e_walks(a.arm)
        stage_e(a.arm, a.seed_draws)
    elif a.cmd == "stage_e_amend":
        stage_e_amend(a.arm)
    elif a.cmd == "champion_chart":
        champion_chart()
    else:
        stage_c()
