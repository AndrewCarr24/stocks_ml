"""Re-grade the 4-week rebuild campaign on the fixed rank-date join.

Every graded number behind the r5 champion — population grid, training-window
sweep, book / floor / stop / cap layers, the $1,586 headline — came from the
campaign scripts kept in data/experiments/r4w_campaign_cache/, which joined a
week's picks to returns through `index.asof(t)`: a Thursday-dated pick (Friday
holiday) was paid for the week that had already ended and its real following
week was dropped (fixed in selection.py, commit dccca9a). Models and rankings
never saw the bug, so this converts the campaign's cached rankings into
`stocks-ml select` stage files on the fixed join, refitting only the Thursday
weeks whose rankings were never cached (deterministic, ~3 s each), and
re-grades every layer and the champion package old join vs fixed join.

  build    data/experiments/champion_2006_2024/          fixed join
           data/experiments/champion_2006_2024_oldjoin/  the campaign's own rows
  cascade  each layer's evidence on both joins, other layers held at the
           champion's choices, then the canonical `stocks-ml select --stage
           cascade` on the fixed files (frozen_config.json + ledger row)
  report   population grid, package, headline, selection inflation on both
           joins -> reports/rank_date_regrade.md, ledger rows amended in place
           (kind/name unchanged, old numbers kept in the notes), chart data
  charts   reports/r5_package_2006_2024.png, reports/r4w_vs_sp500_2021_2024.png
           (needs matplotlib: `uv run --with matplotlib python ops/regrade_campaign.py charts`)
           — the 2026-09 champion on the split-leaky basis; the files were removed
           2026-09-12 when the Stage E champion's charts replaced them
           (`ops/clean_program.py champion_chart` -> reports/champion_vs_sp500_*.png)

Run from the repo root: .venv/bin/python ops/regrade_campaign.py {build,cascade,report,charts,all}

Predates the fill-basis engine: its numbers, and reports/rank_date_regrade.md,
are on the close basis (fills at the rank date's close). The build, cascade and
report steps are not to be re-run under the current selection.py, whose
slice_row grades from the next open; ops/fill_basis_regrade.py archives these
files and re-grades them, reusing this module's helpers and `charts`.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from stocks_ml.selection import HOLDOUT_START

CACHE = Path("data/experiments/r4w_campaign_cache")
NEW = Path("data/experiments/champion_2006_2024")
OLD = Path("data/experiments/champion_2006_2024_oldjoin")
REPORT = Path("reports/rank_date_regrade.md")
LEDGER = Path("models/trials_ledger.json")
PRE_FIX = "dccca9a^"  # last commit with the .asof join
CHART_NOTE = "fixed rank-date join"  # the chart titles' basis (fill_basis_regrade overrides)
LO, HI = pd.Timestamp("2006-01-01"), HOLDOUT_START - pd.Timedelta(days=1)
NESTED_LO, NESTED_HI = pd.Timestamp("2016-01-01"), HOLDOUT_START
CHAMPION = dict(book=6, cap=2, stop=None, floor="70/30")
# (ledger key, label, simulate kwargs); the first five keys are r5_layers.py's
PKG_VARIANTS = (
    ("5y top-6 +stagger ", "5y top-6 +stagger (raw)", dict(cap=None, stop=None, floor="none")),
    ("+ stop-loss -25%", "+ stop-loss -25%", dict(cap=None, stop=-0.25, floor="none")),
    ("80/20 + trend ball", "80/20 + trend ballast", dict(cap=None, stop=None, floor="80/20")),
    ("60/40 + trend ball", "60/40 + trend ballast", dict(cap=None, stop=None, floor="60/40")),
    ("60/40 ballast + st", "60/40 ballast + stop", dict(cap=None, stop=-0.25, floor="60/40")),
    ("70/30 + trend ball", "70/30 + trend ballast", dict(cap=None, stop=None, floor="70/30")),
    ("70/30 + stop", "70/30 + stop", dict(cap=None, stop=-0.25, floor="70/30")),
    ("70/30 + cap2 champ", "70/30 + cap 2 (champion)", dict(cap=2, stop=None, floor="70/30")),
    ("70/30 + cap2 + stop", "70/30 + cap 2 + stop", dict(cap=2, stop=-0.25, floor="70/30")),
)
# r5_layers.py keyed its 2006-2024 and 2006-2012 rows both as r5pkg_2006_*; the
# 2006-2012 numbers won. Those keys keep their window; the full span gets its own.
# PKG_WINDOWS predates the one-convention rule (2026-09-09): "2025" credits the
# first holdout label. Kept for the archived report; do not copy into new graders.
PKG_WINDOWS = (("2006-2024", "2006", "2025"), ("2006", "2006", "2013"),
               ("2013", "2013", "2025"), ("2021", "2021", "2025"))
PY, STRIDE = {"1w": 52, "4w": 13}, {"1w": 1, "4w": 4}


# ---- engines ----
def new_engine():
    import stocks_ml.selection as sel
    return sel


def old_engine():
    """selection.py as it was before the fix, loaded beside the current one."""
    src = subprocess.run(["git", "show", f"{PRE_FIX}:src/stocks_ml/selection.py"],
                         capture_output=True, text=True, check=True).stdout
    OLD.mkdir(parents=True, exist_ok=True)
    path = OLD / "selection_prefix.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("selection_prefix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- campaign caches ----
def load_parts(pattern, key):
    fs = sorted(glob.glob(str(CACHE / pattern)))
    assert fs, f"no {pattern} under {CACHE}"
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["week"] = pd.to_datetime(df["week"])
    return df.drop_duplicates(key).sort_values("week").reset_index(drop=True)


def caches():
    sweep = pd.concat([load_parts("wsweep_part*.parquet", ["years", "week"]),
                       load_parts("wsweep2_part*.parquet", ["years", "week"])],
                      ignore_index=True)
    return {"grid": load_parts("era_grid_full.parquet", ["h", "week"]),
            "hold2": load_parts("era_hold_part*.parquet", ["week"]),
            "hold5": load_parts("era5_hold_part*.parquet", ["week"]),
            "sweep": sweep}


def is_friday(df):
    return df["week"].dt.weekday == 4


def graded_rows(eng, ctx, hold, horizon):
    """slice_row grades of cached rankings (week, tickers); top15 keeps the
    cached list verbatim, as the live job's rankings would."""
    rows = []
    for r in hold.itertuples():
        names = r.tickers.split(",")
        preds = pd.Series(np.arange(len(names), 0, -1.0), index=names)
        row = eng.slice_row(ctx, r.week, horizon, preds)
        if row is not None:
            row["top15"] = r.tickers
            rows.append(row)
    return pd.DataFrame(rows)


def refit_preds(sel, ctx, weeks, horizon, years, path):
    """Ensemble predictions for weeks with no cached ranking, cached per week."""
    store = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    for t in weeks:
        col = t.strftime("%Y-%m-%d")
        if col in store.columns:
            continue
        p = sel.ensemble_preds(ctx, t, horizon, years)
        p = pd.Series(dtype=float) if p is None else p
        store = pd.concat([store, p.rename(col)], axis=1)
        store.to_parquet(path)
        print(f"  refit {horizon}/{years}y {col}: {len(p)} names", flush=True)
    return {pd.Timestamp(c): store[c].dropna() for c in store.columns}


def refit_rows(sel, ctx, preds, horizon):
    rows = [sel.slice_row(ctx, t, horizon, p) for t, p in preds.items() if len(p) >= 20]
    return pd.DataFrame([r for r in rows if r is not None])


def campaign_check(rows, cached):
    """How far a re-grade from the cached 2y rankings sits from the campaign's
    own grid rows on Friday weeks (spy and rand_mean agree exactly; the top-k
    differ where the two runs ordered near-tied scores differently)."""
    m = rows.merge(cached, on="week", suffixes=("", "_c"))
    fri = m[is_friday(m)]
    d = max(float((fri[k] - fri[f"{k}_c"]).abs().max())
            for k in ("spy", "rand_mean", "top3", "top6", "top10"))
    diff = int(((fri["top6"] - fri["top6_c"]).abs() > 1e-9).sum())
    print(f"  4w grid from cached 2y rankings vs campaign rows: {len(fri)} Friday weeks, "
          f"top-6 differs on {diff}, max |diff| {d:.1e}", flush=True)


def build():
    sel, old = new_engine(), old_engine()
    ctx = sel.Ctx()
    c = caches()
    grid, sw = c["grid"], c["sweep"]
    sweep_weeks = sorted(sw["week"].unique())
    NEW.mkdir(parents=True, exist_ok=True)
    # fixed join: the champion's holdings re-joined from the cached rankings
    graded_rows(sel, ctx, c["hold5"], "4w").to_parquet(NEW / "holdings_4w_5y_s0.parquet")
    # grids: Friday rows are the campaign's own, Thursday weeks refit. (The 2y
    # rankings cached by era_pop_hold.py came from a separate run whose
    # near-tied scores order differently in ~8% of weeks: campaign_check.)
    campaign_check(graded_rows(sel, ctx, c["hold2"], "4w"), grid[grid.h == "4w"])
    for h in sel.HORIZONS:
        g = grid[grid.h == h].drop(columns=["era", "h"])
        p = refit_preds(sel, ctx, list(g["week"][~is_friday(g)]), h, sel.REF_WINDOW,
                        NEW / f"refit_preds_{h}_2y.parquet")
        g = (pd.concat([g[is_friday(g)], refit_rows(sel, ctx, p, h)], ignore_index=True)
             .sort_values("week"))
        g.to_parquet(NEW / f"grid_{h}_s0.parquet")
        if h == "4w":
            g4 = g
    # window sweep: 2y rows come from the re-joined grid, the rest as the 1w grid
    for y in sel.WINDOWS:
        if y == sel.REF_WINDOW:
            rows = g4[g4["week"].isin(sweep_weeks)]
        else:
            s = sw[sw.years == y].drop(columns=["years"])
            py = refit_preds(sel, ctx, list(s["week"][~is_friday(s)]), "4w", y,
                             NEW / f"refit_preds_4w_{y}y.parquet")
            rows = (pd.concat([s[is_friday(s)], refit_rows(sel, ctx, py, "4w")], ignore_index=True)
                    .sort_values("week"))
        rows.to_parquet(NEW / f"wsweep_{y}y_s0.parquet")
    # old join: the campaign's own rows, holdings through the old slice_row
    for h in sel.HORIZONS:
        grid[grid.h == h].drop(columns=["era", "h"]).to_parquet(OLD / f"grid_{h}_s0.parquet")
    for y in sel.WINDOWS:
        rows = (grid[(grid.h == "4w") & grid["week"].isin(sweep_weeks)].drop(columns=["era", "h"])
                if y == sel.REF_WINDOW else sw[sw.years == y].drop(columns=["years"]))
        rows.to_parquet(OLD / f"wsweep_{y}y_s0.parquet")
    graded_rows(old, ctx, c["hold5"], "4w").to_parquet(OLD / "holdings_4w_5y_s0.parquet")
    for out in (NEW, OLD):
        print(out.name, {p.name: len(pd.read_parquet(p)) for p in sorted(out.glob("*_s0.parquet"))})


# ---- layers ----
def layer_evidence(eng, ctx, out, lo, hi):
    """run_cascade's evidence per layer, each graded with the other layers held
    at the champion's choices (how the campaign decided them)."""
    grids = {h: eng._load(out, f"grid_{h}_s*.parquet") for h in eng.HORIZONS}
    horizon, hres = eng.decide_horizon(grids, lo, hi)
    sweeps = {y: eng._load(out, f"wsweep_{y}y_s*.parquet") for y in eng.WINDOWS}
    window, wres = eng.decide_window(sweeps, lo, hi)
    hold = eng._load(out, "holdings_4w_5y_s*.parquet")
    book, bres = eng.decide_book(hold, "4w", lo, hi)

    def sr(cap, stop, floor):
        return eng.sharpe(eng.simulate(ctx, hold, "4w", CHAMPION["book"], cap, stop, floor), lo, hi)

    fres = {f: sr(None, None, f) for f in eng.FLOORS}
    floor = max(fres, key=fres.get)
    sres = {str(s): sr(None, s, CHAMPION["floor"]) for s in (None, -0.25)}
    stop = None if sres["None"] >= sres["-0.25"] else -0.25
    cres = {str(c): sr(c, CHAMPION["stop"], CHAMPION["floor"]) for c in (None, 2)}
    cap = None if cres["None"] >= cres["2"] else 2
    return {"horizon": (horizon, hres), "window": (window, wres), "book": (book, bres),
            "floor": (floor, fres), "stop": (stop, sres), "cap": (cap, cres)}


def cascade():
    sel, old = new_engine(), old_engine()
    ctx = sel.Ctx()
    ev = {"old": layer_evidence(old, ctx, OLD, LO, HI), "new": layer_evidence(sel, ctx, NEW, LO, HI)}
    (NEW / "layer_evidence.json").write_text(json.dumps(ev, indent=1, default=str))
    print(layer_table(ev))
    changed = [k for k in ev["new"] if ev["new"][k][0] != ev["old"][k][0]]
    print("argmax changed:", changed or "none")
    if not changed:
        print("canonical cascade on the fixed files:")
        sel.run_select(str(LO.date()), str(HI.date()), name=NEW.name, stage="cascade")


def layer_table(ev):
    lines = ["| layer | menu | old join | fixed join | argmax old -> fixed | champion |",
             "|---|---|---|---|---|---|"]
    champ = {"horizon": "4w", "window": 5, "book": 6, "floor": "70/30", "stop": None, "cap": 2}
    unit = {"horizon": "top-6 %/yr net", "window": "top-6 vs rand %/yr", "book": "%/yr net",
            "floor": "Sharpe", "stop": "Sharpe", "cap": "Sharpe"}
    for k in ev["new"]:
        (ao, ro), (an, rn) = ev["old"][k], ev["new"][k]
        fmt = "{:.3f}" if unit[k] == "Sharpe" else "{:+.1f}"
        cell = lambda r: " / ".join(f"{m}: {fmt.format(v)}" for m, v in r.items())
        lines.append(f"| {k} ({unit[k]}) | {', '.join(map(str, rn))} | {cell(ro)} | {cell(rn)} "
                     f"| {ao} -> {an}{' **CHANGED**' if str(ao) != str(an) else ''} | {champ[k]} |")
    return "\n".join(lines)


# ---- report ----
def pop_table(grids):
    """era_pop_grade.py's population statistics per era x horizon x book."""
    out = {}
    for h, df in grids.items():
        df = df.assign(era=np.where(df["week"] < "2013-01-01", "2001-2012", "2013-2024"))
        for era, g in df.groupby("era"):
            g = g.sort_values("week")
            py, st = PY[h], STRIDE[h]
            sub = g.iloc[::st]
            yrs = len(sub) / (52 / st)
            for k in (3, 6, 10):
                d = g[f"top{k}"] - g["rand_mean"]
                blocks = d.groupby(np.arange(len(d)) // st).mean()
                nav = np.cumprod(1 + sub[f"top{k}"].values)
                out[(era, h, f"top{k}")] = dict(
                    vsrand=float(d.mean() * py * 100),
                    t=float(blocks.mean() / (blocks.std(ddof=1) / np.sqrt(len(blocks)))),
                    cmp=float((nav[-1] ** (1 / yrs) - 1) * 100),
                    sr=float(sub[f"top{k}"].mean() / sub[f"top{k}"].std(ddof=1) * np.sqrt(52 / st)),
                    dd=float((1 - nav / np.maximum.accumulate(nav)).max()), n=len(g))
            snav = np.cumprod(1 + sub["spy"].values)
            out[(era, h, "sp500")] = dict(
                cmp=float((snav[-1] ** (1 / yrs) - 1) * 100),
                sr=float(sub["spy"].mean() / sub["spy"].std(ddof=1) * np.sqrt(52 / st)),
                dd=float((1 - snav / np.maximum.accumulate(snav)).max()), n=len(g))
    return out


def pop_note(s):
    return (f"POPULATION vsRand {s['vsrand']:+.1f}%/yr (block-t {s['t']:+.2f}), "
            f"cmp {s['cmp']:+.1f}%/yr, SR {s['sr']:.2f}, DD {s['dd']:.0%}, n={s['n']}")


def pop_cell(s):
    if "vsrand" not in s:
        return f"cmp {s['cmp']:+.1f}, SR {s['sr']:.2f}, DD {s['dd']:.0%}"
    return f"vsRand {s['vsrand']:+.1f} (t {s['t']:.2f}), cmp {s['cmp']:+.1f}, SR {s['sr']:.2f}, DD {s['dd']:.0%}"


def grade_line(m):
    return (f"${m['terminal_100']:,.0f} ({m['cagr_pct']:+.1f}%/yr, "
            f"SR {m['sharpe']:.2f}, DD {m['max_dd']:.0%})")


def package_series(eng, ctx, hold):
    series = {key: eng.simulate(ctx, hold, "4w", CHAMPION["book"], kw["cap"], kw["stop"], kw["floor"])
              for key, _, kw in PKG_VARIANTS}
    series["sp500"] = ctx.wret["SPY"].reindex(series["70/30 + cap2 champ"].index)
    return series


def report():
    from stocks_ml.models.trials import record_trials
    sel, old = new_engine(), old_engine()
    ctx = sel.Ctx()
    ev = json.loads((NEW / "layer_evidence.json").read_text())
    ev = {j: {k: tuple(v) for k, v in d.items()} for j, d in ev.items()}
    md = ["# Rank-date re-grade of the champion campaign",
          "",
          f"Generated by `ops/regrade_campaign.py` on {pd.Timestamp.today().date()}. Commit dccca9a "
          "fixed the rank-date join in `selection.py`: a Thursday-dated pick (Friday holiday) was paid "
          "for the week that had already ended and its real following week was dropped. Models and "
          "rankings were never affected; every graded number was. Below, each campaign result is "
          "re-graded from the cached rankings on the fixed join (\"fixed\") beside the number as "
          "originally graded (\"old\"). Selection window 2006-01-01 -> 2024-07-18; the holdout "
          "(2024-07-19+) is untouched.",
          "", "## Cascade layers, 2006-2024 (each graded with the other layers at the champion's choices)", "",
          layer_table(ev), ""]
    # population grid
    grids = {j: {h: eng._load(out, f"grid_{h}_s*.parquet") for h in eng.HORIZONS}
             for j, eng, out in (("old", old, OLD), ("new", sel, NEW))}
    pop = {j: pop_table(g) for j, g in grids.items()}
    md += ["## Population grid (era x horizon x book; rates on all weeks, compounding on the "
           "non-overlapping subsequence)", "",
           "| era | h | book | old join | fixed join |", "|---|---|---|---|---|"]
    led = []
    for key in pop["new"]:
        era, h, k = key
        md.append(f"| {era} | {h} | {k} | {pop_cell(pop['old'][key])} | {pop_cell(pop['new'][key])} |")
        if k != "sp500":
            led.append({"kind": "era_grid", "name": f"pop_{era[:4]}_{h}_{k}",
                        "notes": f"{pop_note(pop['new'][key])} | before the rank-date fix (dccca9a): "
                                 f"{pop_note(pop['old'][key])}"})
    # package
    holds = {"old": old._load(OLD, "holdings_4w_5y_s*.parquet"), "new": sel._load(NEW, "holdings_4w_5y_s*.parquet")}
    series = {"old": package_series(old, ctx, holds["old"]), "new": package_series(sel, ctx, holds["new"])}
    labels = {key: label for key, label, _ in PKG_VARIANTS} | {"sp500": "sp500"}
    md += ["", "## Champion package (5y engine, top-6, 4 sleeves, costs included)", ""]
    for wkey, lo, hi in PKG_WINDOWS:
        span = f"{lo}-01 -> 2024-06 (pre-holdout)" if hi == "2025" else f"{lo}-01 -> {int(hi) - 1}-12"
        md += [f"### {span}", "", "| variant | old join | fixed join |", "|---|---|---|"]
        for key, label in labels.items():
            mo = old.metrics(series["old"][key], lo, hi)
            mn = sel.metrics(series["new"][key], lo, hi)
            md.append(f"| {label} | {grade_line(mo)} (n={mo['n_weeks']}) | {grade_line(mn)} (n={mn['n_weeks']}) |")
            led.append({"kind": "r4w_strategy", "name": f"r5pkg_{wkey}_{key}",
                        "notes": f"{span}: {grade_line(mn)} | before the rank-date fix (dccca9a): {grade_line(mo)}"})
        md.append("")
    # headline and selection inflation
    champ = {j: eng.metrics(series[j]["70/30 + cap2 champ"], LO, HOLDOUT_START)
             for j, eng in (("old", old), ("new", sel))}
    spy = {j: eng.metrics(series[j]["sp500"], LO, HOLDOUT_START)
           for j, eng in (("old", old), ("new", sel))}
    nested_old = {"terminal_100": 497.0, "cagr_pct": 21.4}
    from stocks_ml.models.trials import load_ledger
    nested_new = next(r for r in load_ledger(LEDGER)
                      if r["name"] == "nested2_verdict_amended")
    infl = {}
    for j, eng in (("old", old), ("new", sel)):
        m = eng.metrics(series[j]["70/30 + cap2 champ"], NESTED_LO, NESTED_HI)
        honest = nested_old if j == "old" else nested_new
        infl[j] = dict(champ=m, honest=honest, dollars=m["cagr_pct"] - honest["cagr_pct"])
    md += ["## Official record and selection inflation", "",
           f"- Champion 2006-01 -> 2024-06, old join: {grade_line(champ['old'])} vs SPY {grade_line(spy['old'])}",
           f"- Champion 2006-01 -> 2024-06, fixed join: {grade_line(champ['new'])} vs SPY {grade_line(spy['new'])}",
           f"- Champion 2016-01 -> 2024-06 (the nested test's frozen span), old join: "
           f"{grade_line(infl['old']['champ'])} vs honest nested procedure $497 (+21.4%/yr) -> "
           f"inflation {infl['old']['dollars']:+.1f}%/yr",
           f"- Champion 2016-01 -> 2024-06, fixed join: {grade_line(infl['new']['champ'])} vs honest "
           f"nested procedure ${nested_new['terminal_100']:,.0f} ({nested_new['cagr_pct']:+.1f}%/yr) -> "
           f"inflation {infl['new']['dollars']:+.1f}%/yr", ""]
    md += ["## Notes", "",
           "- The fully mechanical `stocks-ml select --stage cascade` on the fixed files freezes "
           "4w / 5y / top-3 / 60-40 / stop -25% / no cap (ledger `select_2006-01-01_2024-07-18`): "
           "`run_cascade` decides the book on the sampled 5-year sweep rows thinned by `iloc[::4]` "
           "(about 50 weeks) and grades the later layers on that book. The champion's top-6, 70/30, "
           "no stop and cap 2 were the owner's documented picks (ledger `champion_r5_*`); on the "
           "population holdings the book argmax is top-6 on both joins (table above).",
           "- Run-to-run noise: the campaign's cached 2y rankings (era_hold) and its 2y grid rows came "
           "from separate runs of the same deterministic config; in about 8% of weeks near-tied scores "
           "(median rank-1 to rank-4 gap 0.00005 vs 0.0026 overall) order differently, which moves the "
           "2y population top-3 number by about 1%/yr (`campaign_check` in the build step). The grids "
           "here therefore keep the campaign's own Friday rows; only Thursday weeks were refit.",
           "- Not re-graded: the 2y-engine layer rows (`layers_*`, `exit_*`) and the sampled-week "
           "`grid_*` rows, superseded by the population grid and the 5y package before the fix; the "
           "legacy weekly pipeline (tag `legacy-final`).", ""]
    led.append({"kind": "champion_change", "name": "champion_r5_7030_cap2_regraded",
                "terminal_100": champ["new"]["terminal_100"], "spy_terminal_100": spy["new"]["terminal_100"],
                "cagr_pct": champ["new"]["cagr_pct"], "spy_cagr_pct": spy["new"]["cagr_pct"],
                "sharpe": champ["new"]["sharpe"], "spy_sharpe": spy["new"]["sharpe"],
                "max_dd": champ["new"]["max_dd"], "spy_max_dd": spy["new"]["max_dd"],
                "n_weeks": champ["new"]["n_weeks"],
                "notes": f"Official record re-graded on the fixed rank-date join (dccca9a), config unchanged "
                         f"(4w/5y/top-6/70-30 trend ballast/no stop/cap 2): 2006-01 -> 2024-06 "
                         f"{grade_line(champ['new'])} vs SPY {grade_line(spy['new'])}; before the fix "
                         f"{grade_line(champ['old'])} vs SPY {grade_line(spy['old'])}. Every cascade layer's "
                         f"argmax is unchanged on the fixed join (reports/rank_date_regrade.md). Selection "
                         f"inflation vs the nested honest procedure 2016-2024: {infl['new']['dollars']:+.1f}%/yr "
                         f"on dollars (was {infl['old']['dollars']:+.1f})."})
    nested = dict(nested_new)
    nested["notes"] = nested["notes"].split(" Champion headline")[0] + (
        f" Champion headline re-graded on the same join: {grade_line(champ['new'])} vs SPY "
        f"{grade_line(spy['new'])} (was $1,586 vs $563); selection inflation {infl['new']['dollars']:+.1f}%/yr "
        f"on dollars (was +3.8).")
    nested.pop("date", None)
    led.append(nested)
    record_trials(led)
    REPORT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"ledger rows upserted: {len(led)}; report: {REPORT}")
    # chart data: navs and the ballast gate on the fixed join
    trace = []
    sel.simulate(ctx, holds["new"], "4w", CHAMPION["book"], CHAMPION["cap"], CHAMPION["stop"],
                 CHAMPION["floor"], trace=trace)
    gate = pd.Series({rec["nxt"]: rec["g"] for rec in trace})
    chart = pd.DataFrame({"champion": series["new"]["70/30 + cap2 champ"],
                          "book": series["new"]["5y top-6 +stagger "],
                          "sp500": series["new"]["sp500"], "gate": gate})
    chart.index.name = "week"
    chart.to_csv(NEW / "chart_series.csv")


# ---- charts ----
def charts():
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, AQUA, PLUM = "#2a78d6", "#eb6834", "#1baf7a", "#8e44ad"
    BLUE_L, BLUE_XL, GRAY = "#86b6ef", "#cde2fb", "#c3c2b7"
    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    df = pd.read_csv(NEW / "chart_series.csv", index_col="week", parse_dates=True)
    # ops/champion_bundle_regrade.py adds the champion-with-bundle line (2026-09-05); it runs
    # to 2024-07 where the clean lines stop at the campaign's 2024-06 cutoff, and a week a
    # line lacks is flat, so every line ends at its own record
    lines = ("champion", "book", "sp500") + (("champion_bundle",) if "champion_bundle" in df else ())
    end = "2024-07" if "champion_bundle" in df else "2024-06"
    for lo, out, title in (("2006-01-01", "reports/r5_package_2006_2024.png", f"2006-01 -> {end}"),
                           ("2021-01-01", "reports/r4w_vs_sp500_2021_2024.png", f"2021-01 -> {end}")):
        d = df[df.index >= lo]
        navs = {k: (1 + d[k].fillna(0)).cumprod() * 100 for k in lines}
        names = {"champion": "champion: 70/30 trend ballast, cap 2", "book": "raw top-6 book",
                 "sp500": "sp500", "champion_bundle": "champion + engineered features* (adopted 2026-09-05)"}
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(11.5, 9), dpi=150, sharex=True,
            gridspec_kw={"height_ratios": [3, 1.3, 0.9], "hspace": 0.12})
        fig.patch.set_facecolor(SURF)
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor(SURF)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(colors=INK2, labelsize=9)
            ax.grid(True, axis="y", color="#e8e7e3", lw=0.8)
        colors = dict(zip(navs, (BLUE, ORANGE, AQUA, PLUM)))
        for k, nav in navs.items():
            ax1.plot(nav.index, nav.values, color=colors[k], lw=2, label=names[k])
        # end labels, pushed apart in log space so they never overlap
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
        ax1.set_yticks([t for t in (25, 50, 75, 100, 125, 150, 200, 400, 800, 1600, 3200, 6400)
                        if lo_y <= t <= hi_y])
        ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
        ax1.set_title(f"r5 champion (4w / 5y / top-6) — growth of $100, {title} "
                      f"(pre-holdout, costs included, {CHART_NOTE})",
                      color=INK, fontsize=12, loc="left", pad=10)
        if "champion_bundle" in navs:
            ax1.text(0.0, -0.03, "* the bundle's candidate ideas were written after reading the whole "
                     "2006-2024 record; the screen's statistics were bounded to 2006-2015 (nested3_v2)",
                     transform=ax1.transAxes, color=INK2, fontsize=7.5, va="top")
        idx = d.index
        ax1.set_xlim(idx[0], idx[-1] + (idx[-1] - idx[0]) * 0.07)
        for k, col in (("champion", BLUE), ("book", ORANGE), ("champion_bundle", PLUM)):
            if k in navs:
                ratio = navs[k] / navs["sp500"]
                ax2.plot(ratio.index, ratio.values, color=col, lw=2)
        ax2.axhline(1.0, color=INK2, lw=1, ls=":")
        ax2.set_ylabel("vs sp500 (x)", color=INK2, fontsize=9)
        g = d["gate"].fillna(0).values
        ax3.stackplot(idx, [np.full(len(idx), 0.7), 0.3 * (1 - g), 0.3 * g],
                      colors=[BLUE_L, BLUE_XL, GRAY],
                      labels=["stock book (70%)", "ballast: SPY", "ballast: treasuries"],
                      edgecolor=SURF, linewidth=0.5)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("champion\nallocation", color=INK2, fontsize=8)
        ax3.legend(loc="lower left", frameon=False, fontsize=8, ncol=3, labelcolor=INK)
        fig.savefig(out, bbox_inches="tight", facecolor=SURF)
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "cascade", "report", "charts", "all"])
    step = ap.parse_args().step
    for name, fn in (("build", build), ("cascade", cascade), ("report", report), ("charts", charts)):
        if step in (name, "all"):
            print(f"== {name}", flush=True)
            fn()
