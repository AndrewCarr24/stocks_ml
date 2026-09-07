"""Re-grade the champion campaign on the fill basis — the live job's rules.

Since 2026-09 `selection.simulate` runs the live ledger (stocks_ml.ledger):
a rank date's target weights fill at the next session's open, every trade
pays 5 bp a side, rebalances under 0.5% of NAV are skipped; and `slice_row`
grades a pick from the first open after its rank date to the first open k
weeks later, the basis the training labels use. Every campaign number before
that was on the close basis: a pick bought at the rank date's close, the book
rebalanced free each week, 10 bp on the rotated sleeve. This script re-grades
the champion's stage files on the fill basis from the campaign's cached
rankings (window sweeps from fresh refits at the campaign's 275 sampled
weeks, every window from one run) and puts the two bases side by side.

  build    data/experiments/champion_2006_2024_closebasis/  the close-basis files, archived once
           data/experiments/champion_2006_2024/             holdings and window sweeps re-graded;
           sample_1w_2y.parquet is the 1w horizon at the same sweep weeks. The two
           population grids stay on the close basis (the horizon layer compares
           them with each other; a fill-basis 1w grid needs a full refit) — the
           sampled horizon check below covers that layer on the fill basis.
  cascade  each layer's evidence three ways: the campaign engine on the close
           basis (the record), the ledger with fills at the decision close, the
           ledger as deployed; the sampled same-rankings horizon and window checks;
           then the canonical `stocks-ml select --stage cascade` on the fill-basis
           files if no argmax moved
  report   package variants, headline, nested inflation, ledger rows amended in
           place (kind/name unchanged, close-basis numbers kept in the notes) ->
           reports/fill_basis_regrade.md, chart data
  charts   the two package PNGs (needs matplotlib:
           `uv run --with matplotlib python ops/fill_basis_regrade.py charts`)

Run from the repo root: .venv/bin/python ops/fill_basis_regrade.py {build,cascade,report,charts,all}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NEW = Path("data/experiments/champion_2006_2024")
CLOSE = Path("data/experiments/champion_2006_2024_closebasis")
REPORT = Path("reports/fill_basis_regrade.md")
LEDGER = Path("models/trials_ledger.json")
PRE_LEDGER = "bb3c523"  # last commit whose simulate filled at the rank date's close
LO, HI = pd.Timestamp("2006-01-01"), pd.Timestamp("2024-07-18")  # holdout starts 2024-07-19
NESTED_LO, NESTED_HI = pd.Timestamp("2016-01-01"), pd.Timestamp("2024-07-19")
CHAMPION = dict(book=6, cap=2, stop=None, floor="70/30")
BASES = ("record", "close", "fill")
BASIS_LABEL = {"record": "close basis (campaign engine)", "close": "ledger, fills at the decision close",
               "fill": "as deployed (next open, 5 bp a side)"}


def campaign():
    spec = importlib.util.spec_from_file_location("regrade_campaign", HERE / "regrade_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def new_engine():
    import stocks_ml.selection as sel
    return sel


def close_engine():
    """selection.py as it was before the ledger engine: fills at the rank
    date's close, free weekly rebalance, 10 bp on the rotated sleeve."""
    src = subprocess.run(["git", "show", f"{PRE_LEDGER}:src/stocks_ml/selection.py"],
                         capture_output=True, text=True, check=True).stdout
    CLOSE.mkdir(parents=True, exist_ok=True)
    path = CLOSE / "selection_closebasis.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("selection_closebasis", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fills_at_close(ctx):
    """The same world with every open replaced by the previous session's
    close: the ledger fills an order decided on d at d's close, and slice_row
    grades close to close."""
    sel = new_engine()
    return SimpleNamespace(smap=ctx.smap, members=ctx.members,
                           **sel.price_frames(ctx.closes, ctx.closes.shift(1)))


def sweep_weeks(rc):
    return sorted(rc.caches()["sweep"]["week"].unique())


def sampled_preds(rc, sel, ctx, weeks, horizon, years):
    """The cached refit at the sweep weeks (refits any week not cached yet)."""
    preds = rc.refit_preds(sel, ctx, list(weeks), horizon, years,
                           NEW / f"refit_preds_{horizon}_{years}y.parquet")
    wk = set(weeks)
    return {t: p for t, p in preds.items() if t in wk}


def top6_agreement(a, b):
    """Weeks where two gradings of the same weeks rank the same six names first."""
    m = a.merge(b, on="week", suffixes=("", "_b"))
    same = [set(x.split(",")[:6]) == set(y.split(",")[:6]) for x, y in zip(m["top15"], m["top15_b"])]
    return int(sum(same)), len(m)


# ---- build ----
def build():
    sel, rc = new_engine(), campaign()
    ctx = sel.Ctx()
    if not (CLOSE / "holdings_4w_5y_s0.parquet").exists():   # archive once
        CLOSE.mkdir(parents=True, exist_ok=True)
        for f in NEW.iterdir():
            if f.is_file():
                shutil.copy2(f, CLOSE / f.name)
        print(f"archived {NEW} -> {CLOSE}")
    c = rc.caches()
    weeks = sweep_weeks(rc)
    # the archive must be what the current slice_row gives on the close basis
    atc = fills_at_close(ctx)
    old = sel._load(CLOSE, "holdings_4w_5y_s*.parquet")
    chk = rc.graded_rows(sel, atc, c["hold5"], "4w").merge(old, on="week", suffixes=("", "_a"))
    d = max(float((chk[k] - chk[f"{k}_a"]).abs().max()) for k in ("spy", "top3", "top6", "top10"))
    assert d < 1e-9, f"close-basis slice_row does not reproduce the archived holdings ({d:.1e})"
    # rand_mean can differ where a member's first print falls within the four
    # sessions next_open looks ahead (FTI, 2017-01-13): the close frame had no
    # return for it, the fill frame does
    dr = (chk["rand_mean"] - chk["rand_mean_a"]).abs()
    assert dr.max() < 1e-3, f"rand_mean off by {dr.max():.1e}"
    print(f"archive check: {len(chk)} holdings weeks reproduced on the close basis (spy, top-k exact; "
          f"rand_mean differs on {int((dr > 1e-9).sum())} weeks by at most {dr.max():.1e})")
    # holdings: the champion's 5y rankings verbatim, graded from the next open
    hold = rc.graded_rows(sel, ctx, c["hold5"], "4w")
    hold.to_parquet(NEW / "holdings_4w_5y_s0.parquet")
    # window sweeps: one fresh refit per window at the campaign's sweep weeks
    for y in sel.WINDOWS:
        rows = rc.refit_rows(sel, ctx, sampled_preds(rc, sel, ctx, weeks, "4w", y), "4w")
        rows.to_parquet(NEW / f"wsweep_{y}y_s0.parquet")
        if y == 2:
            n, m = top6_agreement(rows, rc.graded_rows(sel, ctx, c["hold2"], "4w"))
            print(f"  4w/2y refit vs the campaign's cached 2y rankings: same top-6 on {n} of {m} weeks")
        if y == 5:
            n, m = top6_agreement(rows, hold)
            print(f"  4w/5y refit vs the champion's cached 5y rankings: same top-6 on {n} of {m} weeks")
    rows = rc.refit_rows(sel, ctx, sampled_preds(rc, sel, ctx, weeks, "1w", 2), "1w")
    rows.to_parquet(NEW / "sample_1w_2y.parquet")
    for out in (CLOSE, NEW):
        print(out.name, {p.name: len(pd.read_parquet(p))
                         for p in sorted(out.glob("*.parquet")) if "refit_preds" not in p.name})


# ---- layers ----
def horizon_check(sel, rc, ctx, atc, weeks):
    """1w vs 4w from the same refits (2y window) at the sweep weeks, each
    horizon graded on both bases: top-6 net of COST per holding period as an
    arithmetic mean, annualised (not the compounded number of the layer
    table), the edge over the random member, and the paired fill - close
    difference on the same names and weeks with its t-statistic. `level_se`
    is the standard error of the sampled level; the population grid (close
    basis) at all weeks and at the sampled ones puts that level in scale."""
    rows = {}
    for h in sel.HORIZONS:
        preds = sampled_preds(rc, sel, ctx, weeks, h, sel.REF_WINDOW)
        for basis, c in (("close", atc), ("fill", ctx)):
            rows[(h, basis)] = rc.refit_rows(sel, c, preds, h)
    common = set.intersection(*(set(r["week"]) for r in rows.values()))
    common = {t for t in common if LO <= t <= HI}
    res, paired, pop = {}, {}, {}
    fills = sel.next_open(ctx.opens, ctx.cw.index)
    for h in sel.HORIZONS:
        per_year = 52 / sel.HORIZONS[h]["kweeks"]
        ann = lambda x: float(np.mean(x) * per_year * 100)  # noqa: E731
        g = {b: rows[(h, b)][rows[(h, b)]["week"].isin(common)] for b in ("close", "fill")}
        for b, r in g.items():
            res[f"{h}/{b}"] = {"net": ann(r["top6"] - sel.COST), "vs_rand": ann(r["top6"] - r["rand_mean"])}
        m = g["fill"].merge(g["close"], on="week", suffixes=("_f", "_c"))
        d = (m["top6_f"] - m["top6_c"]).values
        paired[h] = {"diff": ann(d), "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
                     "level_se": ann(m["top6_c"].std(ddof=1) / np.sqrt(len(m)))}
        # the ingredient: the weekend gap (next open / rank-date close - 1) of
        # the six names at entry, credited by the close basis, and at exit
        gaps = {"in": [], "out": []}
        for r in m.itertuples():
            wk = sel.week_slot(ctx.cw.index, r.week)
            i = ctx.cw.index.get_loc(wk)
            names = r.top15_f.split(",")[:6]
            for side, w in (("in", wk), ("out", ctx.cw.index[i + sel.HORIZONS[h]["kweeks"]])):
                gaps[side].append(np.nanmean([fills.at[w, n] / ctx.cw.at[w, n] - 1 for n in names]))
        paired[h].update({f"gap_{k}_bp": float(np.nanmean(v) * 1e4) for k, v in gaps.items()})
        grid = sel._load(CLOSE, f"grid_{h}_s*.parquet")
        grid = grid[(grid["week"] >= LO) & (grid["week"] <= HI)]
        pop[h] = {"n": int(len(grid)), "net": ann(grid["top6"] - sel.COST),
                  "net_sampled": ann(grid[grid["week"].isin(common)]["top6"] - sel.COST)}
    return {"n_weeks": len(common), "res": res, "paired": paired, "population": pop}


def window_check(sel, rc, ctx, atc, weeks):
    """decide_window on the same five refits, both bases."""
    out = {}
    for basis, c in (("close", atc), ("fill", ctx)):
        sweeps = {y: rc.refit_rows(sel, c, sampled_preds(rc, sel, ctx, weeks, "4w", y), "4w")
                  for y in sel.WINDOWS}
        out[basis] = sel.decide_window(sweeps, LO, HI)
    return out


def cascade():
    sel, rc, old = new_engine(), campaign(), close_engine()
    ctx = sel.Ctx()
    atc = fills_at_close(ctx)
    weeks = sweep_weeks(rc)
    ev = {"record": rc.layer_evidence(old, ctx, CLOSE, LO, HI),
          "close": rc.layer_evidence(sel, atc, CLOSE, LO, HI),
          "fill": rc.layer_evidence(sel, ctx, NEW, LO, HI)}
    archived = json.loads((CLOSE / "layer_evidence.json").read_text())["new"]
    for k, (arg, res) in ev["record"].items():
        a_arg, a_res = archived[k]
        assert str(arg) == str(a_arg) and all(abs(res[m] - a_res[str(m)]) < 1e-6 for m in res), \
            f"the campaign engine no longer reproduces the archived {k} evidence"
    checks = {"horizon": horizon_check(sel, rc, ctx, atc, weeks),
              "window": window_check(sel, rc, ctx, atc, weeks)}
    changed = [k for k in ev["fill"] if str(ev["fill"][k][0]) != str(ev["record"][k][0])]
    (NEW / "layer_evidence.json").write_text(json.dumps(
        {"layers": ev, "checks": checks, "changed": changed}, indent=1, default=str))
    print(layer_table(ev))
    print(check_table(checks))
    print("argmax changed, record -> as deployed:", changed or "none")
    if not changed:
        print("canonical cascade on the fill-basis files:")
        sel.run_select(str(LO.date()), str(HI.date()), name=NEW.name, stage="cascade")


def layer_table(ev):
    lines = ["| layer | menu | " + " | ".join(BASIS_LABEL[b] for b in BASES) +
             " | argmax record -> deployed | champion |", "|---|---|---|---|---|---|---|"]
    champ = {"horizon": "4w", "window": 5, "book": 6, "floor": "70/30", "stop": None, "cap": 2}
    unit = {"horizon": "top-6 %/yr net", "window": "top-6 vs rand %/yr", "book": "%/yr net",
            "floor": "Sharpe", "stop": "Sharpe", "cap": "Sharpe"}
    for k in ev["fill"]:
        fmt = "{:.3f}" if unit[k] == "Sharpe" else "{:+.1f}"
        cell = lambda r: " / ".join(f"{m}: {fmt.format(v)}" for m, v in r.items())  # noqa: E731
        a0, a1 = ev["record"][k][0], ev["fill"][k][0]
        cells = " | ".join(cell(ev[b][k][1]) for b in BASES)
        lines.append(f"| {k} ({unit[k]}) | {', '.join(map(str, ev['fill'][k][1]))} | {cells} "
                     f"| {a0} -> {a1}{' **CHANGED**' if str(a0) != str(a1) else ''} | {champ[k]} |")
    return "\n".join(lines)


def check_table(checks):
    h, w = checks["horizon"], checks["window"]
    lines = [f"| same refits, {h['n_weeks']} sampled weeks | close basis | fill basis | fill - close, paired |",
             "|---|---|---|---|"]
    for hz in ("1w", "4w"):
        c, f, p = h["res"][f"{hz}/close"], h["res"][f"{hz}/fill"], h["paired"][hz]
        lines.append(f"| {hz} top-6, 2y window: net %/yr (vs random) | {c['net']:+.1f} ({c['vs_rand']:+.1f}) "
                     f"| {f['net']:+.1f} ({f['vs_rand']:+.1f}) | {p['diff']:+.1f}%/yr, t {p['t']:+.1f} "
                     f"(SE of a sampled level {p['level_se']:.0f}) |")
    for hz in ("1w", "4w"):
        g = h["population"][hz]
        lines.append(f"| {hz} population grid, close basis, {g['n']} weeks: net %/yr all weeks / sampled weeks "
                     f"| {g['net']:+.1f} / {g['net_sampled']:+.1f} | | |")
    cell = lambda r: " / ".join(f"{y}: {v:+.1f}" for y, v in r.items())  # noqa: E731
    lines.append(f"| window, top-6 vs random %/yr (argmax) | {cell(w['close'][1])} ({w['close'][0]}y) "
                 f"| {cell(w['fill'][1])} ({w['fill'][0]}y) | |")
    return "\n".join(lines)


# ---- report ----
def _pre_fix(notes):
    """The 'before the rank-date fix' grade carried in a regrade_campaign note."""
    key = "before the rank-date fix (dccca9a): "
    return notes.split(key, 1)[1] if key in notes else None


def report():
    from stocks_ml.models.trials import load_ledger, record_trials
    sel, rc, old = new_engine(), campaign(), close_engine()
    ctx = sel.Ctx()
    saved = json.loads((NEW / "layer_evidence.json").read_text())
    ev = {b: {k: tuple(v) for k, v in d.items()} for b, d in saved["layers"].items()}
    checks = saved["checks"]
    checks["window"] = {b: tuple(v) for b, v in checks["window"].items()}
    frozen = None if saved["changed"] else json.loads((NEW / "frozen_config.json").read_text())
    ledger = {r["name"]: r for r in load_ledger(LEDGER)}
    holds = {"record": old._load(CLOSE, "holdings_4w_5y_s*.parquet"),
             "fill": sel._load(NEW, "holdings_4w_5y_s*.parquet")}
    series = {"record": rc.package_series(old, ctx, holds["record"]),
              "fill": rc.package_series(sel, ctx, holds["fill"])}
    labels = {key: label for key, label, _ in rc.PKG_VARIANTS} | {"sp500": "sp500"}
    gl = rc.grade_line

    md = ["# Fill-basis re-grade of the champion campaign", "",
          f"Generated by `ops/fill_basis_regrade.py` on {pd.Timestamp.today().date()}. "
          "`selection.simulate` now runs the live job's ledger (`stocks_ml.ledger`): a rank "
          "date's target weights fill at the next session's open, 5 bp a side on every trade, "
          "rebalances under 0.5% of NAV skipped; `slice_row` grades a pick from the first open "
          "after its rank date to the first open k weeks later, the training labels' basis. "
          "Every campaign number before this was on the close basis (a pick bought at the rank "
          "date's close, the book rebalanced free each week, 10 bp on the rotated sleeve). "
          "Below, each result on the fill basis beside the close-basis number of record "
          "(reports/rank_date_regrade.md). Selection window 2006-01-01 -> 2024-07-18; the "
          "holdout (2024-07-19+) is untouched.", "",
          "## Cascade layers, 2006-2024 (each graded with the other layers at the champion's choices)", "",
          layer_table(ev), "",
          "The middle column runs the ledger with every fill priced at the decision close, so its "
          "floor, stop and cap cells isolate the ledger's accounting (5 bp a side, dust threshold) "
          "from fill timing; its horizon, window and book cells are the close-basis rows again "
          "(those layers do not go through the ledger). "
          "The horizon layer's population grids stay on the close basis (the campaign's grid cache "
          "holds no ticker lists; a fill-basis grid needs a full refit); the book layer's holdings "
          "and the window sweeps are on the fill basis. Sampled check of the horizon and window "
          "layers from the same refits, both bases: arithmetic means over the sampled weeks, "
          "annualised, so not comparable with the compounded layer numbers above; the last column "
          "is the fill - close difference on the same names and weeks.", "",
          check_table(checks), ""]
    md += ["## Champion package (5y engine, top-6, 4 sleeves, costs included)", ""]
    led = []
    for wkey, lo, hi in rc.PKG_WINDOWS:
        span = f"{lo}-01 -> 2024-06 (pre-holdout)" if hi == "2025" else f"{lo}-01 -> {int(hi) - 1}-12"
        md += [f"### {span}", "", "| variant | close basis | as deployed |", "|---|---|---|"]
        for key, label in labels.items():
            mr = old.metrics(series["record"][key], lo, hi)
            mf = sel.metrics(series["fill"][key], lo, hi)
            md.append(f"| {label} | {gl(mr)} (n={mr['n_weeks']}) | {gl(mf)} (n={mf['n_weeks']}) |")
            name = f"r5pkg_{wkey}_{key}"
            pre = _pre_fix(ledger[name]["notes"]) if name in ledger else None
            led.append({"kind": "r4w_strategy", "name": name,
                        "notes": f"{span}: {gl(mf)} as deployed (fills at the next open, 5 bp a side) "
                                 f"| close basis, fixed join: {gl(mr)}"
                                 + (f" | before the rank-date fix (dccca9a): {pre}" if pre else "")})
        md.append("")
    # headline and selection inflation against the nested honest procedure
    champ = {b: eng.metrics(series[b]["70/30 + cap2 champ"], LO, pd.Timestamp("2024-07-19"))
             for b, eng in (("record", old), ("fill", sel))}
    spy = sel.metrics(series["fill"]["sp500"], LO, pd.Timestamp("2024-07-19"))
    nested_old = ledger["nested2_verdict_amended"]
    nested_new, nested_spy = nested_replay()
    infl = {}
    for b, eng in (("record", old), ("fill", sel)):
        m = eng.metrics(series[b]["70/30 + cap2 champ"], NESTED_LO, NESTED_HI)
        honest = nested_old if b == "record" else nested_new
        infl[b] = dict(champ=m, honest=honest, dollars=m["cagr_pct"] - honest["cagr_pct"])
    md += ["## Official record and selection inflation", "",
           f"- Champion 2006-01 -> 2024-06, as deployed: {gl(champ['fill'])} vs SPY {gl(spy)}",
           f"- Champion 2006-01 -> 2024-06, close basis: {gl(champ['record'])} vs SPY {gl(spy)}",
           f"- Nested honest procedure 2016-01 -> 2024-06 (4w/5y/top-10/60-40/no stop/cap 2, "
           f"selected on 2006-2015), as deployed: {gl(nested_new)} vs SPY {gl(nested_spy)}; "
           f"close basis: {gl(nested_old)}",
           f"- Champion 2016-01 -> 2024-06, as deployed: {gl(infl['fill']['champ'])} -> selection "
           f"inflation {infl['fill']['dollars']:+.1f}%/yr on dollars (close basis "
           f"{infl['record']['dollars']:+.1f})", ""]
    mech = (f"freezes {frozen['horizon']} / {frozen['train_years']}y / top-{frozen['book']} / "
            f"{frozen['floor']} / stop {frozen['stop']} / cap {frozen['cap']} (ledger "
            f"`select_2006-01-01_2024-07-18`; on the close basis it froze 4w / 5y / top-3 / 60-40 / "
            f"stop -25% / no cap)" if frozen else
            f"was not run: the argmax moved on {', '.join(saved['changed'])} (table above)")
    md += ["## Notes", "",
           f"- The fully mechanical `stocks-ml select --stage cascade` on the fill-basis files {mech}. "
           "`run_cascade` decides "
           "the book on the sampled 5-year sweep rows thinned by `iloc[::4]` and grades the later "
           "layers on that book; the champion's top-6, 70/30, no stop and cap 2 were the owner's "
           "documented picks, and on the population holdings the book argmax is top-6 on both bases.",
           horizon_note(checks, ev),
           "- The window sweeps are fresh refits at the campaign's sampled weeks, one run for all "
           "five windows; the campaign's own rows came from separate runs (near-tied scores order "
           "differently in a few percent of weeks). The holdings keep the champion's cached rankings "
           "verbatim, so the book, floor, stop and cap layers and the package differ from the close "
           "basis only by the fill rules.",
           "- Not re-graded: the population grid (`pop_*` ledger rows, close basis), the 2y-engine "
           "layer rows (`layers_*`, `exit_*`) and the legacy weekly pipeline (tag `legacy-final`).", ""]
    led.append({"kind": "champion_change", "name": "champion_r5_7030_cap2_regraded",
                "terminal_100": champ["fill"]["terminal_100"], "spy_terminal_100": spy["terminal_100"],
                "cagr_pct": champ["fill"]["cagr_pct"], "spy_cagr_pct": spy["cagr_pct"],
                "sharpe": champ["fill"]["sharpe"], "spy_sharpe": spy["sharpe"],
                "max_dd": champ["fill"]["max_dd"], "spy_max_dd": spy["max_dd"],
                "n_weeks": champ["fill"]["n_weeks"],
                "notes": f"Official record as deployed (selection.simulate on the live ledger rules: fills "
                         f"at the next open, 5 bp a side, 0.5%-of-NAV dust threshold), config unchanged "
                         f"(4w/5y/top-6/70-30 trend ballast/no stop/cap 2): 2006-01 -> 2024-06 "
                         f"{gl(champ['fill'])} vs SPY {gl(spy)}; close basis (fixed rank-date join, "
                         f"dccca9a) {gl(champ['record'])}; before the fix $1,586 (+16.8%/yr, SR 0.73, "
                         f"DD 58%). " + ("Every cascade layer's argmax is unchanged on the fill basis"
                                            if not saved["changed"] else
                                            f"Argmax moved on {', '.join(saved['changed'])}")
                         + f" (reports/fill_basis_regrade.md). Selection inflation vs the nested honest "
                         f"procedure 2016-2024: {infl['fill']['dollars']:+.1f}%/yr on dollars (close "
                         f"basis {infl['record']['dollars']:+.1f})."})
    nested = dict(nested_old)
    nested.pop("date", None)
    nested.update({k: nested_new[k] for k in ("terminal_100", "cagr_pct", "sharpe", "max_dd", "n_weeks")})
    nested.update({"spy_terminal_100": nested_spy["terminal_100"], "spy_cagr_pct": nested_spy["cagr_pct"],
                   "spy_sharpe": nested_spy["sharpe"], "spy_max_dd": nested_spy["max_dd"]})
    nested["notes"] = (f"Nested honest procedure (4w/5y/top-10/60-40/no stop/cap 2, selected on 2006-2015) "
                       f"as deployed, 2016-01 -> 2024-06: {gl(nested_new)} vs SPY {gl(nested_spy)}; close "
                       f"basis {gl(nested_old)} vs SPY {gl({k.replace('spy_', ''): v for k, v in nested_old.items() if k.startswith('spy_')})}. "
                       f"Champion headline on the same basis: {gl(champ['fill'])} vs SPY {gl(spy)}; "
                       f"selection inflation {infl['fill']['dollars']:+.1f}%/yr on dollars (close basis "
                       f"{infl['record']['dollars']:+.1f}). Earlier: " + nested_old["notes"].split(" Champion headline")[0])
    led.append(nested)
    record_trials(led)
    REPORT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"ledger rows upserted: {len(led)}; report: {REPORT}")
    trace = []
    sel.simulate(ctx, holds["fill"], "4w", CHAMPION["book"], CHAMPION["cap"], CHAMPION["stop"],
                 CHAMPION["floor"], trace=trace)
    gate = pd.Series({rec["nxt"]: rec["g"] for rec in trace})
    chart = pd.DataFrame({"champion": series["fill"]["70/30 + cap2 champ"],
                          "book": series["fill"]["5y top-6 +stagger "],
                          "sp500": series["fill"]["sp500"], "gate": gate})
    chart.index.name = "week"
    chart.to_csv(NEW / "chart_series.csv")


def horizon_note(checks, ev):
    """What the sampled horizon check resolves, from the saved numbers."""
    p, pop = checks["horizon"]["paired"], checks["horizon"]["population"]
    lead = ev["record"]["horizon"][1]
    return (f"- The sampled horizon check does not resolve the basis effect on 1w: fill - close "
            f"{p['1w']['diff']:+.1f}%/yr paired, t {p['1w']['t']:+.1f} (4w: {p['4w']['diff']:+.1f}, "
            f"t {p['4w']['t']:+.1f}), and a sampled 1w level carries an SE of {p['1w']['level_se']:.0f}%/yr "
            f"(the population 1w grid's arithmetic net is {pop['1w']['net']:+.1f}%/yr over "
            f"{pop['1w']['n']} weeks, {pop['1w']['net_sampled']:+.1f} at the sampled weeks). The horizon "
            f"decision stands on the close basis, where 4w leads 1w by "
            f"{lead['4w'] - lead['1w']:+.1f}%/yr compounded. A close-basis grade credits each pick with "
            f"the weekend gap after its rank date, which the deployed book never earns, and 1w takes "
            f"four times as many of those gaps a year as 4w; on the sample the 1w top-6 gap in at "
            f"entry ({p['1w']['gap_in_bp']:+.0f} bp) exceeds the gap at exit ({p['1w']['gap_out_bp']:+.0f} bp), "
            f"so the close basis flatters the shorter horizon if anything. A fill-basis horizon layer "
            f"at population scale needs a full 1w/2y refit (about 1,200 weeks), not run.")


def nested_replay():
    """The nested test's frozen procedure graded as deployed, through the OOS
    explorer's replay (its rankings and world store)."""
    spec = importlib.util.spec_from_file_location("oos_build", HERE.parent / "app/oos/build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sel = new_engine()
    v = mod.VARIANTS["oos"]                       # the clean nested3_v2 walk, frozen_config.json
    _, rets, spy, _ = mod.replay(v, mod.load_config(v))
    return sel.metrics(rets, NESTED_LO, NESTED_HI), sel.metrics(spy, NESTED_LO, NESTED_HI)


def charts():
    rc = campaign()
    rc.CHART_NOTE = "as deployed: fills at the next open, 5 bp a side"
    rc.charts()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "cascade", "report", "charts", "all"])
    step = ap.parse_args().step
    for name, fn in (("build", build), ("cascade", cascade), ("report", report), ("charts", charts)):
        if step in (name, "all"):
            print(f"== {name}", flush=True)
            fn()
