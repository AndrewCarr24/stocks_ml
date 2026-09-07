"""The OpenFE arm, v2: the champion's yardstick and estimator select.

Registered before any v2 number (ledger preregistration
openfe_arm_v2_rank_2006_2015, 2026-09-05) as a rule change after v1's grade,
authorized by the owner. v1 (ops/openfe_arm.py) kept OpenFE's two RMSE
stages; on demedianed 4-week returns they cannot see the ranking signal the
champion is judged by (v1's stage 2 kept one tree that was worse than no
tree). v2 replaces them:

  fit       stage 1 = every candidate's weekly Spearman IC with label_4w on the
            selection rows (2006-01-06 -> 2015-11-20, the full population) as a
            t statistic; candidates in descending |t| are kept only when they
            are not a near-copy (|corr| > MAX_ABS_CORR on the rows' per-week
            ranks) of the champion's 71 features or of a candidate already
            kept; the first BUNDLE_SIZE kept -> formulas.json (stage1_ic.parquet,
            bundle.csv). No significance gate: the walk is the admission test.
  features  the bundle for every panel row, ranked per week -> generated.parquet
  walk      stage 2 = the champion's own walk (4w / 5y / K=4) with ctx.extra =
            the 7 ideas + the bundle, 2006-01 -> 2024-07-18
  grade     selection.simulate at the champion's settings on the live rules;
            2016-01 -> 2024-07 beside the clean, ideas and v1 lines of record
            and SPY, 2006-2015 as in-sample -> grades.json,
            reports/openfe_arm_v2.md, a ledger row

The inputs and the candidate space are v1's (data/experiments/openfe_2006_2015/
inputs.parquet). Run from the repo root:
  PYTHONPATH=src:. .venv/bin/python ops/openfe_arm_v2.py [--generated-only] {fit,features,walk,grade,all}
--generated-only is v3, the arm that serves the owner's purpose (replace the ideas bundle): the
ideas are nowhere in it. Without the flag the script is v2 as run (the ideas ride along).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ops.openfe_arm import (CHAMPION, CHUNK, FIT_HI, FIT_LO, HI, LABEL_DAYS, N_JOBS, RECORD, SPAN,
                            STORE, WALK_HI, WALK_LO, WINDOWS, candidates, fit_rows, gl, log)
from ops import openfe_arm as v1

V1_OUT = v1.OUT                                   # inputs.parquet, grades.json of v1
V2_OUT = Path("data/experiments/openfe_v2_2006_2015")
ARMS = {
    # v2 carried the champion's 7 ideas in the walk and held them in the dedup: it measured what the
    # generator adds beyond the ideas. The owner's purpose is to REPLACE the ideas bundle (its candidate
    # ideas were written from 2016-2024) with a bundle selected on 2006-2015 alone: v3, the ideas nowhere.
    "v2": dict(out=V2_OUT, report=Path("reports/openfe_arm_v2.md"), prereg="openfe_arm_v2_rank_2006_2015",
               ledger="openfe_arm_v2_ideas_generated_2016_2024", ideas=True, bar="ideas",
               label="+ ideas + generated (v2)"),
    "v3": dict(out=Path("data/experiments/openfe_v3_2006_2015"), report=Path("reports/openfe_arm_v3.md"),
               prereg="openfe_arm_v3_generated_2006_2015", ledger="openfe_arm_v3_generated_2016_2024",
               ideas=False, bar="clean", label="+ generated (v3, no ideas)"),
}
ARM = "v2"
OUT, REPORT, PREREG, LEDGER_NAME, IDEAS_IN_ARM, BAR, LABEL = (None,) * 7
BUNDLE_SIZE = 40
MIN_WEEKS = 260                                   # half the scoring window's weeks
MAX_ABS_CORR = 0.9


def configure(arm: str):
    global ARM, OUT, REPORT, PREREG, LEDGER_NAME, IDEAS_IN_ARM, BAR, LABEL
    a = ARMS[arm]
    ARM, OUT, REPORT, PREREG, LEDGER_NAME = arm, a["out"], a["report"], a["prereg"], a["ledger"]
    IDEAS_IN_ARM, BAR, LABEL = a["ideas"], a["bar"], a["label"]


configure(ARM)


# the screened ideas bundle v2 carried (the champion's features 2026-09-05 -> 2026-09-06;
# the spec's features are the generated bundle since, so this list is fixed here)
IDEAS_2026_09_05 = ["x_8k_distress_26w", "x_cash_runway", "x_dps_yoy", "x_earn_react_mean4",
                    "x_earn_window", "x_price_level", "f_vol_chg_12w"]


def champion_ideas():
    return list(IDEAS_2026_09_05)


def ideas():
    """The ideas that ride in this arm: the champion's 7 for v2, none for v3."""
    return champion_ideas() if IDEAS_IN_ARM else []


# ----------------------------------------------------------------------------- stage 1 (workers)
_W = {}


def _init_worker(path):
    os.environ["OMP_NUM_THREADS"] = "1"
    from stocks_ml.features import generated as gen
    d = pd.read_parquet(path)
    _W["inputs"] = d[gen.input_names(d.drop(columns=["y"]))]
    _W["dates"] = d["date"].to_numpy()
    _W["ic"] = gen.WeeklyIC(d["y"].to_numpy(float), _W["dates"], MIN_WEEKS)


def _ic_chunk(formulas):
    from stocks_ml.features import generated as gen
    out = []
    for f in formulas:
        try:
            m, t, w = _W["ic"](gen.rank_week(gen.evaluate(f, _W["inputs"]), _W["dates"]))
        except Exception as e:                    # a degenerate column: scored as useless
            m, t, w = float("nan"), float("nan"), 0
            print(f"  {f}: {e!r}", flush=True)
        out.append((f, m, t, w))
    return out


def stage1(path, formulas, n_jobs):
    chunks = [formulas[i:i + CHUNK] for i in range(0, len(formulas), CHUNK)]
    res, t0, done = [], time.time(), 0
    with Pool(n_jobs, initializer=_init_worker, initargs=(str(path),)) as pool:
        for i, part in enumerate(pool.imap_unordered(_ic_chunk, chunks)):
            res.extend(part)
            done += len(part)
            if (i + 1) % max(1, len(chunks) // 10) == 0 or i + 1 == len(chunks):
                el = time.time() - t0
                log(f"stage 1: {done:,}/{len(formulas):,} in {el / 60:.1f} min "
                    f"(eta {el / done * (len(formulas) - done) / 60:.1f} min)")
    df = pd.DataFrame(res, columns=["formula", "ic", "t", "weeks"])
    df["abs_t"] = df["t"].abs()
    df = df.sort_values("abs_t", ascending=False, na_position="last").drop(columns="abs_t")
    return df.reset_index(drop=True), time.time() - t0


# ----------------------------------------------------------------------------- fit
def fit():
    from stocks_ml.features import generated as gen
    OUT.mkdir(parents=True, exist_ok=True)
    rows, base_cols = fit_rows(FIT_LO, FIT_HI)
    forms, names, constant = candidates(rows)
    log(f"fit: {len(rows):,} rows {rows['date'].min().date()} -> {rows['date'].max().date()}, "
        f"{len(forms):,} candidates over {len(names)} inputs ({len(constant)} week-constant)")
    path = OUT / "stage1_data.parquet"
    meta = {"rows": int(len(rows)), "weeks": int(rows["date"].nunique()),
            "first_week": str(rows["date"].min().date()), "last_week": str(rows["date"].max().date()),
            "candidates": len(forms), "inputs": len(names), "week_constant": constant,
            "min_weeks": MIN_WEEKS, "max_abs_corr": MAX_ABS_CORR, "n_jobs": N_JOBS}
    if not (OUT / "stage1_ic.parquet").exists() and ARM != "v2" and (V2_OUT / "stage1_ic.parquet").exists():
        import shutil
        shutil.copy(V2_OUT / "stage1_ic.parquet", OUT / "stage1_ic.parquet")
        m2 = json.loads((V2_OUT / "fit.json").read_text())
        (OUT / "fit.json").write_text(json.dumps({k: m2[k] for k in
                                                  ("stage1_seconds", "scored", "abs_t_over_2", "abs_t_over_4")}, indent=1))
        log("fit: stage 1 is v2's computation (the held set plays no part in it): scores copied")
    if (OUT / "stage1_ic.parquet").exists():
        scores = pd.read_parquet(OUT / "stage1_ic.parquet")
        meta.update(json.loads((OUT / "fit.json").read_text()))
        log(f"fit: stage 1 loaded from cache ({len(scores):,} scores)")
    else:
        data = rows.drop(columns=base_cols).rename(columns={"label_4w": "y"})
        data.attrs = {}
        data.to_parquet(path, index=False)
        scores, wall = stage1(path, forms, N_JOBS)
        scores.to_parquet(OUT / "stage1_ic.parquet", index=False)
        meta.update(stage1_seconds=round(wall), scored=int(scores["t"].notna().sum()),
                    abs_t_over_2=int((scores["t"].abs() > 2).sum()), abs_t_over_4=int((scores["t"].abs() > 4).sum()))
        (OUT / "fit.json").write_text(json.dumps(meta, indent=1))
        top = scores.iloc[0]
        log(f"stage 1: {len(scores):,} scored in {wall / 60:.1f} min; {meta['scored']:,} with >= {MIN_WEEKS} weeks; "
            f"|t| > 2: {meta['abs_t_over_2']:,}, > 4: {meta['abs_t_over_4']:,}; best {top['formula']} t {top['t']:+.1f}")
    # dedup against what the champion's model would already hold (the ranked base; v2 also the 7 ideas), then the kept
    held_names = base_cols + ideas()
    if ideas():
        pan = pd.read_parquet(f"{STORE}/panel_sf.parquet", columns=["date", "ticker"] + ideas())
        rows = rows.merge(pan, on=["date", "ticker"], how="left")
        assert rows[ideas()].notna().all().all(), "an idea column is missing on the scoring rows"
    held = rows[held_names].to_numpy(np.float32)
    inp = rows[names]
    dates = rows["date"].to_numpy()
    order = scores[scores["t"].notna()]["formula"].tolist()
    t0 = time.time()
    scanned = []

    def column(f):
        scanned.append(f)
        return gen.rank_week(gen.evaluate(f, inp), dates)

    kept = gen.dedup(order, held, column, BUNDLE_SIZE, MAX_ABS_CORR)
    s = scores.set_index("formula")
    table = pd.DataFrame({"formula": [f for f, _ in kept], "ic": [s.loc[f, "ic"] for f, _ in kept],
                          "t": [s.loc[f, "t"] for f, _ in kept], "weeks": [int(s.loc[f, "weeks"]) for f, _ in kept],
                          "rank": [order.index(f) + 1 for f, _ in kept], "max_corr_held": [r for _, r in kept]})
    table.to_csv(OUT / "bundle.csv", index=False)
    formulas = {f"{gen.PREFIX}{i:02d}": f for i, f in enumerate(table["formula"])}
    (OUT / "formulas.json").write_text(json.dumps(formulas, indent=1))
    meta.update(bundle=len(formulas), dedup_scanned=len(scanned), dedup_seconds=round(time.time() - t0),
                held=held_names, bundle_min_abs_t=float(table["t"].abs().min()))
    (OUT / "fit.json").write_text(json.dumps(meta, indent=1))
    log(f"dedup: {len(scanned):,} candidates scanned in {time.time() - t0:.0f}s, {len(formulas)} kept "
        f"(|t| down to {meta['bundle_min_abs_t']:.1f}) -> {OUT / 'formulas.json'}")
    for n, f in list(formulas.items())[:12]:
        log(f"  {n} = {f}  t {s.loc[f, 't']:+.1f}")


# ----------------------------------------------------------------------------- features / walk
def features():
    from stocks_ml.features import generated as gen
    formulas = json.loads((OUT / "formulas.json").read_text())
    inp = pd.read_parquet(V1_OUT / "inputs.parquet")
    t0 = time.time()
    out = gen.add_generated(inp[["date", "ticker"]], inp, formulas)
    out.to_parquet(OUT / "generated.parquet", index=False)
    log(f"features: {len(formulas)} g_ columns on {len(out):,} rows in {time.time() - t0:.0f}s")


def ctx_with_bundle():
    """The champion's Ctx with the bundle's g_ columns on the panel and
    ctx.extra = the 7 ideas + the bundle."""
    import stocks_ml.selection as sel
    from stocks_ml.features import generated as gen
    formulas = json.loads((OUT / "formulas.json").read_text())
    ctx = sel.Ctx(STORE)
    g = pd.read_parquet(OUT / "generated.parquet")
    g = g.set_index(pd.MultiIndex.from_frame(g[["date", "ticker"]])).drop(columns=["date", "ticker"])
    g = g.reindex(pd.MultiIndex.from_frame(ctx.pan[["date", "ticker"]]))
    missing = int(g.isna().any(axis=1).sum())
    g.index = ctx.pan.index
    ctx.pan = pd.concat([ctx.pan.drop(columns=[c for c in ctx.pan.columns if c.startswith(gen.PREFIX)]),
                         g.fillna(0.0)], axis=1)
    ctx.extra = ideas() + list(formulas)
    assert all(c in ctx.pan.columns for c in ctx.extra)
    log(f"ctx: panel {ctx.pan.shape}, extra = {len(ideas())} ideas + {len(formulas)} generated ({ARM})"
        + (f", {missing} rows without inputs (neutral-filled)" if missing else ""))
    return sel, ctx, formulas


def walk():
    sel, ctx, _ = ctx_with_bundle()
    stem = sel.holdings_name("4w", 5, ctx.extra)
    log(f"walk: {stem} on {WALK_LO.date()} -> {WALK_HI.date()} (4w / 5y / K={sel.K_COPIES})")
    t0 = time.time()
    sel.stage_holdings(ctx, str(OUT), "4w", 5, WALK_LO, WALK_HI)
    n = len(pd.read_parquet(OUT / f"{stem}_s0.parquet"))
    log(f"walk: {n} rank weeks in {(time.time() - t0) / 3600:.2f} h -> {OUT / (stem + '_s0.parquet')}")


# ----------------------------------------------------------------------------- grade
def grade():
    from stocks_ml.models.trials import record_trials
    sel, ctx, formulas = ctx_with_bundle()
    stem = sel.holdings_name("4w", 5, ctx.extra)
    h = pd.read_parquet(OUT / f"{stem}_s0.parquet")
    h["week"] = pd.to_datetime(h["week"])
    h = h[h.week < HI].sort_values("week")
    s = sel.simulate(ctx, h, **CHAMPION)
    spy = ctx.wret["SPY"].reindex(s.index)
    res = {"arm": ARM, "rank_weeks": int(len(h)), "first_rank_week": str(h.week.min().date()),
           "last_rank_week": str(h.week.max().date()), "ideas": ideas(), "features": formulas,
           "holdings": f"{stem}_s0.parquet"}
    for w, (a, b) in WINDOWS.items():
        res[w] = {"config": sel.metrics(s, pd.Timestamp(a), pd.Timestamp(b)),
                  "sp500": sel.metrics(spy, pd.Timestamp(a), pd.Timestamp(b))}
    (OUT / "grades.json").write_text(json.dumps(res, indent=1))
    rec = json.loads(RECORD.read_text())
    fitmeta = json.loads((OUT / "fit.json").read_text())
    bundle = pd.read_csv(OUT / "bundle.csv").set_index("formula")
    # the lines beside this arm's, in table order; earlier arms only when graded
    lines = [("clean (record)", {w: rec["clean_recorded"][w]["config"] for w in WINDOWS}),
             ("+ ideas* (record)", {w: rec["bundle_recorded"][w]["config"] for w in WINDOWS})]
    for name, out in (("+ generated (v1)", V1_OUT), ("+ ideas + generated (v2)", V2_OUT)):
        if out != OUT and (out / "grades.json").exists():
            g = json.loads((out / "grades.json").read_text())
            lines.append((name, {w: g[w]["config"] for w in WINDOWS}))
    lines.append((LABEL, {w: res[w]["config"] for w in WINDOWS}))
    lines.append(("SPY", {w: res[w]["sp500"] for w in WINDOWS}))
    num = res["2016_2024"]["config"]["terminal_100"]
    bar_name, bar = (("+ ideas*", rec["bundle_recorded"]["2016_2024"]["config"]["terminal_100"]) if BAR == "ideas"
                     else ("clean", rec["clean_recorded"]["2016_2024"]["config"]["terminal_100"]))
    passes = num > bar
    weeks = [t for t in ctx.weeks if WALK_LO <= t <= min(WALK_HI, HI)]
    skipped = sorted(set(weeks) - set(h.week))
    fit_hi = FIT_HI - pd.Timedelta(days=LABEL_DAYS)
    held = f"{len(fitmeta.get('held', []))} features (64 base + the 7 ideas)" if IDEAS_IN_ARM else "64 base features"
    purpose = ("This arm carried the champion's 7 ideas in the walk and held them in the dedup: it measures what the "
               "generator adds beyond the ideas. That was not the owner's purpose (see v3)." if IDEAS_IN_ARM else
               "The owner's purpose: replace the ideas bundle — whose candidate ideas were written after reading "
               "2016-2024 — with a bundle selected on 2006-2015 alone. The ideas are nowhere in this arm: not in the "
               "walk, not held in the dedup.")
    if passes:
        verdict = (f"the generated bundle beats the {bar_name} line it had to beat. "
                   + ("Adoption is the owner's decision; nothing has been changed."
                      if IDEAS_IN_ARM else
                      "It is a valid feature set with no asterisk; whether it replaces the ideas bundle in the "
                      "champion and the live job is the owner's decision (the + ideas* line is reference, not the "
                      "bar). Nothing has been changed."))
    else:
        verdict = (f"the generated bundle does not beat the {bar_name} line it had to beat. No change to the "
                   "champion, the spec or the live job.")
    md = [f"# The OpenFE arm, {ARM}: the champion's yardstick selects"
          + (" — generated only" if not IDEAS_IN_ARM else ""), "",
          f"Generated by `ops/openfe_arm_v2.py{' --generated-only' if not IDEAS_IN_ARM else ''}` on "
          f"{pd.Timestamp.today().date()}; the rule is the ledger's preregistration `{PREREG}`, written before any "
          f"{ARM} number. {purpose} Same inputs and candidates as v1 ({fitmeta.get('inputs')} inputs, "
          f"{fitmeta.get('candidates', 0):,} formulas), scored on {fitmeta.get('first_week')} -> "
          f"{fitmeta.get('last_week')} ({fitmeta.get('rows', 0):,} stock-weeks, {fitmeta.get('weeks')} weeks, the "
          "full population). Stage 1: each candidate's weekly Spearman IC with the 4-week demedianed return as a t "
          f"statistic (candidates with fewer than {MIN_WEEKS} valid weeks excluded). Dedup: in descending |t|, a "
          f"candidate is kept only if its |correlation| with every column already held — the champion's {held} and "
          f"every candidate kept before it — is at most {MAX_ABS_CORR}; the first {BUNDLE_SIZE} kept are the bundle. "
          "No significance gate. Stage 2 is the champion's own walk (4w / 5y / K=4) with "
          + ("the 7 ideas + the bundle" if IDEAS_IN_ARM else "the bundle as the only extra features")
          + f", {res['first_rank_week']} -> {res['last_rank_week']}, graded by `selection.simulate` at the champion's "
          "settings (top-6 / 70-30 trend ballast / no stop / cap 2) on the live rules. Clean and + ideas* are the "
          "records in `champion_bundle_grades.json` (walks from 2001-06); the earlier arms are their own reports; SPY "
          "on this walk's weeks. Holdout untouched.", "",
          "## Champion settings", "",
          "| window | " + " | ".join(n for n, _ in lines) + " |", "|---|" + "---|" * len(lines)]
    for w in WINDOWS:
        md.append(f"| {SPAN[w]} | " + " | ".join(gl(m[w]) for _, m in lines) + " |")
    md += ["", f"**The rule's verdict:** {LABEL} ${num:,.0f} vs {bar_name} ${bar:,.0f} on 2016-01 -> 2024-07 — "
           + verdict, "",
           "The asterisk on the ideas line: its thirty candidates were written after reading the whole 2006-2024 "
           "record. " + ("The v2 line inherits it (the ideas ride along)." if IDEAS_IN_ARM else
                         "This arm's line carries no asterisk — every choice in it was made on 2006-2015.")
           + " The caveat recorded in v1's preregistration remains: the input rule was written by someone who knows "
           "those ideas exist; the rule is generic (every field, one standard window).", "",
           f"## The bundle ({len(formulas)} formulas; selection-window IC, {FIT_LO.date()} -> {fit_hi.date()}; "
           "no grading year read)", "",
           "| column | formula | IC | t | weeks | stage-1 rank | max corr with held |", "|---|---|---|---|---|---|---|"]
    for n, f in formulas.items():
        b = bundle.loc[f]
        md.append(f"| {n} | `{f}` | {b['ic']:+.4f} | {b['t']:+.1f} | {int(b['weeks'])} | {int(b['rank'])} | {b['max_corr_held']:.2f} |")
    md += ["", "## Notes", "",
           f"1. **Stage 1.** {fitmeta.get('scored', 0):,} of {fitmeta.get('candidates', 0):,} candidates had "
           f"{MIN_WEEKS}+ valid weeks; {fitmeta.get('abs_t_over_2', 0):,} scored |t| > 2 and "
           f"{fitmeta.get('abs_t_over_4', 0):,} |t| > 4. With this many candidates the largest |t| under the null is "
           "about 4.5 (search inflation), which is why the t is reported and the walk decides. The dedup scanned "
           f"{fitmeta.get('dedup_scanned', 0):,} candidates to keep {len(formulas)} (|t| down to "
           f"{fitmeta.get('bundle_min_abs_t', 0):.1f}); the correlation is the pooled Pearson correlation of the "
           "per-week ranked columns on the scoring rows. Stage 1 took "
           f"{fitmeta.get('stage1_seconds', 0) / 60:.1f} min on {fitmeta.get('n_jobs')} workers"
           + (" (v2's computation, reused)" if ARM != "v2" else "") + f", the dedup {fitmeta.get('dedup_seconds', 0)} s.", "",
           f"2. **Weeks.** The walk covers {res['first_rank_week']} -> {res['last_rank_week']}, {res['rank_weeks']} "
           f"rank weeks; {len(skipped)} week{'s' if len(skipped) != 1 else ''} in range produced no ranking (the "
           "ensemble's predictions had fewer than 20 distinct values, the same skip rule as the records)"
           + (f": {', '.join(str(t.date()) for t in skipped)}" if skipped else "")
           + ". The ledger holds through a skipped week. The records' walks start 2001-06, so their 2006-2015 line "
           "begins fully invested while this walk fills its sleeves over its first weeks; the 2016 -> 2024-07 "
           "number is on the same basis (calendar-based sleeve rotation).", ""]
    REPORT.write_text("\n".join(md))
    print("\n".join(md))
    m, sp = res["2016_2024"]["config"], res["2016_2024"]["sp500"]
    ins = res["2006_2015"]["config"]
    beside = "; ".join(f"{n} {gl(mm['2016_2024'])}" for n, mm in lines if n not in (LABEL, "SPY"))
    record_trials([{
        "kind": "r4w_strategy", "name": LEDGER_NAME,
        "config": {**CHAMPION, "train_years": 5, "fills": "next open", "cost_bps_per_side": 5.0,
                   "features": ideas() + list(formulas), "formulas": formulas},
        "terminal_100": m["terminal_100"], "cagr_pct": m["cagr_pct"], "sharpe": m["sharpe"],
        "max_dd": m["max_dd"], "n_weeks": m["n_weeks"], "spy_terminal_100": sp["terminal_100"],
        "spy_cagr_pct": sp["cagr_pct"], "spy_sharpe": sp["sharpe"], "spy_max_dd": sp["max_dd"],
        "notes": (f"The OpenFE arm {ARM}'s one grade (preregistration {PREREG}): the champion's settings with "
                  + (f"its 7 ideas + {len(formulas)} generated features" if IDEAS_IN_ARM else
                     f"{len(formulas)} generated features and NO ideas (the ideas bundle's clean replacement candidate)")
                  + " chosen by the weekly-IC t statistic and dedup (formulas in config), 2016-01 -> 2024-07: "
                  f"{gl(m)} vs {beside} vs SPY {gl(sp)}; 2006-2015 in-sample {gl(ins)} vs clean "
                  f"{gl(rec['clean_recorded']['2006_2015']['config'])} vs + ideas* "
                  f"{gl(rec['bundle_recorded']['2006_2015']['config'])}. Rule's verdict ({bar_name} is the bar): "
                  + verdict + f" Walk {res['holdings']} {res['first_rank_week']} -> {res['last_rank_week']}; "
                  f"{REPORT}. Holdout untouched.")}])
    log(f"grade: 2016-01 -> 2024-07 {LABEL} {gl(m)} | {beside} | SPY {gl(sp)}"
        f" -> {'PASSES' if passes else 'does not pass'} the rule ({bar_name} ${bar:,.0f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fit", "features", "walk", "grade", "all"])
    ap.add_argument("--generated-only", action="store_true",
                    help="v3: the ideas nowhere (the owner's purpose: replace the ideas bundle); default v2")
    args = ap.parse_args()
    configure("v3" if args.generated_only else "v2")
    step = args.step
    for name, fn in (("fit", fit), ("features", features), ("walk", walk), ("grade", grade)):
        if step in (name, "all"):
            log(f"== {name}")
            fn()
