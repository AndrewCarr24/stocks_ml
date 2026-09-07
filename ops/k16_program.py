"""The K=16 program (owner's go 2026-09-07): two more sixteen-copy walks.

  clean_2016_2024     step 1 - the clean line (base features, no bundle) at
                      sixteen copies on 2016-01 -> 2024-07. Preregistration
                      k16_clean_2016_2024: the generated bundle stays iff its
                      K=16 ($896, k16_seed_spread) beats the clean line's K=16
                      on the same weeks, settings and rules.
  champion_2006_2015  step 2 - the champion (generated bundle) at sixteen
                      copies on 2006-01 -> 2015-12. Preregistration
                      k16_champion_2006_2015: joined with the seed spread's
                      2016-2024 copies it is the K=16 walk over 2006-2024; the
                      cascade's book-down layers are re-decided on its
                      2006-2015 holdings at K=16 and at each K=4 draw.

Every walk fits copies c = 1..16 under replication.py's standard
(random_state c + whole-week bootstrap seeded by c) on every rank week of its
span, checkpoints the per-copy scores every ten weeks and resumes from them.
Ensembles are graded as ops/k16_seed_spread.py grades them: the mean over the
copies, simulated at the champion's settings on the live rules. Holdout
untouched (rank weeks end 2024-07-12). Nothing here changes the champion, the
spec, K or the live job: the protocol switch is step 3, on the owner's go.

Run from the repo root:
  PYTHONPATH=src:. .venv/bin/python ops/k16_program.py {walk,grade,all} {clean_2016_2024,champion_2006_2015}
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ops.k16_seed_spread import (CHAMPION, CHECKPOINT, K, RECORD, STORE, V3, V3_HOLDINGS, agreement,
                                 copy_preds, ensemble_rows, gl, log)
from ops.k16_seed_spread import OUT as SPREAD

CLEAN_WALK = Path("data/experiments/champion_2006_2024/holdings_4w_5y_s0.parquet")   # the clean walk of record
ENSEMBLES = {"copies 1-4 (record, K=4)": range(1, 5), "copies 5-8 (K=4)": range(5, 9),
             "copies 9-12 (K=4)": range(9, 13), "copies 13-16 (K=4)": range(13, 17),
             "copies 1-8 (K=8)": range(1, 9), "copies 9-16 (K=8)": range(9, 17),
             "copies 1-16 (K=16)": range(1, 17)}
WINDOWS = {"2006_2015": (pd.Timestamp("2006"), pd.Timestamp("2016")),
           "2016_2024": (pd.Timestamp("2016"), pd.Timestamp("2025")),
           "pre_holdout": (pd.Timestamp("2006"), pd.Timestamp("2025"))}
SELECT = (pd.Timestamp("2006-01-01"), pd.Timestamp("2015-12-31"))      # the cascade's window (nested3)
RUNS = {
    "clean_2016_2024": dict(arm="clean", lo=pd.Timestamp("2016-01-01"), hi=pd.Timestamp("2024-07-18"),
                            out=Path("data/experiments/k16_clean_2016_2024"),
                            report=Path("reports/k16_clean_line.md")),
    "champion_2006_2015": dict(arm="generated", lo=pd.Timestamp("2006-01-01"), hi=pd.Timestamp("2015-12-31"),
                               out=Path("data/experiments/k16_champion_2006_2015"),
                               report=Path("reports/k16_champion_record.md")),
}


def context(arm):
    import stocks_ml.selection as sel
    from stocks_ml.features.bundle import FEATURES
    ctx = sel.Ctx(STORE)
    ctx.extra = list(FEATURES) if arm == "generated" else []
    missing = [c for c in ctx.extra if c not in ctx.pan.columns]
    if missing:
        raise RuntimeError(f"panel lacks {missing[:3]}...")
    return sel, ctx


def load_preds(path):
    p = pd.read_parquet(path)
    p["week"] = pd.to_datetime(p["week"])
    return p


def load_walk(path):
    h = pd.read_parquet(path)
    h["week"] = pd.to_datetime(h["week"])
    return h.sort_values("week").reset_index(drop=True)


# ----------------------------------------------------------------------------- walk
def walk(run):
    r = RUNS[run]
    sel, ctx = context(r["arm"])
    r["out"].mkdir(parents=True, exist_ok=True)
    h = sel.HORIZONS["4w"]
    cfg2 = ctx.world_cfg(5)
    weeks = [t for t in ctx.weeks if r["lo"] <= t <= r["hi"]]
    path = r["out"] / "preds.parquet"
    frames, done = [], set()
    if path.exists():
        old = load_preds(path)
        frames.append(old)
        done = set(old["week"].unique())
    todo = [t for t in weeks if t not in done]
    log(f"walk {run}: {len(weeks)} rank weeks {weeks[0].date()} -> {weeks[-1].date()}, {len(todo)} to do, "
        f"K={K} copies each; panel {ctx.pan.shape}, features = base"
        + (f" + {len(ctx.extra)} generated" if ctx.extra else " only (clean)"))
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
            log(f"  {i}/{len(todo)} weeks, {el / i:.1f} s/week, ~{el / i * (len(todo) - i) / 3600:.2f} h left")
    log(f"walk {run}: done in {(time.time() - t0) / 3600:.2f} h -> {path}")


# ----------------------------------------------------------------------------- grade
def same_rows(a, b, sel=None, ctx=None, means=None):
    """Row-for-row reproduction of a recorded walk `b` by the rows `a` on the
    weeks both have: identical top-15 lists, equal book returns. Two things
    make a faithful rerun differ without a single score differing, and both
    are sorted out with the ensemble's scores (`means`, week -> Series):

    * ties: the shallow trees give many names exactly the same score, and
      slice_row orders ties by whatever order the scores arrive in, so a
      rerun can list tied names in another order or swap tied names across
      the 15th (or a book's) slot;
    * delistings: the clean record's rows were built by regrade_campaign.py
      from the campaign's cached rankings verbatim, which rank every member
      the way the live job does; slice_row ranks only names with a forward
      label, so a name that leaves the tape within the horizon (ADT1 2016-04,
      WFM 2017-08, SBNY 2023-03) is in the record's list and not in a rerun's.

    Returns (exact, up_to_ties, n_common, counts) with counts of the
    differing weeks by cause: tie_order, delisted_in_horizon, unexplained."""
    common = sorted(set(a.week) & set(b.week))
    x = a[a.week.isin(common)].sort_values("week").reset_index(drop=True)
    y = b[b.week.isin(common)].sort_values("week").reset_index(drop=True)
    exact = bool(x[["week", "top15"]].equals(y[["week", "top15"]]) and np.allclose(
        x[["top3", "top6", "top10"]].to_numpy(), y[["top3", "top6", "top10"]].to_numpy()))
    counts = {"tie_order": 0, "delisted_in_horizon": 0, "unexplained": 0}
    if exact or means is None:
        counts["unexplained"] = sum(u != v for u, v in zip(x.top15, y.top15))
        return exact, exact, len(common), counts
    for t, u, v in zip(x.week, x.top15, y.top15):
        if u == v:
            continue
        r = ctx.fwd["4w"].loc[sel.week_slot(ctx.fwd["4w"].index, t)]
        uni = [n for n in ctx.members[t] if n in r.index and not pd.isna(r[n])]
        q = means[t].loc[means[t].index.intersection(pd.Index(uni))]
        lst = v.split(",")
        gone = [n for n in lst if n in ctx.members[t] and n not in q.index]   # ranked then, unlabelled now
        lst = [n for n in lst if n not in gone]
        srt = q.sort_values(ascending=False)
        thr = srt.iloc[len(lst) - 1]
        must, may = set(srt[srt > thr].index), set(srt[srt == thr].index)
        if (must <= set(lst) <= must | may
                and all(q.get(m, -np.inf) >= q.get(n, -np.inf) for m, n in zip(lst, lst[1:]))):
            counts["delisted_in_horizon" if gone else "tie_order"] += 1
        else:
            counts["unexplained"] += 1
    return exact, counts["unexplained"] == 0, len(common), counts


def pairs_agreement(means):
    k4 = [n for n in ENSEMBLES if n.endswith("K=4)")]
    pairs = {"K=4 vs K=4 (6 disjoint pairs)": [(a, b) for a, b in itertools.combinations(k4, 2)],
             "K=8 vs K=8 (1 disjoint pair)": [("copies 1-8 (K=8)", "copies 9-16 (K=8)")]}
    out = {}
    for label, ps in pairs.items():
        vals = [agreement(means[a], means[b]) for a, b in ps]
        out[label] = {"spearman": float(np.mean([v[0] for v in vals])),
                      "spearman_range": [float(min(v[0] for v in vals)), float(max(v[0] for v in vals))],
                      "top6_shared": float(np.mean([v[1] for v in vals])), "weeks": vals[0][2]}
    return out


def slug(name):
    return name.split(" (")[0].replace(" ", "_").replace("-", "_")


def grade_clean():
    """Step 1: the clean line's seven ensembles beside the generated bundle's,
    then the preregistered rule."""
    from stocks_ml.models.trials import record_trials
    r = RUNS["clean_2016_2024"]
    lo, hi, grade = r["lo"], r["hi"], WINDOWS["2016_2024"]
    sel, ctx = context("clean")
    preds = load_preds(r["out"] / "preds.parquet")
    record = load_walk(CLEAN_WALK)
    prefix = record[record.week < lo]
    record_span = record[(record.week >= lo) & (record.week < hi)]
    rec = json.loads(RECORD.read_text())["clean_recorded"]["2016_2024"]["config"]
    gen = json.loads((SPREAD / "grades.json").read_text())["ensembles"]
    res = {"rank_weeks": int(preds.week.nunique()), "first_rank_week": str(preds.week.min().date()),
           "last_rank_week": str(preds.week.max().date()), "copies": K, "ensembles": {}}
    means = {}
    for name, copies in ENSEMBLES.items():
        rows, means[name] = ensemble_rows(sel, ctx, preds, copies)
        hold = pd.concat([prefix, rows], ignore_index=True).sort_values("week")
        hold = hold[hold.week < hi]
        s = sel.simulate(ctx, hold, **CHAMPION)
        m = sel.metrics(s, *grade)
        entry = {"copies": list(copies), "config": m, "rows": int(len(rows)),
                 "sp500": sel.metrics(ctx.wret["SPY"].reindex(s.index), *grade)}
        if name.startswith("copies 1-4"):
            ok, ties, n, why = same_rows(rows, record_span, sel, ctx, means[name])
            entry["reproduces_record"] = ok
            entry["reproduces_record_up_to_ties"] = ties
            entry["common_weeks"] = n
            entry["weeks_differing"] = sum(why.values())
            entry["differing_weeks_by_cause"] = why
            nd = entry["weeks_differing"]
            # the record's own credited weeks (its walk ends 2024-06-14): must read $589.8
            own = hold[hold.week.isin(set(record.week))]
            entry["on_record_weeks"] = sel.metrics(sel.simulate(ctx, own, **CHAMPION), *grade)
            entry["record"] = rec
            log(f"copies 1-4 reproduce the clean walk of record row for row: {ok} on {n} common weeks "
                f"({nd} differ: {why}; every difference explained: {ties}); "
                f"on the record's own weeks {gl(entry['on_record_weeks'])} vs record {gl(rec)}")
        res["ensembles"][name] = entry
        log(f"{name}: {gl(m)} on {m['n_weeks']} weeks; SPY {gl(entry['sp500'])}")
    res["agreement"] = pairs_agreement(means)
    ens = res["ensembles"]
    gen_by = {slug(n): e for n, e in gen.items()}
    clean16 = ens["copies 1-16 (K=16)"]["config"]["terminal_100"]
    gen16 = gen_by["copies_1_16"]["config"]["terminal_100"]
    keep = gen16 > clean16
    res["rule"] = {"generated_k16": gen16, "clean_k16": clean16, "bundle_stays": keep}
    (r["out"] / "grades.json").write_text(json.dumps(res, indent=1))
    k4 = [n for n in ENSEMBLES if n.endswith("K=4)")]
    c_terms = [ens[n]["config"]["terminal_100"] for n in k4]
    g_terms = [gen_by[slug(n)]["config"]["terminal_100"] for n in k4]
    verdict = (f"the generated bundle STAYS (K=16: ${gen16:,.0f} vs clean ${clean16:,.0f})" if keep else
               f"the CLEAN line wins at K=16 (${clean16:,.0f} vs generated ${gen16:,.0f}): the bundle's adoption "
               "is reverted per the preregistration (live change on the owner's go) and step 2 reruns on the clean line")
    md = ["# The clean line at sixteen copies, 2016-01 -> 2024-07", "",
          f"Generated by `ops/k16_program.py grade clean_2016_2024` on {pd.Timestamp.today().date()}; the rule is the "
          "ledger's preregistration `k16_clean_2016_2024` (written before any number). The champion's model at its "
          "settings (4w / 5y / purge 35) on the panel's base features only (no bundle), sixteen copies per rank week "
          f"under replication.py's seeding, {res['first_rank_week']} -> {res['last_rank_week']} ({res['rank_weeks']} "
          "weeks) on the research store; each copy's scores saved. Every ensemble starts 2016 from the clean walk of "
          "record's own pre-2016 rows, is simulated at the champion's settings (top-6 / cap 2 / no stop / 70-30 trend "
          "ballast) on the live ledger's rules and graded on 2016-01-01 -> 2025-01-01 (exclusive), beside the generated "
          "bundle's ensembles from `reports/k16_seed_spread.md` (same weeks, settings and rules). Holdout untouched.", "",
          "## 2016-01 -> 2024-07 at the champion's settings: clean vs generated, copy for copy", "",
          "| ensemble | clean: terminal $100 | CAGR | Sharpe | max DD | generated: terminal $100 | CAGR | Sharpe | max DD | weeks |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in ens.items():
        m, g = e["config"], gen_by[slug(name)]["config"]
        md.append(f"| {name} | ${m['terminal_100']:,.0f} | {m['cagr_pct']:+.1f}% | {m['sharpe']:.2f} | {m['max_dd']:.0%} "
                  f"| ${g['terminal_100']:,.0f} | {g['cagr_pct']:+.1f}% | {g['sharpe']:.2f} | {g['max_dd']:.0%} | {m['n_weeks']} |")
    sp = ens["copies 1-16 (K=16)"]["sp500"]
    md.append(f"| SPY (this walk's weeks) | ${sp['terminal_100']:,.0f} | {sp['cagr_pct']:+.1f}% | {sp['sharpe']:.2f} | "
              f"{sp['max_dd']:.0%} | | | | | {sp['n_weeks']} |")
    e14 = ens["copies 1-4 (record, K=4)"]
    why = e14["differing_weeks_by_cause"]
    md += ["", f"Copies 1-4 reproduce the clean walk of record exactly on {e14['common_weeks'] - e14['weeks_differing']} "
           f"of {e14['common_weeks']} common weeks; every other week is explained without any score differing: "
           f"**{e14['reproduces_record_up_to_ties']}** ({why['tie_order']} weeks reorder or swap names with exactly equal "
           "scores — the shallow trees give many names the same score and slice_row orders ties by arrival order; "
           f"{why['delisted_in_horizon']} weeks list a name that left the tape within the horizon — ADT1, WFM, SBNY — "
           "which the record's rows keep because regrade_campaign.py built them from the cached rankings verbatim, as "
           f"the live job ranks, while slice_row ranks labelled names only; {why['unexplained']} unexplained). Those "
           "delisting names move the record's grade by $1 (kept $589.8, removed $591.2). On the record's own credited "
           f"weeks {gl(e14['on_record_weeks'])} vs the record {gl(rec)}.", "",
           f"Four K=4 draws: clean ${min(c_terms):,.0f} to ${max(c_terms):,.0f} (mean ${np.mean(c_terms):,.0f}); "
           f"generated ${min(g_terms):,.0f} to ${max(g_terms):,.0f} (mean ${np.mean(g_terms):,.0f}). K=8 draws: clean "
           f"${ens['copies 1-8 (K=8)']['config']['terminal_100']:,.0f} / ${ens['copies 9-16 (K=8)']['config']['terminal_100']:,.0f}, "
           f"generated ${gen_by['copies_1_8']['config']['terminal_100']:,.0f} / ${gen_by['copies_9_16']['config']['terminal_100']:,.0f}. "
           f"K=16: clean ${clean16:,.0f}, generated ${gen16:,.0f}.", "",
           f"**Preregistered rule (generated stays iff its K=16 beats clean's K=16): {verdict}.**", "",
           "## How much two independent clean ensembles agree (week-averaged)", "",
           "| ensembles | Spearman of scores | range | top-6 names shared | weeks |", "|---|---|---|---|---|"]
    for label, a in res["agreement"].items():
        md.append(f"| {label} | {a['spearman']:.2f} | {a['spearman_range'][0]:.2f}-{a['spearman_range'][1]:.2f} | "
                  f"{a['top6_shared']:.1f} of 6 | {a['weeks']} |")
    md.append("")
    r["report"].write_text("\n".join(md))
    print("\n".join(md))
    entries = []
    for name, e in ens.items():
        m, spm = e["config"], e["sp500"]
        entries.append({
            "kind": "seed_replication", "name": f"k16_clean_{slug(name)}_2016_2024",
            "config": {**CHAMPION, "train_years": 5, "fills": "next open", "cost_bps_per_side": 5.0,
                       "copies": e["copies"], "features": []},
            "terminal_100": m["terminal_100"], "cagr_pct": m["cagr_pct"], "sharpe": m["sharpe"],
            "max_dd": m["max_dd"], "n_weeks": m["n_weeks"], "spy_terminal_100": spm["terminal_100"],
            "spy_cagr_pct": spm["cagr_pct"], "spy_sharpe": spm["sharpe"], "spy_max_dd": spm["max_dd"],
            "notes": (f"Seed spread of the clean line (preregistration k16_clean_2016_2024): {name} of sixteen, "
                      f"2016-01 -> 2024-07: {gl(m)} vs SPY {gl(spm)}; the generated bundle's same copies "
                      f"{gl(gen_by[slug(name)]['config'])}. reports/k16_clean_line.md. Holdout untouched.")})
    entries.append({
        "kind": "verdict", "name": "k16_clean_verdict",
        "notes": (f"Outcome of preregistration k16_clean_2016_2024 ({pd.Timestamp.today().date()}, "
                  f"reports/k16_clean_line.md): (a) copies 1-4 reproduce the clean walk of record exactly on "
                  f"{e14['common_weeks'] - e14['weeks_differing']} of {e14['common_weeks']} common weeks and every other "
                  f"week is explained with no score differing: {e14['reproduces_record_up_to_ties']} "
                  f"({why['tie_order']} tie-order weeks: equal scores ordered by arrival in slice_row; "
                  f"{why['delisted_in_horizon']} weeks where the record lists a name that left the tape within the "
                  f"horizon (ADT1, WFM, SBNY) — the record's rows keep the cached rankings verbatim (regrade_campaign.py), "
                  f"slice_row ranks labelled names only; worth $1 on the record's grade); ({gl(e14['on_record_weeks'])} on its "
                  f"own weeks vs the record {gl(rec)}); (b) clean's four K=4 draws ${min(c_terms):,.0f}-${max(c_terms):,.0f} "
                  f"(mean ${np.mean(c_terms):,.0f}) vs the bundle's ${min(g_terms):,.0f}-${max(g_terms):,.0f} "
                  f"(mean ${np.mean(g_terms):,.0f}); (c) K=16: clean {gl(ens['copies 1-16 (K=16)']['config'])} vs "
                  f"generated {gl(gen_by['copies_1_16']['config'])}, SPY {gl(sp)}. RULE: {verdict}. Agreement between "
                  f"disjoint clean ensembles: K=4 Spearman {res['agreement']['K=4 vs K=4 (6 disjoint pairs)']['spearman']:.2f}, "
                  f"K=8 {res['agreement']['K=8 vs K=8 (1 disjoint pair)']['spearman']:.2f}. No change to the champion, the "
                  "spec, K or the live job by this row: the protocol switch is step 3 on the owner's go.")})
    log(f"ledger: {record_trials(entries)} rows")
    return res


def cascade_at(sel, ctx, hold):
    """run_cascade's book-down layers on the given holdings over the
    selection window: book by cost-adjusted compounded %/yr, then floor, stop
    and cap by Sharpe, each at the picks above it."""
    lo, hi = SELECT
    hold = hold[(hold.week >= lo) & (hold.week <= hi)]
    book, bres = sel.decide_book(hold, "4w", lo, hi)
    fres = {f: sel.sharpe(sel.simulate(ctx, hold, "4w", book, None, None, f), lo, hi) for f in sel.FLOORS}
    floor = max(fres, key=fres.get)
    sres = {str(s): sel.sharpe(sel.simulate(ctx, hold, "4w", book, None, s, floor), lo, hi) for s in (None, -0.25)}
    stop = None if sres["None"] >= sres["-0.25"] else -0.25
    cres = {str(c): sel.sharpe(sel.simulate(ctx, hold, "4w", book, c, stop, floor), lo, hi) for c in (None, 2)}
    cap = None if cres["None"] >= cres["2"] else 2
    return {"book": int(book), "floor": floor, "stop": stop, "cap": cap,
            "evidence": {"book": {str(k): round(v, 2) for k, v in bres.items()},
                         "floor": {k: round(v, 3) for k, v in fres.items()},
                         "stop": {k: round(v, 3) for k, v in sres.items()},
                         "cap": {k: round(v, 3) for k, v in cres.items()}}}


def grade_champion():
    """Step 2: the sixteen copies over 2006-2024 (this walk + the seed
    spread's), graded on the arm's windows; the cascade's book-down layers
    re-decided on 2006-2015 at every K."""
    from stocks_ml.models.trials import record_trials
    r = RUNS["champion_2006_2015"]
    sel, ctx = context("generated")
    preds = pd.concat([load_preds(r["out"] / "preds.parquet"), load_preds(SPREAD / "preds.parquet")],
                      ignore_index=True).sort_values(["week", "ticker"])
    v3 = load_walk(V3_HOLDINGS)
    v3_grades = json.loads((V3 / "grades.json").read_text())
    res = {"rank_weeks": int(preds.week.nunique()), "first_rank_week": str(preds.week.min().date()),
           "last_rank_week": str(preds.week.max().date()), "copies": K, "ensembles": {}, "cascade": {}}
    means = {}
    for name, copies in ENSEMBLES.items():
        rows, means[name] = ensemble_rows(sel, ctx, preds, copies)
        rows = rows.sort_values("week").reset_index(drop=True)
        s = sel.simulate(ctx, rows, **CHAMPION)
        spy = ctx.wret["SPY"].reindex(s.index)
        entry = {"copies": list(copies), "rows": int(len(rows)),
                 "windows": {w: {"config": sel.metrics(s, *b), "sp500": sel.metrics(spy, *b)} for w, b in WINDOWS.items()}}
        if name.startswith("copies 1-4"):
            ok, ties, n, why = same_rows(rows[rows.week < WINDOWS["2006_2015"][1]],
                                         v3[v3.week < WINDOWS["2006_2015"][1]], sel, ctx, means[name])
            entry["reproduces_v3"] = ok
            entry["reproduces_v3_up_to_ties"] = ties
            entry["common_weeks"] = n
            entry["weeks_differing"] = sum(why.values())
            entry["differing_weeks_by_cause"] = why
            log(f"copies 1-4 reproduce v3's walk row for row on 2006-2015: {ok} on {n} common weeks "
                f"({entry['weeks_differing']} differ: {why}; every difference explained: {ties})")
        res["ensembles"][name] = entry
        res["cascade"][name] = cascade_at(sel, ctx, rows)
        c = res["cascade"][name]
        log(f"{name}: " + "; ".join(f"{w} {gl(entry['windows'][w]['config'])}" for w in WINDOWS)
            + f" | cascade at this K: top-{c['book']} / {c['floor']} / stop {c['stop']} / cap {c['cap']}")
    res["agreement"] = pairs_agreement(means)
    (r["out"] / "grades.json").write_text(json.dumps(res, indent=1))
    ens, cas = res["ensembles"], res["cascade"]
    k4 = [n for n in ENSEMBLES if n.endswith("K=4)")]
    e14 = ens["copies 1-4 (record, K=4)"]
    md = ["# The champion at sixteen copies, 2006-01 -> 2024-07", "",
          f"Generated by `ops/k16_program.py grade champion_2006_2015` on {pd.Timestamp.today().date()}; the rule is the "
          "ledger's preregistration `k16_champion_2006_2015` (written before any number). The champion's model at its "
          "settings (4w / 5y / purge 35 / the generated bundle of 40), sixteen copies per rank week under replication.py's "
          f"seeding, 2006-01-06 -> 2015-12-31 (this walk) joined with the same copies on 2016-01-08 -> 2024-07-12 "
          f"(`k16_seed_spread_2016_2024`): {res['rank_weeks']} rank weeks, each copy's scores saved. Every ensemble is "
          "simulated from 2006-01 at the champion's settings (top-6 / cap 2 / no stop / 70-30 trend ballast) on the live "
          "ledger's rules and read on the arm's three windows (exclusive ends). Holdout untouched.", "",
          "## At the champion's settings", "",
          "| ensemble | 2006-2015: terminal $100 | Sharpe | DD | 2016-2024: terminal $100 | Sharpe | DD | 2006-2024: terminal $100 | CAGR | Sharpe | DD |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]

    def cells(w, key="config"):
        m = w[key]
        return f"${m['terminal_100']:,.0f} | {m['sharpe']:.2f} | {m['max_dd']:.0%}"
    for name, e in ens.items():
        w = e["windows"]
        p = w["pre_holdout"]["config"]
        md.append(f"| {name} | {cells(w['2006_2015'])} | {cells(w['2016_2024'])} | ${p['terminal_100']:,.0f} | "
                  f"{p['cagr_pct']:+.1f}% | {p['sharpe']:.2f} | {p['max_dd']:.0%} |")
    v = {w: v3_grades[w]["config"] for w in WINDOWS}
    md.append(f"| v3 record (K=4, one walk) | ${v['2006_2015']['terminal_100']:,.0f} | {v['2006_2015']['sharpe']:.2f} | "
              f"{v['2006_2015']['max_dd']:.0%} | ${v['2016_2024']['terminal_100']:,.0f} | {v['2016_2024']['sharpe']:.2f} | "
              f"{v['2016_2024']['max_dd']:.0%} | ${v['pre_holdout']['terminal_100']:,.0f} | {v['pre_holdout']['cagr_pct']:+.1f}% | "
              f"{v['pre_holdout']['sharpe']:.2f} | {v['pre_holdout']['max_dd']:.0%} |")
    w = ens["copies 1-16 (K=16)"]["windows"]
    p = w["pre_holdout"]["sp500"]
    md.append(f"| SPY | {cells(w['2006_2015'], 'sp500')} | {cells(w['2016_2024'], 'sp500')} | ${p['terminal_100']:,.0f} | "
              f"{p['cagr_pct']:+.1f}% | {p['sharpe']:.2f} | {p['max_dd']:.0%} |")
    t06 = [ens[n]["windows"]["2006_2015"]["config"]["terminal_100"] for n in k4]
    t16 = ens["copies 1-16 (K=16)"]["windows"]["2006_2015"]["config"]["terminal_100"]
    why = e14["differing_weeks_by_cause"]
    md += ["", f"Copies 1-4 reproduce v3's walk on 2006-2015 exactly on {e14['common_weeks'] - e14['weeks_differing']} of "
           f"{e14['common_weeks']} common weeks; every other week is explained without any score differing: "
           f"**{e14['reproduces_v3_up_to_ties']}** ({why['tie_order']} tie-order weeks, {why['delisted_in_horizon']} "
           f"delisting weeks, {why['unexplained']} unexplained; see reports/k16_clean_line.md).", "",
           f"2006-2015, four K=4 draws: ${min(t06):,.0f} to ${max(t06):,.0f} (mean ${np.mean(t06):,.0f}); K=16 ${t16:,.0f}. "
           "The 2016-2024 column differs slightly from `reports/k16_seed_spread.md` because each ensemble now enters 2016 "
           "with its own sleeves rather than v3's.", "",
           "## The cascade's book-down layers re-decided on 2006-2015 (selection window only)", "",
           "Same metrics and order as `selection.run_cascade`: book by cost-adjusted compounded %/yr (top-3 / top-6 / "
           "top-10), then floor, stop and cap by Sharpe at the picks above. The champion's standing settings are top-6 / "
           "70-30 / no stop / cap 2; the nested3 cascade of record (K=4, clean line) picked top-6 / 60-40 / stop -25% / cap 2.", "",
           "| ensemble | book (%/yr top-3 / top-6 / top-10) | floor (Sharpe none / halfgate / 80-20 / 70-30 / 60-40) | stop (SR none / -25%) | cap (SR none / 2) |",
           "|---|---|---|---|---|"]
    for name, c in cas.items():
        ev = c["evidence"]
        floors = " / ".join(f"{ev['floor'][f]:.3f}" for f in sel.FLOORS)
        md.append(f"| {name} | **top-{c['book']}** ({ev['book']['3']:.2f} / {ev['book']['6']:.2f} / {ev['book']['10']:.2f}) "
                  f"| **{c['floor']}** ({floors}) "
                  f"| **{c['stop']}** ({ev['stop']['None']:.3f} / {ev['stop']['-0.25']:.3f}) "
                  f"| **{c['cap']}** ({ev['cap']['None']:.3f} / {ev['cap']['2']:.3f}) |")
    md += ["", "Per the preregistration the champion's settings stand; a layer whose K=16 argmax differs is the owner's "
           "decision at step 3, and no alternative setting reads 2016-2024 unless adopted (then once).", "",
           "## How much two independent ensembles agree (week-averaged, 2006-2024)", "",
           "| ensembles | Spearman of scores | range | top-6 names shared | weeks |", "|---|---|---|---|---|"]
    for label, a in res["agreement"].items():
        md.append(f"| {label} | {a['spearman']:.2f} | {a['spearman_range'][0]:.2f}-{a['spearman_range'][1]:.2f} | "
                  f"{a['top6_shared']:.1f} of 6 | {a['weeks']} |")
    md.append("")
    r["report"].write_text("\n".join(md))
    print("\n".join(md))
    entries = []
    for name, e in ens.items():
        p, spm, c = e["windows"]["pre_holdout"]["config"], e["windows"]["pre_holdout"]["sp500"], cas[name]
        entries.append({
            "kind": "seed_replication", "name": f"k16_champion_{slug(name)}_2006_2024",
            "config": {**CHAMPION, "train_years": 5, "fills": "next open", "cost_bps_per_side": 5.0,
                       "copies": e["copies"], "features": list(ctx.extra)},
            "terminal_100": p["terminal_100"], "cagr_pct": p["cagr_pct"], "sharpe": p["sharpe"],
            "max_dd": p["max_dd"], "n_weeks": p["n_weeks"], "spy_terminal_100": spm["terminal_100"],
            "spy_cagr_pct": spm["cagr_pct"], "spy_sharpe": spm["sharpe"], "spy_max_dd": spm["max_dd"],
            "windows": {w: e["windows"][w]["config"] for w in WINDOWS},
            "cascade_2006_2015": {k: c[k] for k in ("book", "floor", "stop", "cap")},
            "notes": (f"The champion at sixteen copies (preregistration k16_champion_2006_2015): {name}, 2006-01 -> "
                      f"2024-07 at the champion's settings: " + "; ".join(
                          f"{w} {gl(e['windows'][w]['config'])}" for w in WINDOWS) + f"; SPY pre-holdout {gl(spm)}. "
                      f"Cascade book-down layers on its 2006-2015 holdings: top-{c['book']} / {c['floor']} / stop "
                      f"{c['stop']} / cap {c['cap']}. reports/k16_champion_record.md. Holdout untouched; no change to "
                      "the champion, the spec, K or the live job by this row.")})
    c16 = cas["copies 1-16 (K=16)"]
    entries.append({
        "kind": "verdict", "name": "k16_champion_2006_2015_verdict",
        "notes": (f"Outcome of preregistration k16_champion_2006_2015 ({pd.Timestamp.today().date()}, "
                  f"reports/k16_champion_record.md): (a) copies 1-4 reproduce v3 on 2006-2015 exactly on "
                  f"{e14['common_weeks'] - e14['weeks_differing']} of {e14['common_weeks']} weeks, every other week "
                  f"explained with no score differing: {e14['reproduces_v3_up_to_ties']} ({why}); (b) 2006_2015 K=4 draws ${min(t06):,.0f}-"
                  f"${max(t06):,.0f} (mean ${np.mean(t06):,.0f}), K=16 ${t16:,.0f}; K=16 over 2006-2024 "
                  f"{gl(ens['copies 1-16 (K=16)']['windows']['pre_holdout']['config'])} vs SPY {gl(spm)}, on 2016-2024 "
                  f"{gl(ens['copies 1-16 (K=16)']['windows']['2016_2024']['config'])}; (c) cascade at K=16 on 2006-2015: "
                  f"top-{c16['book']} / {c16['floor']} / stop {c16['stop']} / cap {c16['cap']} (evidence "
                  f"{json.dumps(c16['evidence'])}); the four K=4 draws pick "
                  + ", ".join(f"top-{cas[n]['book']}/{cas[n]['floor']}/stop {cas[n]['stop']}/cap {cas[n]['cap']}" for n in k4)
                  + ". The champion's settings (top-6 / 70-30 / no stop / cap 2) stand; any change is the owner's "
                  "decision at step 3, written down before the live change.")})
    log(f"ledger: {record_trials(entries)} rows")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["walk", "grade", "all"])
    ap.add_argument("run", choices=list(RUNS))
    a = ap.parse_args()
    if a.step in ("walk", "all"):
        log(f"== walk {a.run}")
        walk(a.run)
    if a.step in ("grade", "all"):
        log(f"== grade {a.run}")
        (grade_clean if a.run == "clean_2016_2024" else grade_champion)()
