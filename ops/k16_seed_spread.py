"""The champion's seed spread: sixteen copies on 2016-01 -> 2024-07, one walk.

Registered before any number (ledger preregistration k16_seed_spread_2016_2024,
2026-09-06; owner's go the same day). The champion averages K=4 week-bootstrap
copies (selection.ensemble_preds; replication.py's standard). Copy-to-copy
rank correlation at one Friday is 0.12-0.18, so the $863 of record on
2016-01 -> 2024-07 (openfe_v3_2006_2015) is one draw of the dice. This walk
fits SIXTEEN copies of the champion's model on every rank week of that span
and saves each copy's predictions, so one run answers three questions:

  copies 1-4        = v3's walk row for row (a code-path check: $863.4)
  5-8, 9-12, 13-16  three more K=4 draws: the spread the champion's number sits in
  1-8, 9-16, 1-16   what K=8 and K=16 buy

Every ensemble starts 2016 from v3's own pre-2016 holdings rows (the same
sleeve state), is simulated at the champion's settings (top-6 / cap 2 / no
stop / 70-30 trend ballast) on the live ledger's rules and graded on
2016-01-01 -> 2025-01-01 (exclusive; rank weeks end 2024-07-12, the holdout
is untouched) beside the clean and + ideas* records and SPY. Nothing here
changes the champion, the spec, K or the live job: raising K is a protocol
change and the owner's decision on compute.

Run from the repo root:
  PYTHONPATH=src:. .venv/bin/python ops/k16_seed_spread.py {walk,grade,curve,all}
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

STORE = "data/sharadar_world2000"
OUT = Path("data/experiments/k16_seed_spread_2016_2024")
REPORT = Path("reports/k16_seed_spread.md")
V3 = Path("data/experiments/openfe_v3_2006_2015")
V3_HOLDINGS = V3 / "holdings_4w_5y_x724d05_s0.parquet"
RECORD = Path("data/experiments/champion_2006_2024/champion_bundle_grades.json")
LO, HI = pd.Timestamp("2016-01-01"), pd.Timestamp("2024-07-18")   # holdout starts 2024-07-19
GRADE = (pd.Timestamp("2016"), pd.Timestamp("2025"))              # the arm's 2016_2024 window
K = 16
CHAMPION = dict(horizon="4w", book=6, cap=2, stop=None, floor="70/30")
ENSEMBLES = {"copies 1-4 (v3, K=4)": range(1, 5), "copies 5-8 (K=4)": range(5, 9),
             "copies 9-12 (K=4)": range(9, 13), "copies 13-16 (K=4)": range(13, 17),
             "copies 1-8 (K=8)": range(1, 9), "copies 9-16 (K=8)": range(9, 17),
             "copies 1-16 (K=16)": range(1, 17)}
CHECKPOINT = 10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def context():
    import stocks_ml.selection as sel
    from stocks_ml.features.bundle import FEATURES
    ctx = sel.Ctx(STORE)
    ctx.extra = list(FEATURES)           # the champion's bundle: panel g_ columns == v3's overlay, bit for bit
    missing = [c for c in ctx.extra if c not in ctx.pan.columns]
    if missing:
        raise RuntimeError(f"panel lacks {missing[:3]}...")
    return sel, ctx


def copy_preds(sel, ctx, t, c, h, cfg2):
    """One copy of the champion's model at rank week t, exactly as
    selection.ensemble_preds fits copy c."""
    from stocks_ml.models.walk import walk_forward_predictions
    from stocks_ml.models.xgb import TimeTailEarlyStopXGB
    from stocks_ml.models.replication import WeekBootstrapEstimator
    est = WeekBootstrapEstimator(TimeTailEarlyStopXGB(**sel.MODEL_PARAMS, **sel.fixed(h["purge"])),
                                 bootstrap_seed=c)
    wf = walk_forward_predictions(ctx.pan, est, cfg2, start=t, end=t, label_col=h["label"],
                                  purge_days=h["purge"], extra_features=tuple(ctx.extra))
    return wf.preds.get(t)


# ----------------------------------------------------------------------------- walk
def walk():
    sel, ctx = context()
    OUT.mkdir(parents=True, exist_ok=True)
    h = sel.HORIZONS["4w"]
    cfg2 = ctx.world_cfg(5)
    weeks = [t for t in ctx.weeks if LO <= t <= HI]
    path = OUT / "preds.parquet"
    frames, done = [], set()
    if path.exists():
        old = pd.read_parquet(path)
        old["week"] = pd.to_datetime(old["week"])
        frames.append(old)
        done = set(old["week"].unique())
    todo = [t for t in weeks if t not in done]
    log(f"walk: {len(weeks)} rank weeks {weeks[0].date()} -> {weeks[-1].date()}, {len(todo)} to do, "
        f"K={K} copies each; panel {ctx.pan.shape}, extra = {len(ctx.extra)} generated")
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
    log(f"walk: done in {(time.time() - t0) / 3600:.2f} h -> {path}")


# ----------------------------------------------------------------------------- grade
def gl(m):
    return f"${m['terminal_100']:,.0f} ({m['cagr_pct']:+.1f}%/yr, SR {m['sharpe']:.2f}, DD {m['max_dd']:.0%})"


def ensemble_rows(sel, ctx, preds, copies):
    """slice_row for the mean of the given copies at every week, as
    ensemble_preds would rank them (mean over copies, >= 20 distinct)."""
    cols = [f"c{c}" for c in copies]
    rows, means = [], {}
    for t, g in preds.groupby("week"):
        p = g.set_index("ticker")[[c for c in cols if c in g.columns]].mean(axis=1)
        if p.nunique() < 20:
            continue
        row = sel.slice_row(ctx, t, "4w", p)
        if row is not None:
            rows.append(row)
            means[t] = p
    return pd.DataFrame(rows), means


def agreement(a: dict, b: dict):
    """Week-averaged Spearman between two ensembles' scores and their top-6
    overlap, over the weeks both ranked."""
    rho, top6 = [], []
    for t in sorted(set(a) & set(b)):
        x, y = a[t].align(b[t], join="inner")
        rho.append(x.corr(y, method="spearman"))
        top6.append(len(set(x.nlargest(6).index) & set(y.nlargest(6).index)))
    return float(np.mean(rho)), float(np.mean(top6)), len(rho)


def grade():
    from stocks_ml.models.trials import record_trials
    sel, ctx = context()
    preds = pd.read_parquet(OUT / "preds.parquet")
    preds["week"] = pd.to_datetime(preds["week"])
    v3 = pd.read_parquet(V3_HOLDINGS)
    v3["week"] = pd.to_datetime(v3["week"])
    prefix = v3[v3.week < LO].sort_values("week")
    v3_span = v3[(v3.week >= LO) & (v3.week < HI)].sort_values("week").reset_index(drop=True)
    rec = json.loads(RECORD.read_text())
    res = {"rank_weeks": int(preds.week.nunique()), "first_rank_week": str(preds.week.min().date()),
           "last_rank_week": str(preds.week.max().date()), "copies": K, "ensembles": {}}
    means = {}
    for name, copies in ENSEMBLES.items():
        rows, means[name] = ensemble_rows(sel, ctx, preds, copies)
        hold = pd.concat([prefix, rows], ignore_index=True).sort_values("week")
        hold = hold[hold.week < HI]
        s = sel.simulate(ctx, hold, **CHAMPION)
        spy = ctx.wret["SPY"].reindex(s.index)
        m = sel.metrics(s, *GRADE)
        entry = {"copies": list(copies), "config": m, "sp500": sel.metrics(spy, *GRADE), "rows": int(len(rows))}
        if name.startswith("copies 1-4"):
            same = rows.reset_index(drop=True)
            cmp = same[["week", "top15"]].equals(v3_span[["week", "top15"]]) and np.allclose(
                same[["top3", "top6", "top10"]].to_numpy(), v3_span[["top3", "top6", "top10"]].to_numpy())
            entry["reproduces_v3"] = bool(cmp)
            entry["v3_terminal_100"] = json.loads((V3 / "grades.json").read_text())["2016_2024"]["config"]["terminal_100"]
            log(f"copies 1-4 reproduce v3 row for row: {cmp} ({len(same)} vs {len(v3_span)} rows)")
        res["ensembles"][name] = entry
        log(f"{name}: {gl(m)} on {m['n_weeks']} weeks; SPY {gl(entry['sp500'])}")
    k4 = [n for n in ENSEMBLES if n.endswith("K=4)")]
    pairs = {"K=4 vs K=4 (6 disjoint pairs)": [(a, b) for a, b in itertools.combinations(k4, 2)],
             "K=8 vs K=8 (1 disjoint pair)": [("copies 1-8 (K=8)", "copies 9-16 (K=8)")]}
    res["agreement"] = {}
    for label, ps in pairs.items():
        vals = [agreement(means[a], means[b]) for a, b in ps]
        res["agreement"][label] = {"spearman": float(np.mean([v[0] for v in vals])),
                                   "spearman_range": [float(min(v[0] for v in vals)), float(max(v[0] for v in vals))],
                                   "top6_shared": float(np.mean([v[1] for v in vals])), "weeks": vals[0][2]}
    (OUT / "grades.json").write_text(json.dumps(res, indent=1))
    ens = res["ensembles"]
    k4_terms = [ens[n]["config"]["terminal_100"] for n in k4]
    clean, ideas = rec["clean_recorded"]["2016_2024"]["config"], rec["bundle_recorded"]["2016_2024"]["config"]
    md = ["# The champion's seed spread: sixteen copies, 2016-01 -> 2024-07", "",
          f"Generated by `ops/k16_seed_spread.py` on {pd.Timestamp.today().date()}; the rule is the ledger's "
          "preregistration `k16_seed_spread_2016_2024` (written before any number). The champion's model at its "
          "settings (4w / 5y / purge 35 / the generated bundle of 40) fitted sixteen times per rank week under "
          "replication.py's seeding (copy c: random_state c + whole-week bootstrap seeded by c), "
          f"{res['first_rank_week']} -> {res['last_rank_week']} ({res['rank_weeks']} weeks) on the research store; "
          "each copy's scores saved. Every ensemble below starts 2016 from v3's own pre-2016 holdings rows, is "
          "simulated at the champion's settings (top-6 / cap 2 / no stop / 70-30 trend ballast) on the live "
          "ledger's rules and graded on 2016-01-01 -> 2025-01-01 (exclusive). Holdout untouched. Nothing here "
          "changes the champion, the spec, K or the live job.", "",
          "## 2016-01 -> 2024-07 at the champion's settings", "",
          "| ensemble | terminal $100 | CAGR | Sharpe | max DD | weeks |", "|---|---|---|---|---|---|"]
    for name, e in ens.items():
        m = e["config"]
        md.append(f"| {name} | ${m['terminal_100']:,.0f} | {m['cagr_pct']:+.1f}% | {m['sharpe']:.2f} | "
                  f"{m['max_dd']:.0%} | {m['n_weeks']} |")
    for label, m in (("clean (record, K=4)", clean), ("+ ideas* (record, K=4)", ideas)):
        md.append(f"| {label} | ${m['terminal_100']:,.0f} | {m['cagr_pct']:+.1f}% | {m['sharpe']:.2f} | "
                  f"{m['max_dd']:.0%} | {m['n_weeks']} |")
    sp = ens["copies 1-16 (K=16)"]["sp500"]
    md.append(f"| SPY (this walk's weeks) | ${sp['terminal_100']:,.0f} | {sp['cagr_pct']:+.1f}% | {sp['sharpe']:.2f} | "
              f"{sp['max_dd']:.0%} | {sp['n_weeks']} |")
    md += ["", f"Copies 1-4 reproduce v3's walk row for row: **{ens['copies 1-4 (v3, K=4)']['reproduces_v3']}** "
           f"(v3's record ${ens['copies 1-4 (v3, K=4)']['v3_terminal_100']:,.1f}).", "",
           f"Four K=4 draws: ${min(k4_terms):,.0f} to ${max(k4_terms):,.0f} (mean ${np.mean(k4_terms):,.0f}); "
           f"the two K=8 draws ${ens['copies 1-8 (K=8)']['config']['terminal_100']:,.0f} and "
           f"${ens['copies 9-16 (K=8)']['config']['terminal_100']:,.0f}; K=16 "
           f"${ens['copies 1-16 (K=16)']['config']['terminal_100']:,.0f}.", "",
           "## How much two independent ensembles agree (week-averaged)", "",
           "| ensembles | Spearman of scores | range | top-6 names shared | weeks |", "|---|---|---|---|---|"]
    for label, a in res["agreement"].items():
        md.append(f"| {label} | {a['spearman']:.2f} | {a['spearman_range'][0]:.2f}-{a['spearman_range'][1]:.2f} | "
                  f"{a['top6_shared']:.1f} of 6 | {a['weeks']} |")
    md += ["", "The clean line's own seed spread is unmeasured (same-bar caveat). The champion's settings were "
           "selected at K=4; raising K applies to every finalist alike and is the owner's decision on compute "
           "(the weekly job's fit is ~6 s per copy). A K=16 number is not a champion record unless K=16 is "
           "adopted, in which case the record basis becomes a K=16 walk over 2006-2024.", ""]
    REPORT.write_text("\n".join(md))
    print("\n".join(md))
    entries = []
    for name, e in ens.items():
        m, spm = e["config"], e["sp500"]
        slug = name.split(" (")[0].replace(" ", "_").replace("-", "_")
        entries.append({
            "kind": "seed_replication", "name": f"k16_seed_spread_{slug}_2016_2024",
            "config": {**CHAMPION, "train_years": 5, "fills": "next open", "cost_bps_per_side": 5.0,
                       "copies": e["copies"], "features": list(ctx.extra)},
            "terminal_100": m["terminal_100"], "cagr_pct": m["cagr_pct"], "sharpe": m["sharpe"],
            "max_dd": m["max_dd"], "n_weeks": m["n_weeks"], "spy_terminal_100": spm["terminal_100"],
            "spy_cagr_pct": spm["cagr_pct"], "spy_sharpe": spm["sharpe"], "spy_max_dd": spm["max_dd"],
            "notes": (f"Seed spread of the champion (preregistration k16_seed_spread_2016_2024): {name} of "
                      f"sixteen, 2016-01 -> 2024-07: {gl(m)} vs SPY {gl(spm)}; the four K=4 draws "
                      f"${min(k4_terms):,.0f}-${max(k4_terms):,.0f}, K=16 "
                      f"${ens['copies 1-16 (K=16)']['config']['terminal_100']:,.0f}; clean record {gl(clean)}, "
                      f"+ ideas* record {gl(ideas)}. reports/k16_seed_spread.md. Holdout untouched; no change to "
                      "the champion, the spec, K or the live job.")})
    log(f"ledger: {record_trials(entries)} rows")
    return res


# ----------------------------------------------------------------------------- curve (exploratory)
def curve(n_subsets=12, ks=(2, 4, 8, 12), seed=0):
    """Not preregistered: the same saved predictions re-averaged over random
    subsets of the sixteen copies, to show how the graded result moves with
    K (mean and range over subsets; K=16 is the one full set)."""
    from stocks_ml.models.trials import record_trials
    sel, ctx = context()
    preds = pd.read_parquet(OUT / "preds.parquet")
    preds["week"] = pd.to_datetime(preds["week"])
    v3 = pd.read_parquet(V3_HOLDINGS)
    v3["week"] = pd.to_datetime(v3["week"])
    prefix = v3[v3.week < LO].sort_values("week")
    rng = np.random.default_rng(seed)
    res = {}
    for k in list(ks) + [K]:
        subsets = [tuple(range(1, K + 1))] if k == K else \
            [tuple(sorted(rng.choice(np.arange(1, K + 1), size=k, replace=False).tolist())) for _ in range(n_subsets)]
        vals = []
        for sub in subsets:
            rows, _ = ensemble_rows(sel, ctx, preds, sub)
            hold = pd.concat([prefix, rows], ignore_index=True).sort_values("week")
            m = sel.metrics(sel.simulate(ctx, hold[hold.week < HI], **CHAMPION), *GRADE)
            vals.append({"copies": list(sub), **m})
        t = [v["terminal_100"] for v in vals]
        sr = [v["sharpe"] for v in vals]
        res[str(k)] = {"n": len(vals), "terminal_mean": float(np.mean(t)), "terminal_min": float(min(t)),
                       "terminal_max": float(max(t)), "sharpe_mean": float(np.mean(sr)),
                       "sharpe_min": float(min(sr)), "sharpe_max": float(max(sr)), "subsets": vals}
        log(f"K={k}: {len(vals)} subsets, terminal ${np.mean(t):,.0f} (${min(t):,.0f}-${max(t):,.0f}), "
            f"Sharpe {np.mean(sr):.2f} ({min(sr):.2f}-{max(sr):.2f})")
    (OUT / "curve.json").write_text(json.dumps(res, indent=1))
    md = ["", "## Exploratory: how the result moves with K (not preregistered)", "",
          f"The same saved predictions re-averaged over {n_subsets} random subsets of the sixteen copies at each K "
          "(K=16 is the one full set), graded exactly as above.", "",
          "| copies averaged | subsets | terminal $100: mean (range) | Sharpe: mean (range) |", "|---|---|---|---|"]
    for k, r in res.items():
        md.append(f"| K={k} | {r['n']} | ${r['terminal_mean']:,.0f} (${r['terminal_min']:,.0f}-${r['terminal_max']:,.0f}) "
                  f"| {r['sharpe_mean']:.2f} ({r['sharpe_min']:.2f}-{r['sharpe_max']:.2f}) |")
    md.append("")
    text = REPORT.read_text()
    marker = "\n## Exploratory: how the result moves with K"
    if marker in text:
        text = text[:text.index(marker)]
    REPORT.write_text(text.rstrip("\n") + "\n" + "\n".join(md))
    print("\n".join(md))
    record_trials([{
        "kind": "seed_replication", "name": "k16_seed_spread_k_curve_2016_2024",
        "config": {**CHAMPION, "train_years": 5, "features": list(ctx.extra), "subsets_per_k": n_subsets},
        "notes": ("Exploratory (not preregistered) companion to k16_seed_spread_2016_2024: the sixteen saved copies "
                  "re-averaged over random subsets, 2016-01 -> 2024-07 at the champion's settings; terminal $100 "
                  "mean (range) by K: " + "; ".join(f"K={k} ${r['terminal_mean']:,.0f} (${r['terminal_min']:,.0f}-"
                                                     f"${r['terminal_max']:,.0f}, n={r['n']})" for k, r in res.items())
                  + ". reports/k16_seed_spread.md. No change to the champion, the spec, K or the live job.")}])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["walk", "grade", "curve", "all"])
    step = ap.parse_args().step
    for name, fn in (("walk", walk), ("grade", grade), ("curve", curve)):
        if step in (name, "all"):
            log(f"== {name}")
            fn()
