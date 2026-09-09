"""The OpenFE arm: generated features with no human ideas, one grade.

Registered before any number (ledger preregistration openfe_arm_2006_2015,
2026-09-05). A third line for the nested test beside the clean champion and
the "+ engineered features*" bundle: formulas from a mechanical generator
(OpenFE's order-1 numeric space; features/generated.py) over inputs written
down as a rule, scored on the selection window 2006-2015 only, computed for
every panel row, walked once at the champion's settings and graded once on
2016-01 -> 2024-07-19. The holdout (2024-07-19+) is untouched.

  inputs    the raw inputs for every panel row (the 64 base features before
            ranking, every SF1 field as level and yoy, every 8-K item code as a
            26-week count, the close) -> inputs.parquet, inputs.json
  timing    2010's rows, TIMING_CANDIDATES random formulas: projects stage 1 on
            the full window -> timing.json; over BOUND_HOURS = stop and report
  fit       the base model (out-of-fold init scores), stage 1 on every candidate
            (rank per week, OpenFE's predictive metric), stage 2 on the base + the
            store inputs + the TOP_STAGE1 best -> the BUNDLE_SIZE columns with the
            most gain: formulas.json (stage1_scores.parquet, stage2_gains.csv)
  features  the bundle for every panel row, ranked per week -> generated.parquet
  walk      the champion's walk (4w / 5y / K=4) with ctx.extra = the bundle,
            2006-01 -> 2024-07-18 -> holdings_4w_5y_x<hash>_s0.parquet
  grade     selection.simulate at the champion's settings on the live rules;
            2016-01 -> 2024-07 beside the clean and ideas lines of record and
            SPY, 2006-2015 as in-sample -> grades.json, reports/openfe_arm.md,
            ledger rows

Run from the repo root with LightGBM available:
  PYTHONPATH=src:. uv run --with lightgbm python ops/openfe_arm.py {inputs,timing,fit,features,walk,grade,all}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from stocks_ml.selection import HOLDOUT_START

STORE = "data/sharadar_world2000"
OUT = Path("data/experiments/openfe_2006_2015")
REPORT = Path("reports/openfe_arm.md")
RECORD = Path("data/experiments/champion_2006_2024/champion_bundle_grades.json")
FIT_LO, FIT_HI = pd.Timestamp("2006-01-01"), pd.Timestamp("2015-12-31")
LABEL_DAYS = 35                                   # label_4w's span plus the purge margin
WALK_LO, WALK_HI = pd.Timestamp("2006-01-01"), HOLDOUT_START - pd.Timedelta(days=1)
HI = HOLDOUT_START - pd.Timedelta(days=1)         # last pre-holdout session
CHAMPION = dict(horizon="4w", book=6, cap=2, stop=None, floor="70/30")
WINDOWS = {"2016_2024": ("2016", HOLDOUT_START), "2006_2015": ("2006", "2016"),
           "pre_holdout": ("2006", HOLDOUT_START)}
SPAN = {"2016_2024": "2016-01 -> 2024-07 (the number)", "2006_2015": "2006-2015 (in-sample)",
        "pre_holdout": "2006-01 -> 2024-07"}
BLOCK_WEEKS, FOLDS, PURGE_DAYS = 26, 5, 35
TOP_STAGE1, BUNDLE_SIZE = 2000, 40
TIMING_CANDIDATES, TIMING_YEAR, BOUND_HOURS = 3000, 2010, 6.0
CHUNK = 50
N_JOBS = os.cpu_count() or 4


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- inputs
def inputs():
    from stocks_ml.config import load_config
    from stocks_ml.features import generated as gen
    from stocks_ml.features.panel import feature_cols
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frame = gen.raw_inputs(STORE, load_config(), log)
    names = gen.input_names(frame)
    base = [c for c in names if c.startswith("f_")]
    store = [c for c in names if c.startswith(gen.STORE_PREFIX)]
    import pyarrow.parquet as pq
    panel_feats = feature_cols(pd.DataFrame(columns=pq.read_schema(f"{STORE}/panel_sf.parquet").names))
    assert base == panel_feats, f"raw base features differ from panel_sf's: {set(base) ^ set(panel_feats)}"
    frame.to_parquet(OUT / "inputs.parquet", index=False)
    meta = {"rows": int(len(frame)), "first_week": str(frame["date"].min().date()),
            "last_week": str(frame["date"].max().date()), "n_inputs": len(names), "base": base,
            "store": store, "seconds": round(time.time() - t0)}
    (OUT / "inputs.json").write_text(json.dumps(meta, indent=1))
    log(f"inputs: {len(frame):,} rows x {len(names)} inputs ({len(base)} base, {len(store)} store) "
        f"in {time.time() - t0:.0f}s -> {OUT / 'inputs.parquet'}")


# ----------------------------------------------------------------------------- scoring data
def fit_rows(lo, hi):
    """The scoring rows: inputs in [lo, hi - LABEL_DAYS] with a label, sorted by
    date, joined with the label and the ranked base features from panel_sf."""
    from stocks_ml.features.panel import feature_cols
    inp = pd.read_parquet(OUT / "inputs.parquet")
    inp = inp[(inp["date"] >= lo) & (inp["date"] <= hi - pd.Timedelta(days=LABEL_DAYS))]
    pan = pd.read_parquet(f"{STORE}/panel_sf.parquet")
    feats = feature_cols(pan)
    pan = pan[["date", "ticker", "label_4w"] + feats]
    pan = pan[(pan["date"] >= lo) & (pan["date"] <= hi - pd.Timedelta(days=LABEL_DAYS))]
    pan.columns = ["date", "ticker", "label_4w"] + [f"base:{c}" for c in feats]
    rows = inp.merge(pan, on=["date", "ticker"], how="inner").dropna(subset=["label_4w"])
    rows = rows.sort_values(["date", "ticker"]).reset_index(drop=True)
    return rows, [f"base:{c}" for c in feats]


def prepare(rows, base_cols, path, n_jobs):
    """Folds, the base model's out-of-fold init scores, the stage-1 split
    (val = fold 0's blocks, purged train) -> the scoring file the workers read."""
    from stocks_ml.features import generated as gen
    y = rows["label_4w"].to_numpy(float)
    dates = rows["date"].to_numpy()
    fold = gen.week_blocks(dates, BLOCK_WEEKS, FOLDS)
    t0 = time.time()
    init = gen.init_scores(rows[base_cols].to_numpy(np.float32), y, dates, fold, PURGE_DAYS, n_jobs=n_jobs)
    val = fold == 0
    train = gen.purged_train(dates, fold, 0, PURGE_DAYS)
    rmse = gen.init_metric(init, y, val)
    base_rmse0 = float(np.sqrt(np.mean((y[val] - y[train].mean()) ** 2)))
    log(f"base model: {len(y):,} rows, {int(val.sum()):,} val / {int(train.sum()):,} train, "
        f"val RMSE {rmse:.5f} vs constant {base_rmse0:.5f} ({time.time() - t0:.0f}s)")
    data = rows.drop(columns=base_cols + ["label_4w"]).copy()
    data["y"], data["init"], data["fold"], data["train"], data["val"] = y, init, fold, train, val
    data.attrs = {}
    data.to_parquet(path, index=False)
    return {"rows": int(len(y)), "val_rows": int(val.sum()), "train_rows": int(train.sum()),
            "base_val_rmse": rmse, "constant_val_rmse": base_rmse0, "base_seconds": round(time.time() - t0)}


# ----------------------------------------------------------------------------- stage 1 (workers)
_W = {}


def _init_worker(path):
    os.environ["OMP_NUM_THREADS"] = "1"
    from stocks_ml.features import generated as gen
    d = pd.read_parquet(path)
    _W["inputs"] = d[gen.input_names(d.drop(columns=["y", "init", "fold", "train", "val"]))]
    _W["dates"] = d["date"].to_numpy()
    _W["y"], _W["init"] = d["y"].to_numpy(float), d["init"].to_numpy(float)
    _W["train"], _W["val"] = d["train"].to_numpy(bool), d["val"].to_numpy(bool)
    _W["rmse"] = gen.init_metric(_W["init"], _W["y"], _W["val"])


def _score_chunk(formulas):
    from stocks_ml.features import generated as gen
    out = []
    for f in formulas:
        t0 = time.time()
        try:
            r = gen.rank_week(gen.evaluate(f, _W["inputs"]), _W["dates"])
            s = gen.predictive_score(r, _W["y"], _W["init"], _W["train"], _W["val"], _W["rmse"])
        except Exception as e:                    # a degenerate column: scored as useless
            s = float("nan")
            print(f"  {f}: {e!r}", flush=True)
        out.append((f, s, time.time() - t0))
    return out


def stage1(path, formulas, n_jobs, label):
    """Every formula's stage-1 score, CHUNK formulas per task."""
    chunks = [formulas[i:i + CHUNK] for i in range(0, len(formulas), CHUNK)]
    res, t0, done = [], time.time(), 0
    with Pool(n_jobs, initializer=_init_worker, initargs=(str(path),)) as pool:
        for i, part in enumerate(pool.imap_unordered(_score_chunk, chunks)):
            res.extend(part)
            done += len(part)
            if (i + 1) % max(1, len(chunks) // 20) == 0 or i + 1 == len(chunks):
                el = time.time() - t0
                log(f"{label}: {done:,}/{len(formulas):,} in {el / 60:.1f} min "
                    f"(eta {el / done * (len(formulas) - done) / 60:.1f} min)")
    df = pd.DataFrame(res, columns=["formula", "score", "seconds"]).sort_values("score", ascending=False)
    return df.reset_index(drop=True), time.time() - t0


def candidates(rows):
    from stocks_ml.features import generated as gen
    names = [c for c in gen.input_names(rows) if not c.startswith("base:") and c != "label_4w"]
    constant = gen.week_constant(rows, names)
    store = [c for c in names if c.startswith(gen.STORE_PREFIX)]
    forms = gen.enumerate_candidates(names, constant, identity=store)
    return forms, names, sorted(constant)


# ----------------------------------------------------------------------------- timing
def timing():
    lo, hi = pd.Timestamp(f"{TIMING_YEAR}-01-01"), pd.Timestamp(f"{TIMING_YEAR}-12-31")
    rows, base_cols = fit_rows(lo, hi)
    full, _ = fit_rows(FIT_LO, FIT_HI)
    forms_full, names, constant = candidates(full)
    n_full, n_rows_full = len(forms_full), len(full)
    del full
    log(f"timing: {len(rows):,} rows of {TIMING_YEAR}; full window {n_rows_full:,} rows, "
        f"{n_full:,} candidates over {len(names)} inputs ({len(constant)} week-constant)")
    meta = prepare(rows, base_cols, OUT / "timing_data.parquet", N_JOBS)
    rng = np.random.default_rng(0)
    sample = [forms_full[i] for i in sorted(rng.choice(n_full, size=min(TIMING_CANDIDATES, n_full), replace=False))]
    scores, wall = stage1(OUT / "timing_data.parquet", sample, N_JOBS, "timing stage 1")
    per = wall / len(sample)
    proj = wall * (n_full / len(sample)) * (n_rows_full / len(rows)) / 3600
    res = {"year": TIMING_YEAR, "n_jobs": N_JOBS, **meta, "candidates_timed": len(sample),
           "seconds": round(wall, 1), "seconds_per_candidate_wall": round(per, 4),
           "seconds_per_candidate_cpu": round(float(scores["seconds"].mean()), 4),
           "full_rows": n_rows_full, "full_candidates": n_full, "inputs": len(names),
           "week_constant": constant, "projected_stage1_hours": round(proj, 2), "bound_hours": BOUND_HOURS,
           "verdict": "go" if proj <= BOUND_HOURS else "stop"}
    (OUT / "timing.json").write_text(json.dumps(res, indent=1))
    scores.to_parquet(OUT / "timing_scores.parquet", index=False)
    log(f"timing: {len(sample)} candidates in {wall / 60:.1f} min on {N_JOBS} workers "
        f"({per * 1000:.0f} ms each wall); full stage 1 projected {proj:.2f} h "
        f"(bound {BOUND_HOURS:.0f} h) -> {res['verdict'].upper()}")
    return res


# ----------------------------------------------------------------------------- fit
def fit():
    from stocks_ml.features import generated as gen
    rows, base_cols = fit_rows(FIT_LO, FIT_HI)
    forms, names, constant = candidates(rows)
    log(f"fit: {len(rows):,} rows {rows['date'].min().date()} -> {rows['date'].max().date()}, "
        f"{len(forms):,} candidates over {len(names)} inputs ({len(constant)} week-constant)")
    path = OUT / "stage1_data.parquet"
    if (OUT / "stage1_scores.parquet").exists():
        scores = pd.read_parquet(OUT / "stage1_scores.parquet")
        meta = json.loads((OUT / "fit.json").read_text())
        log(f"fit: stage 1 loaded from cache ({len(scores):,} scores)")
    else:
        meta = prepare(rows, base_cols, path, N_JOBS)
        scores, wall = stage1(path, forms, N_JOBS, "stage 1")
        scores.to_parquet(OUT / "stage1_scores.parquet", index=False)
        meta.update(candidates=len(forms), inputs=len(names), week_constant=constant,
                    stage1_seconds=round(wall), positive_scores=int((scores["score"] > 0).sum()))
        (OUT / "fit.json").write_text(json.dumps(meta, indent=1))
        log(f"stage 1: {len(scores):,} scored in {wall / 3600:.2f} h; "
            f"{meta['positive_scores']:,} positive; best {scores.iloc[0]['formula']} {scores.iloc[0]['score']:.6f}")
    # stage 2: the base (ranked, as the model sees it) + every store input + the best survivors
    d = pd.read_parquet(path)
    y, init = d["y"].to_numpy(float), d["init"].to_numpy(float)
    train, val = d["train"].to_numpy(bool), d["val"].to_numpy(bool)
    dates = d["date"].to_numpy()
    store = [c for c in names if c.startswith(gen.STORE_PREFIX)]
    keep = scores[scores["score"] > 0].head(TOP_STAGE1)["formula"].tolist()
    cols = list(dict.fromkeys(store + keep))
    log(f"stage 2: {len(base_cols)} base + {len(cols)} columns ({len(store)} store inputs, "
        f"{len(keep)} survivors) on {len(y):,} rows")
    t0 = time.time()
    X = np.empty((len(y), len(base_cols) + len(cols)), dtype=np.float32)
    X[:, :len(base_cols)] = rows[base_cols].to_numpy(np.float32)
    inp = d[names]
    for j, f in enumerate(cols):
        X[:, len(base_cols) + j] = gen.rank_week(gen.evaluate(f, inp), dates)
    log(f"stage 2: matrix {X.shape} built in {time.time() - t0:.0f}s")
    t0 = time.time()
    gains = gen.gain_ranking(X, base_cols + cols, y, init, train, val, n_jobs=N_JOBS)
    log(f"stage 2: fitted in {time.time() - t0:.0f}s")
    cand = gains[[not c.startswith("base:") for c in gains.index]]
    s1 = scores.set_index("formula")["score"]
    table = pd.DataFrame({"formula": cand.index, "gain": cand.values,
                          "stage1_score": [s1.get(f, np.nan) for f in cand.index]})
    table["share"] = table["gain"] / gains.sum()
    table.to_csv(OUT / "stage2_gains.csv", index=False)
    chosen = table[table["gain"] > 0].head(BUNDLE_SIZE)
    formulas = {f"{gen.PREFIX}{i:02d}": f for i, f in enumerate(chosen["formula"])}
    (OUT / "formulas.json").write_text(json.dumps(formulas, indent=1))
    base_share = float(gains[[c.startswith("base:") for c in gains.index]].sum() / gains.sum())
    meta.update(stage2_columns=len(cols), bundle=len(formulas), base_gain_share=round(base_share, 4),
                bundle_gain_share=round(float(chosen["gain"].sum() / gains.sum()), 4),
                stage2_best_iteration=gains.attrs.get("best_iteration"), stage2_val_rmse=gains.attrs.get("val_rmse"),
                stage2_positive=int((gains > 0).sum()))
    (OUT / "fit.json").write_text(json.dumps(meta, indent=1))
    log(f"bundle: {len(formulas)} formulas, {meta['bundle_gain_share']:.1%} of the gain "
        f"(base {base_share:.1%}); stage 2 kept {meta['stage2_best_iteration']} tree(s), "
        f"{meta['stage2_positive']} columns with gain -> {OUT / 'formulas.json'}")
    for n, f in list(formulas.items())[:10]:
        log(f"  {n} = {f}")


# ----------------------------------------------------------------------------- features
def features():
    from stocks_ml.features import generated as gen
    formulas = json.loads((OUT / "formulas.json").read_text())
    inp = pd.read_parquet(OUT / "inputs.parquet")
    t0 = time.time()
    out = gen.add_generated(inp[["date", "ticker"]], inp, formulas)
    out.to_parquet(OUT / "generated.parquet", index=False)
    log(f"features: {len(formulas)} g_ columns on {len(out):,} rows in {time.time() - t0:.0f}s")


def ctx_with_bundle():
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
    ctx.extra = list(formulas)
    log(f"ctx: panel {ctx.pan.shape}, {len(ctx.extra)} generated features"
        + (f", {missing} rows without inputs (neutral-filled)" if missing else ""))
    return sel, ctx, formulas


# ----------------------------------------------------------------------------- walk
def walk():
    sel, ctx, _ = ctx_with_bundle()
    stem = sel.holdings_name("4w", 5, ctx.extra)
    log(f"walk: {stem} on {WALK_LO.date()} -> {WALK_HI.date()} (4w / 5y / K={sel.K_COPIES})")
    t0 = time.time()
    sel.stage_holdings(ctx, str(OUT), "4w", 5, WALK_LO, WALK_HI)
    n = len(pd.read_parquet(OUT / f"{stem}_s0.parquet"))
    log(f"walk: {n} rank weeks in {(time.time() - t0) / 3600:.2f} h -> {OUT / (stem + '_s0.parquet')}")


# ----------------------------------------------------------------------------- grade
def weekly_ic(frame, cols):
    """Mean weekly Spearman IC with label_4w and its t statistic, per column."""
    out = {}
    for c in cols:
        s = frame.groupby("date")[[c, "label_4w"]].corr(method="spearman").xs("label_4w", level=1)[c].dropna()
        s = s[s.abs() < 1]
        out[c] = (float(s.mean()), float(s.mean() / s.std() * np.sqrt(len(s))), int(len(s)))
    return out


def notes(ctx, formulas, fitmeta, res, h):
    """What the reader should know beside the table: the stage-2 collapse, the
    selection-window IC of the columns, the skipped weeks, the walk's start."""
    from stocks_ml.features import generated as gen
    fit_hi = FIT_HI - pd.Timedelta(days=LABEL_DAYS)
    pan = ctx.pan[(ctx.pan.date >= FIT_LO) & (ctx.pan.date <= fit_hi)]
    ideas = [c for c in ("x_price_level", "f_log_mktcap") if c in pan.columns]
    ic = weekly_ic(pan, list(formulas) + ideas)
    weeks = [t for t in ctx.weeks if WALK_LO <= t <= min(WALK_HI, HI)]
    skipped = sorted(set(weeks) - set(pd.to_datetime(h.week)))
    md = ["## Notes", "",
          f"1. **Stage 2 kept {fitmeta.get('stage2_best_iteration')} tree.** With OpenFE's stage-2 parameters "
          f"(early stop {gen.STAGE2_STOP} on the validation fold) the gain model was worse than the base model from its "
          f"first tree (val RMSE {fitmeta.get('stage2_val_rmse', 0):.6f} vs {fitmeta.get('base_val_rmse', 0):.6f}), so "
          f"the gain ranking is that one tree's splits: {fitmeta.get('stage2_positive')} columns with any gain, "
          f"{len(formulas)} of them generated — the bundle is {len(formulas)}, not {BUNDLE_SIZE}. The base model itself "
          f"took almost nothing off a constant (val RMSE {fitmeta.get('base_val_rmse', 0):.6f} vs "
          f"{fitmeta.get('constant_val_rmse', 0):.6f}): on demedianed 4-week returns the signal is in the ranking, "
          "not the level, and an RMSE-driven selector is starved. The rule was not changed after seeing this.", "",
          f"2. **Selection-window IC** (weekly Spearman with label_4w, {FIT_LO.date()} -> {fit_hi.date()}; no grading "
          "year read):", "", "| column | formula | IC | t |", "|---|---|---|---|"]
    for c in sorted(formulas, key=lambda c: -abs(ic[c][1])):
        md.append(f"| {c} | `{formulas[c]}` | {ic[c][0]:+.4f} | {ic[c][1]:+.1f} |")
    for c in ideas:
        md.append(f"| {c} (for scale) | | {ic[c][0]:+.4f} | {ic[c][1]:+.1f} |")
    md += ["", "3. **Weeks.** The walk covers "
           f"{res['first_rank_week']} -> {res['last_rank_week']}, {res['rank_weeks']} rank weeks; {len(skipped)} weeks "
           "in range produced no ranking (the ensemble's predictions had fewer than 20 distinct values, the same "
           "skip rule as the records)"
           + (f": {', '.join(str(t.date()) for t in skipped)}" if skipped else "")
           + ". The ledger holds through a skipped week. The records' walks start 2001-06, so their 2006-2015 line "
           "begins fully invested while this walk fills its sleeves over its first weeks; the 2016 -> 2024-07 "
           "number is on the same basis (calendar-based sleeve rotation).", ""]
    return md


def gl(m):
    return f"${m['terminal_100']:,.0f} ({m['cagr_pct']:+.1f}%/yr, SR {m['sharpe']:.2f}, DD {m['max_dd']:.0%})"


def grade():
    from stocks_ml.models.trials import record_trials
    sel, ctx, formulas = ctx_with_bundle()
    stem = sel.holdings_name("4w", 5, ctx.extra)
    h = pd.read_parquet(OUT / f"{stem}_s0.parquet")
    h["week"] = pd.to_datetime(h["week"])
    h = h[h.week < HI].sort_values("week")
    s = sel.simulate(ctx, h, **CHAMPION)
    spy = ctx.wret["SPY"].reindex(s.index)
    res = {"rank_weeks": int(len(h)), "first_rank_week": str(h.week.min().date()),
           "last_rank_week": str(h.week.max().date()), "features": formulas, "holdings": f"{stem}_s0.parquet"}
    for w, (a, b) in WINDOWS.items():
        res[w] = {"config": sel.metrics(s, pd.Timestamp(a), pd.Timestamp(b)),
                  "sp500": sel.metrics(spy, pd.Timestamp(a), pd.Timestamp(b))}
    (OUT / "grades.json").write_text(json.dumps(res, indent=1))
    rec = json.loads(RECORD.read_text())
    gains = pd.read_csv(OUT / "stage2_gains.csv")
    fitmeta = json.loads((OUT / "fit.json").read_text())
    tim = json.loads((OUT / "timing.json").read_text()) if (OUT / "timing.json").exists() else {}
    md = ["# The OpenFE arm: generated features, one grade", "",
          f"Generated by `ops/openfe_arm.py` on {pd.Timestamp.today().date()}; the rule is the ledger's "
          "preregistration `openfe_arm_2006_2015` (written before any number). A mechanical generator "
          "(OpenFE's order-1 numeric space: every unordered pair of inputs under + - * / min max, plus abs) "
          f"over {fitmeta.get('inputs')} inputs written down as a rule — the champion's "
          "64 model features before ranking, every SF1 field (trailing 12 months) as level and year-over-year "
          "change, every 8-K item code as a 26-week count, the close — scored on 2006-2015 only: "
          f"{fitmeta.get('candidates', 0):,} candidates, ranked per week and scored against the base model's "
          f"residual (OpenFE's predictive metric), the {TOP_STAGE1} best plus every store input into one "
          f"LightGBM beside the 64 base features, the {len(formulas)} columns with the most gain kept. "
          "No human idea entered; no screen stage; no cascade re-selection. The walk is the champion's "
          f"(4w / 5y / K=4) with the bundle as extra features, {res['first_rank_week']} -> {res['last_rank_week']}; "
          "graded by `selection.simulate` at the champion's settings (top-6 / 70-30 trend ballast / no stop / "
          "cap 2) on the live ledger's rules. The clean and ideas lines are the records in "
          "`champion_bundle_grades.json` (walks from 2001-06); SPY on this walk's weeks. Holdout untouched.", "",
          "## Champion settings", "",
          "| window | clean (record) | + ideas* (record) | + generated | SPY |", "|---|---|---|---|---|"]
    for w in WINDOWS:
        md.append(f"| {SPAN[w]} | {gl(rec['clean_recorded'][w]['config'])} | {gl(rec['bundle_recorded'][w]['config'])} "
                  f"| {gl(res[w]['config'])} | {gl(res[w]['sp500'])} |")
    md += ["", "The asterisk: the ideas bundle's thirty candidates were written after reading the whole "
           "2006-2024 record. The generated line carries no asterisk — the caveat recorded in the "
           "preregistration is that the input rule was written by someone who knows those ideas exist; the "
           "rule is generic (every field, one standard window).", "",
           f"## The bundle ({len(formulas)} formulas, {fitmeta.get('bundle_gain_share', 0):.1%} of the stage-2 gain; "
           f"the 64 base features {fitmeta.get('base_gain_share', 0):.1%})", "",
           "| column | formula | stage-2 gain share | stage-1 score |", "|---|---|---|---|"]
    gi = gains.set_index("formula")
    for n, f in formulas.items():
        md.append(f"| {n} | `{f}` | {gi.loc[f, 'share']:.2%} | {gi.loc[f, 'stage1_score']:.2e} |")
    md += ["", "## Cost", "",
           f"Timing run ({tim.get('year')}: {tim.get('candidates_timed')} candidates, "
           f"{tim.get('seconds_per_candidate_wall', 0) * 1000:.0f} ms each wall on {tim.get('n_jobs')} workers) "
           f"projected {tim.get('projected_stage1_hours')} h for stage 1 against the {BOUND_HOURS:.0f} h bound; "
           f"stage 1 took {fitmeta.get('stage1_seconds', 0) / 3600:.2f} h, "
           f"{fitmeta.get('positive_scores', 0):,} of {fitmeta.get('candidates', 0):,} candidates scored above zero.", ""]
    md += notes(ctx, formulas, fitmeta, res, h)
    REPORT.write_text("\n".join(md))
    print("\n".join(md))
    m, sp = res["2016_2024"]["config"], res["2016_2024"]["sp500"]
    ins = res["2006_2015"]["config"]
    record_trials([{
        "kind": "r4w_strategy", "name": "openfe_arm_generated_2016_2024",
        "config": {**CHAMPION, "train_years": 5, "fills": "next open", "cost_bps_per_side": 5.0,
                   "features": list(formulas), "formulas": formulas},
        "terminal_100": m["terminal_100"], "cagr_pct": m["cagr_pct"], "sharpe": m["sharpe"],
        "max_dd": m["max_dd"], "n_weeks": m["n_weeks"], "spy_terminal_100": sp["terminal_100"],
        "spy_cagr_pct": sp["cagr_pct"], "spy_sharpe": sp["sharpe"], "spy_max_dd": sp["max_dd"],
        "notes": (f"The OpenFE arm's one grade (preregistration openfe_arm_2006_2015): the champion's settings "
                  f"with {len(formulas)} generated features (no human ideas; formulas in config), 2016-01 -> "
                  f"2024-07: {gl(m)} vs clean {gl(rec['clean_recorded']['2016_2024']['config'])} vs + ideas* "
                  f"{gl(rec['bundle_recorded']['2016_2024']['config'])} vs SPY {gl(sp)}; 2006-2015 in-sample "
                  f"{gl(ins)} vs clean {gl(rec['clean_recorded']['2006_2015']['config'])} vs + ideas* "
                  f"{gl(rec['bundle_recorded']['2006_2015']['config'])}. Walk {res['holdings']} "
                  f"{res['first_rank_week']} -> {res['last_rank_week']}; reports/openfe_arm.md. Holdout untouched; "
                  "no change to the champion, the spec or the live job.")}])
    log(f"grade: 2016-01 -> 2024-07 generated {gl(m)} | clean {gl(rec['clean_recorded']['2016_2024']['config'])} "
        f"| ideas* {gl(rec['bundle_recorded']['2016_2024']['config'])} | SPY {gl(sp)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["inputs", "timing", "fit", "features", "walk", "grade", "all"])
    step = ap.parse_args().step
    for name, fn in (("inputs", inputs), ("timing", timing), ("fit", fit), ("features", features),
                     ("walk", walk), ("grade", grade)):
        if step in (name, "all"):
            log(f"== {name}")
            res = fn()
            if name == "timing" and step == "all" and res["verdict"] != "go":
                log("timing: projection over the bound -> stopping before the fit (report to the owner)")
                sys.exit(3)
