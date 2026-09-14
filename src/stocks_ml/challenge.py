"""`stocks-ml challenge`: the challenger protocol as one process.

A candidate is a recipe — label, training window, extra panel columns,
overrides of selection.MODEL_PARAMS — anything `stocks-ml train` can walk. A
new label, a shorter window, a set of engineered columns, a learning rate:
each is a recipe, walked and judged the same way. The rules are the
registered ones (reports/clean_improvement_registration.md: rules 2, 5, 7,
8) and every constant lives at the top of this file:

  stage 1  every SAMPLE_EVERY-th week of the selection window at K=SAMPLE_K
           for each candidate; the incumbent's own walk cut to the same
           weeks and copies. A sample ranks only: the top ADVANCE candidates
           go on, nothing is decided.
  stage 2  every week of the selection window at K=FULL_K; the MODEL SCORE —
           the mean over the top-3, top-6 and top-10 books of the selection
           metric (backtest.selection_metric: cost-adjusted compounded %/yr,
           held 4 weeks; the same three numbers the procedure's book layer
           takes the argmax of) — on the weeks every walk shares, incumbent
           included; the paired weekly t of the three-book average return
           and the single-copy spread reported, never gated; the leak audit
           on every candidate. The argmax of the score is the frozen model;
           if it is the incumbent, the incumbent stands. (Owner's rule,
           2026-09-14: one book would miss a recipe that shines at another
           size; the mean over the menu is steadier than any one book and
           cannot be won by a lucky spike.)
  stage 3  (--k16) the winning candidate at K=16 on both segments, then
           `stocks-ml eval --incumbent`: the one look at 2016-2024 with the
           falsification test. Adoption is `stocks-ml procedure`, on the
           owner's go — never here.

Every stage records a ledger row (kind `challenge`); <out>/challenge.json
holds the whole record. Walks resume, so a rerun picks up where it stopped.

    stocks-ml challenge --out data/experiments/challenges/labels \\
        --candidate label=label_4w_sector_rank --candidate label=label_4w_sector_log
    stocks-ml challenge --out ... --candidate train_years=5 \\
        --candidate "features=x_a+x_b" --candidate "params=learning_rate:0.01+n_estimators:300"

`stocks-ml challenge-fast` is the prototype: the same candidates on a
stratified random sample of the selection window (FAST_PER_YEAR weeks from
each year, seed FAST_SEED) at K=SAMPLE_K, against the incumbent cut to the
same weeks and copies — minutes per candidate, for trying any change
(labels, windows, features, params) before spending a full challenge on
it. It ranks; it decides nothing; it prints the `challenge` line for
whatever beat the incumbent. Strategy layers need no walk at all: prototype
a menu change with `stocks-ml backtest --book/--floor/--stop/--cap` on the
champion's saved walk; `procedure` decides them.

A feature package plugs in upstream: its columns must be in the panel,
point-in-time, built by the panel code the live job runs every Saturday;
any screen that picked them must have read the selection window only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.backtest import holdings, load_preds, paired_t, selection_metric
from stocks_ml.leak_audit import audit_segments, leak_line
from stocks_ml.models.trials import record_trials
from stocks_ml.train import STORE, context, recipe, walk

SELECT = (pd.Timestamp("2006-01-01"), pd.Timestamp("2015-12-31"))
SAMPLE_EVERY = 4          # stage 1: every 4th rank week
SAMPLE_K = 4              # stage 1: copies
FULL_K = 4                # stage 2: copies, every week
ADVANCE = 2               # candidates that go from stage 1 to stage 2
SCORE_BOOKS = (3, 6, 10)  # the model score: the mean of the selection metric over these books
FINAL_K = 16              # stage 3: the deployed ensemble size
FAST_PER_YEAR = 13        # challenge-fast: weeks drawn from each year of the window
FAST_SEED = 0             # ... with this seed (the same weeks for every candidate)
RECIPE_KEYS = ("label", "train_years", "features", "params")


def parse_candidate(text: str, base: dict) -> dict:
    """'label=...,train_years=5,features=a+b,params=learning_rate:0.01+max_depth:4'
    over the incumbent's recipe as the base: unspecified fields are inherited."""
    r = {"label": base["label"], "train_years": base["train_years"],
         "features": list(base.get("features") or []), "params": dict(base.get("params") or {})}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"candidate field {part!r} is not key=value")
        k, v = [x.strip() for x in part.split("=", 1)]
        if k not in RECIPE_KEYS:
            raise ValueError(f"candidate field {k!r} is not one of {RECIPE_KEYS}")
        if k == "label":
            r["label"] = v
        elif k == "train_years":
            r["train_years"] = int(v)
        elif k == "features":
            r["features"] = [f for f in v.split("+") if f]
        else:
            r["params"] = {}
            for kv in [x for x in v.split("+") if x]:
                if ":" not in kv:
                    raise ValueError(f"param {kv!r} is not name:value")
                pk, pv = kv.split(":", 1)
                r["params"][pk.strip()] = pv.strip()
    return recipe(r["label"], r["train_years"], r["features"], r["params"])


def candidate_name(rec: dict) -> str:
    """A directory name for a recipe: label and window, plus a short hash of
    any features or params."""
    name = f"{rec['label']}_{rec['train_years']}y"
    extra = {k: rec[k] for k in ("features", "params") if rec.get(k)}
    if extra:
        name += "_" + hashlib.sha256(json.dumps(extra, sort_keys=True).encode()).hexdigest()[:8]
    return name


def incumbent_recipe(preds_path: Path) -> dict:
    rec = json.loads((Path(preds_path).parent / "spec.json").read_text())
    if not rec.get("recipe"):
        raise SystemExit(f"{preds_path} has no recipe in its record; the challenge needs the "
                         "incumbent's label and window")
    return rec["recipe"]


def sample_weeks(ctx, lo, hi, every: int) -> list:
    """Exactly the weeks `train --every` walks."""
    return [t for t in ctx.weeks if lo <= t <= hi][::every]


def stratified_weeks(ctx, lo, hi, per_year: int = FAST_PER_YEAR, seed: int = FAST_SEED) -> list:
    """`per_year` rank weeks drawn without replacement from each calendar
    year of [lo, hi] (all of a year's weeks if it has fewer), seeded: every
    candidate walks the same weeks."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    rng = np.random.default_rng(seed)
    weeks = pd.DatetimeIndex([t for t in ctx.weeks if lo <= t <= hi])
    out = []
    for y in sorted(set(weeks.year)):
        ys = list(weeks[weeks.year == y])
        take = min(per_year, len(ys))
        out += [ys[i] for i in sorted(rng.choice(len(ys), size=take, replace=False))]
    return sorted(out)


def cut_walk(preds_path: Path, weeks, k: int) -> pd.DataFrame:
    """The incumbent's saved walk at the given weeks, copies 1..k — the same
    seeds a fresh --k walk would use, so the comparison is same-basis."""
    p = load_preds([preds_path])
    cols = [f"c{c}" for c in range(1, k + 1)]
    missing = [c for c in cols if c not in p.columns]
    if missing:
        raise SystemExit(f"{preds_path} lacks copies {missing}")
    return p[p.week.isin(set(pd.Timestamp(w) for w in weeks))][["week", "ticker", *cols]] \
        .reset_index(drop=True)


def model_score(metric: dict) -> float:
    """The mean of the selection metric over SCORE_BOOKS — what a recipe is
    judged on. -inf when a book is missing (the walk did not rank enough)."""
    vals = [metric.get(b, float("nan")) for b in SCORE_BOOKS]
    return float(np.mean(vals)) if all(v == v for v in vals) else float("-inf")


def books_mean(h: pd.DataFrame) -> pd.Series:
    """Per week, the mean 4-week return of the SCORE_BOOKS books."""
    return sum(h[f"top{b}"] for b in SCORE_BOOKS) / len(SCORE_BOOKS)


def metrics_on_common(sel, ctx, walks: dict, k: int, lo=SELECT[0], hi=SELECT[1]):
    """The selection metric per book for every walk on the rank weeks they
    all share (same weeks, same copies: the same-bar rule). Returns
    ({name: {book: %/yr}}, {name: holdings on the common weeks}, n_common)."""
    H = {n: holdings(sel, ctx, p, range(1, k + 1)) for n, p in walks.items()}
    common = None
    for h in H.values():
        w = set(h.week[(h.week >= lo) & (h.week <= hi)])
        common = w if common is None else common & w
    H = {n: h[h.week.isin(common)].reset_index(drop=True) for n, h in H.items()}
    M = {n: selection_metric(sel, h, lo, hi) for n, h in H.items()}
    return M, H, len(common)


def copy_metrics(sel, ctx, preds: pd.DataFrame, k: int, common, lo=SELECT[0], hi=SELECT[1]) -> list:
    """The model score of each single copy on the common weeks."""
    out = []
    for c in range(1, k + 1):
        h = holdings(sel, ctx, preds, [c])
        h = h[h.week.isin(common)]
        out.append(model_score(selection_metric(sel, h, lo, hi)))
    return out


def paired_t_books(ha: pd.DataFrame, hb: pd.DataFrame) -> float:
    """Paired weekly t of the two walks' three-book average 4-week returns
    (overlapping holds: read it as roughly twice its honest size)."""
    a = books_mean(ha.set_index("week"))
    b = books_mean(hb.set_index("week"))
    return paired_t((a - b.reindex(a.index)).dropna())


def rank_by_metric(M: dict, names) -> list:
    """Names sorted by the model score, highest first."""
    return sorted(names, key=lambda n: -model_score(M[n]))


def mean_excess(h: pd.DataFrame) -> float:
    """The three-book average's mean 4-week return over the universe mean,
    in percentage points per hold — the arithmetic read beside the
    compounded score (less path noise; not the score)."""
    return float((books_mean(h) - h["rand_mean"]).mean() * 100)


def run_fast(candidates: list, incumbent: Path, out: Path, store: str = STORE,
             per_year: int = FAST_PER_YEAR, seed: int = FAST_SEED, lo=SELECT[0], hi=SELECT[1],
             log=print) -> dict:
    """`stocks-ml challenge-fast`: the candidates on a stratified random
    sample of the selection window at K=SAMPLE_K against the incumbent on
    the same weeks; ranked by the model score; nothing decided."""
    lo, hi, out, incumbent = pd.Timestamp(lo), pd.Timestamp(hi), Path(out), Path(incumbent)
    out.mkdir(parents=True, exist_ok=True)
    inc_rec = incumbent_recipe(incumbent)
    names = {candidate_name(c): c for c in candidates}
    if len(names) != len(candidates):
        raise SystemExit("two candidates have the same recipe")
    sel, ctx, _ = context(store)
    weeks = stratified_weeks(ctx, lo, hi, per_year, seed)
    desc = f"{len(weeks)} weeks, a stratified random sample ({per_year} per year, seed {seed})"
    log(f"challenge-fast: {len(names)} candidates vs the incumbent {incumbent} ({inc_rec}) on {desc} "
        f"of {lo.date()} -> {hi.date()} at K={SAMPLE_K}; ranks only, decides nothing")
    walks = {"incumbent": cut_walk(incumbent, weeks, SAMPLE_K)}
    for n, c in names.items():
        log(f"fast: {n}")
        p = walk(store, lo, hi, c["label"], c["train_years"], SAMPLE_K, out / n / f"fast_s{seed}",
                 log=log, features=c.get("features", ()), params=c.get("params"),
                 weeks=weeks, sample=desc)
        walks[n] = load_preds([p])
    M, H, n_common = metrics_on_common(sel, ctx, walks, SAMPLE_K, lo, hi)
    order = rank_by_metric(M, names)
    rows = {}
    for n in ["incumbent", *order]:
        rows[n] = {"metric": M[n], "mean_excess_pp": round(mean_excess(H[n]), 2),
                   "copies": copy_metrics(sel, ctx, walks[n], SAMPLE_K, set(H[n].week), lo, hi),
                   "paired_t_vs_incumbent": (None if n == "incumbent"
                                             else round(paired_t_books(H[n], H["incumbent"]), 2))}
    md = [f"| model | top-3 | top-6 | top-10 | score (mean of the three) | 3-book avg − universe, pp/hold | "
          f"copies 1-{SAMPLE_K} | paired t vs incumbent |", "|---|---|---|---|---|---|---|---|"]
    for n, r in rows.items():
        md.append(f"| {n} | " + " | ".join(f"{r['metric'].get(b, float('nan')):+.2f}" for b in SCORE_BOOKS)
                  + f" | **{model_score(r['metric']):+.2f}** | {r['mean_excess_pp']:+.2f} | "
                  + ", ".join(f"{v:+.2f}" for v in r["copies"])
                  + f" | {'—' if r['paired_t_vs_incumbent'] is None else f'{r['paired_t_vs_incumbent']:+.2f}'} |")
    better = [n for n in order if model_score(M[n]) > model_score(M["incumbent"])]
    log(f"challenge-fast ({n_common} common weeks), model score (mean over top-3/6/10): "
        + ", ".join(f"{n} {model_score(M[n]):+.2f}" for n in ["incumbent", *order]))
    log("\n".join(md))
    if better:
        log("above the incumbent on the sample (a ranking, not a decision): " + ", ".join(better))
        log("promote with: stocks-ml challenge --out <dir> " +
            " ".join(f"--candidate '{candidate_text(names[n], inc_rec)}'" for n in better) + " --k16")
    else:
        log("nothing above the incumbent on the sample")
    res = {"out": str(out), "store": store, "selection_window": [str(lo.date()), str(hi.date())],
           "sample": {"per_year": per_year, "seed": seed, "weeks": [str(w.date()) for w in weeks]},
           "k": SAMPLE_K, "incumbent": {"preds": str(incumbent), "recipe": inc_rec},
           "candidates": names, "rows": rows, "order": order, "better_than_incumbent": better, "md": md}
    (out / f"fast_s{seed}.json").write_text(json.dumps(res, indent=1, default=str))
    for n in order:
        record_trials([{"kind": "challenge_fast", "name": f"challenge_fast_{out.name}_s{seed}_{n}",
                        "cv_metric": model_score(M[n]), "config": names[n],
                        "notes": json.dumps({**rows[n], "sample": desc, "better_than_incumbent": n in better},
                                            default=str)}])
    log(f"challenge-fast -> {out / f'fast_s{seed}.json'}")
    return res


def candidate_text(rec: dict, base: dict) -> str:
    """A recipe back as --candidate text (only the fields that differ from
    the incumbent's)."""
    parts = []
    if rec["label"] != base["label"]:
        parts.append(f"label={rec['label']}")
    if rec["train_years"] != base["train_years"]:
        parts.append(f"train_years={rec['train_years']}")
    if list(rec.get("features") or []) != list(base.get("features") or []):
        parts.append("features=" + "+".join(rec.get("features") or []))
    if dict(rec.get("params") or {}) != dict(base.get("params") or {}):
        parts.append("params=" + "+".join(f"{k}:{v}" for k, v in (rec.get("params") or {}).items()))
    return ",".join(parts)


def _ledger(out: Path, stage: str, name: str, rec: dict, note: dict) -> None:
    record_trials([{"kind": "challenge", "name": f"challenge_{out.name}_{stage}_{name}",
                    "cv_metric": model_score(note.get("metric", {})) if note.get("metric") else None,
                    "config": rec, "notes": json.dumps(note, default=str)}])


def run(candidates: list, incumbent: Path, out: Path, store: str = STORE, k16: bool = False,
        lo=SELECT[0], hi=SELECT[1], log=print) -> dict:
    lo, hi, out, incumbent = pd.Timestamp(lo), pd.Timestamp(hi), Path(out), Path(incumbent)
    out.mkdir(parents=True, exist_ok=True)
    inc_rec = incumbent_recipe(incumbent)
    names = {candidate_name(c): c for c in candidates}
    if len(names) != len(candidates):
        raise SystemExit("two candidates have the same recipe")
    for n, c in names.items():
        if c == recipe(inc_rec["label"], inc_rec["train_years"], inc_rec.get("features", ()),
                       inc_rec.get("params")):
            raise SystemExit(f"candidate {n} is the incumbent's own recipe")
    sel, ctx, _ = context(store)
    res = {"out": str(out), "store": store, "selection_window": [str(lo.date()), str(hi.date())],
           "rules": {"sample_every": SAMPLE_EVERY, "sample_k": SAMPLE_K, "full_k": FULL_K,
                     "advance": ADVANCE, "score_books": SCORE_BOOKS, "final_k": FINAL_K},
           "incumbent": {"preds": str(incumbent), "recipe": inc_rec}, "candidates": names}
    log(f"challenge: {len(names)} candidates vs the incumbent {incumbent} ({inc_rec}); "
        f"selection window {lo.date()} -> {hi.date()}; rules {res['rules']}")

    # ---- stage 1: the sample ranks
    weeks = sample_weeks(ctx, lo, hi, SAMPLE_EVERY)
    walks = {"incumbent": cut_walk(incumbent, weeks, SAMPLE_K)}
    for n, c in names.items():
        log(f"stage 1: {n} — every {SAMPLE_EVERY}th week at K={SAMPLE_K}")
        p = walk(store, lo, hi, c["label"], c["train_years"], SAMPLE_K, out / n / "sample",
                 every=SAMPLE_EVERY, log=log, features=c.get("features", ()), params=c.get("params"))
        walks[n] = load_preds([p])
    M1, _, n1 = metrics_on_common(sel, ctx, walks, SAMPLE_K, lo, hi)
    order1 = rank_by_metric(M1, names)
    advance = order1[:ADVANCE]
    res["stage1"] = {"weeks": n1, "metric": M1, "order": order1, "advance": advance}
    log(f"stage 1 ({n1} sample weeks, ranking only), model score: " +
        "; ".join(f"{n} {model_score(M1[n]):+.2f}" for n in ["incumbent", *order1])
        + f" — advancing {advance}")
    for n in names:
        _ledger(out, "stage1", n, names[n], {"metric": M1[n], "weeks": n1, "advanced": n in advance,
                                              "note": "sample; ranks only"})

    # ---- stage 2: every week, the frozen comparison
    walks = {"incumbent": cut_walk(incumbent, [t for t in ctx.weeks if lo <= t <= hi], FULL_K)}
    for n in advance:
        c = names[n]
        log(f"stage 2: {n} — every week at K={FULL_K}")
        p = walk(store, lo, hi, c["label"], c["train_years"], FULL_K, out / n / "select",
                 log=log, features=c.get("features", ()), params=c.get("params"))
        walks[n] = load_preds([p])
    M2, H2, n2 = metrics_on_common(sel, ctx, walks, FULL_K, lo, hi)
    detail = {}
    for n in ["incumbent", *advance]:
        preds = walks[n]
        detail[n] = {"metric": M2[n],
                     "copies": copy_metrics(sel, ctx, preds, FULL_K, set(H2[n].week), lo, hi),
                     "paired_t_vs_incumbent": (None if n == "incumbent"
                                               else round(paired_t_books(H2[n], H2["incumbent"]), 2))}
        if n != "incumbent":
            la = audit_segments(store, [str(out / n / "select" / "preds.parquet")], ctx)
            detail[n]["leak_audit"] = {"verdict": la["VERDICT"], "line": leak_line(la)}
    eligible = ["incumbent"] + [n for n in advance if detail[n]["leak_audit"]["verdict"] == "PASS"]
    order2 = rank_by_metric(M2, eligible)
    winner = order2[0]
    res["stage2"] = {"weeks": n2, "detail": detail, "order": order2, "winner": winner}
    md = [f"| model | top-3 | top-6 | top-10 | score (mean of the three) | copies 1-{FULL_K} | paired t vs incumbent | leak audit |",
          "|---|---|---|---|---|---|---|---|"]
    for n in ["incumbent", *advance]:
        d = detail[n]
        md.append(f"| {'**' + n + '**' if n == winner else n} | " +
                  " | ".join(f"{d['metric'].get(b, float('nan')):+.2f}" for b in SCORE_BOOKS) +
                  f" | **{model_score(d['metric']):+.2f}** | " + ", ".join(f"{v:+.2f}" for v in d["copies"]) +
                  f" | {'—' if d['paired_t_vs_incumbent'] is None else f'{d['paired_t_vs_incumbent']:+.2f}'}"
                  f" | {d.get('leak_audit', {}).get('verdict', '—')} |")
    res["stage2"]["md"] = md
    log(f"stage 2 ({n2} common weeks, every week, K={FULL_K}): the argmax of the model score "
        f"(mean over top-3/6/10) is {winner}" + (" — the incumbent stands" if winner == "incumbent" else ""))
    log("\n".join(md))
    for n in advance:
        _ledger(out, "stage2", n, names[n], {**detail[n], "weeks": n2, "winner": n == winner})
    (out / "challenge.json").write_text(json.dumps(res, indent=1, default=str))

    # ---- stage 3: the winner at K=16 and the one look
    if winner != "incumbent" and k16:
        from stocks_ml.eval import run as eval_run
        from stocks_ml.selection import HOLDOUT_START
        c = names[winner]
        wdir = out / f"{winner}_k{FINAL_K}"
        for seg, (a, b) in {"select": (lo, hi),
                            "extend": (hi + pd.Timedelta(days=1), HOLDOUT_START - pd.Timedelta(days=1))}.items():
            log(f"stage 3: {winner} {seg} {a.date()} -> {b.date()} at K={FINAL_K}")
            walk(store, a, b, c["label"], c["train_years"], FINAL_K, wdir / seg, log=log,
                 features=c.get("features", ()), params=c.get("params"))
        res["stage3"] = {"walk": str(wdir),
                         "eval": eval_run(wdir, incumbent.parent.parent, store=store, k=FINAL_K, log=log)}
        (out / "challenge.json").write_text(json.dumps(res, indent=1, default=str))
        log(f"stage 3 done: adopt with `stocks-ml procedure --preds {wdir / 'select' / 'preds.parquet'}` "
            "on the owner's go")
    elif winner != "incumbent":
        log(f"next: `stocks-ml challenge ... --k16` (or train {winner} at K={FINAL_K} on both segments "
            f"and run `stocks-ml eval --walk <walk> --incumbent {incumbent.parent.parent}`)")
    log(f"challenge -> {out / 'challenge.json'}")
    return res
