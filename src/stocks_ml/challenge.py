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
  stage 2  every week of the selection window at K=FULL_K (16, the deployed
           ensemble size; at K=4 models K=16 separates are coin flips); the MODEL SCORE —
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
  stage 3  (--k16) the winning candidate at K=FINAL_K on both segments (the
           stage-2 walk is reused when FULL_K == FINAL_K, so only 2016-2024
           is walked), then `stocks-ml eval --incumbent`: the one look at
           2016-2024 with the falsification test. Adoption is `stocks-ml
           procedure`, on the owner's go — never here.

Every stage records a ledger row (kind `challenge`); <out>/challenge.json
holds the whole record. Walks resume, so a rerun picks up where it stopped.

    stocks-ml challenge --out data/experiments/challenges/labels \\
        --candidate label=label_4w_sector_rank --candidate label=label_4w_sector_log
    stocks-ml challenge --out ... --candidate train_years=5 \\
        --candidate "features=x_a+x_b" --candidate "params=learning_rate:0.01+n_estimators:300"

`stocks-ml challenge-fast` is the prototype: the same candidates on a
stratified random sample of the selection window (FAST_PER_YEAR weeks from
each year, seed FAST_SEED) at K=FAST_K, against the incumbent cut to the
same weeks and copies — under an hour per candidate, for trying any change
(labels, windows, features, params) before spending a full challenge on
it. It ranks; it decides nothing. Its calibration is the SEED TWIN: the
incumbent's own recipe walked again with fresh seeds (copies FINAL_K+1..
2*FINAL_K, every week of the selection window, once per champion, cached at
<incumbent walk>/twin/). The twin vs the incumbent on NULL_DRAWS other draws
of the same sample design is what a boost of zero looks like; a candidate's
gap is reported as a percentile of that null and flagged only above
FLAG_PERCENTILE — the false-positive rate is then a measured number. It
prints the `challenge` line for what was flagged. Strategy layers need no walk at all: prototype
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
FULL_K = 16               # stage 2: copies, every week — the deployed ensemble size (K=4 could not
                          # separate models that K=16 does: the 2026-09-14 resampling; set 2026-09-15)
ADVANCE = 2               # candidates that go from stage 1 to stage 2
SCORE_BOOKS = (3, 6, 10)  # the model score: the mean of the selection metric over these books
FINAL_K = 16              # stage 3: the deployed ensemble size
FAST_PER_YEAR = 26        # challenge-fast: weeks drawn from each year of the window (26: a
                          # 7-point boost is detected ~19 times in 20; 13 was ~4 in 5)
FAST_SEED = 0             # ... with this seed (the same weeks for every candidate)
NULL_DRAWS = 1000         # the seed twin vs the incumbent on this many other draws: the null
FLAG_PERCENTILE = 90      # a candidate is flagged above this percentile of the null (FPR 10%)
FAST_K = 16               # challenge-fast: copies — the deployed ensemble size. At K=4 two models
                          # a few points apart are a coin flip on any sample (the 2026-09-14
                          # resampling: rank vs the incumbent 50% at K=4, 81% at K=16 on 130 weeks)
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


def refuse_leaky_features(ctx, candidates: list, lo, hi, log=print) -> None:
    """Every feature a candidate names is checked against the future split
    factor on the selection window (leak_audit.feature_factor_check) before
    a single copy is fit; a failing feature ends the run."""
    from stocks_ml.leak_audit import FEATURE_FACTOR_LIMIT, feature_factor_check
    feats = sorted({f for c in candidates for f in (c.get("features") or [])})
    if not feats:
        return
    missing = [f for f in feats if f not in ctx.pan.columns]
    if missing:
        raise SystemExit(f"the panel lacks the recipe's feature columns {missing}")
    res = feature_factor_check(ctx, feats, lo, hi)
    for f in feats:
        r = res[f]
        log(f"feature leak check: {f} corr with the future split factor {r['corr_with_future_split_factor']:+.3f} "
            f"({r['weeks']} weeks) -> {r['verdict']}")
    if res["VERDICT"] == "FAIL":
        raise SystemExit(f"feature(s) correlate with the FUTURE split factor beyond {FEATURE_FACTOR_LIMIT}: "
                         f"{[f for f in feats if res[f]['verdict'] == 'FAIL']} — a look-ahead; not walked")


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


def cut_walk(preds_path: Path, weeks, k: int, copies=None) -> pd.DataFrame:
    """The incumbent's saved walk at the given weeks, copies 1..k (or the
    explicit `copies`, renamed c1..ck so a seed twin reads as a K walk) —
    the same seeds a fresh --k walk would use, so the comparison is
    same-basis."""
    p = load_preds([preds_path])
    copies = list(copies) if copies is not None else list(range(1, k + 1))
    cols = [f"c{c}" for c in copies]
    missing = [c for c in cols if c not in p.columns]
    if missing:
        raise SystemExit(f"{preds_path} lacks copies {missing}")
    out = p[p.week.isin(set(pd.Timestamp(w) for w in weeks))][["week", "ticker", *cols]].reset_index(drop=True)
    out.columns = ["week", "ticker", *[f"c{i}" for i in range(1, len(cols) + 1)]]
    return out


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


def twin_dir(incumbent: Path) -> Path:
    """The seed twin lives beside the incumbent's walk: <walk>/twin/."""
    return Path(incumbent).parent.parent / "twin"


def twin_copies() -> range:
    return range(FINAL_K + 1, 2 * FINAL_K + 1)


def ensure_twin(incumbent: Path, inc_rec: dict, store: str, lo, hi, workers: int = 1, log=print) -> Path:
    """Walk the incumbent's recipe with fresh seeds on every week of the
    selection window (resumes if partly done; instant if complete)."""
    d = twin_dir(incumbent)
    log(f"seed twin: {inc_rec} copies {twin_copies().start}..{twin_copies().stop - 1} -> {d}")
    walk(store, lo, hi, inc_rec["label"], inc_rec["train_years"], FINAL_K, d, log=log,
         features=inc_rec.get("features", ()), params=inc_rec.get("params"), workers=workers,
         copies=twin_copies())
    return d / "preds.parquet"


def null_gaps(sel, ctx, incumbent: Path, twin: Path, per_year: int, lo=SELECT[0], hi=SELECT[1],
              draws: int = NULL_DRAWS, log=print) -> np.ndarray:
    """The model-score gap of the seed twin over the incumbent on `draws`
    stratified samples (seeds 1..draws; the candidates' draw is seed 0) —
    what a boost of zero looks like under this sample design. Cached at
    <twin dir>/null_<per_year>.json."""
    cache = Path(twin).parent / f"null_{per_year}.json"
    if cache.exists():
        rec = json.loads(cache.read_text())
        if rec["draws"] == draws and rec["k"] == FINAL_K:
            return np.array(rec["gaps"])
    hi_ = holdings(sel, ctx, load_preds([incumbent]), range(1, FINAL_K + 1))
    ht = holdings(sel, ctx, load_preds([twin]), twin_copies())
    common = set(hi_.week) & set(ht.week)
    hi_, ht = hi_[hi_.week.isin(common)], ht[ht.week.isin(common)]
    gaps = []
    for sd in range(1, draws + 1):
        w = set(stratified_weeks(ctx, lo, hi, per_year, sd))
        gaps.append(model_score(selection_metric(sel, ht[ht.week.isin(w)], lo, hi))
                    - model_score(selection_metric(sel, hi_[hi_.week.isin(w)], lo, hi)))
    gaps = np.array(gaps)
    seed_luck = model_score(selection_metric(sel, ht, lo, hi)) - model_score(selection_metric(sel, hi_, lo, hi))
    cache.write_text(json.dumps({"draws": draws, "k": FINAL_K, "per_year": per_year, "weeks": len(common),
                                 "seed_luck_full_walk": round(float(seed_luck), 2),
                                 "gaps": [round(float(g), 4) for g in gaps],
                                 "percentiles": {str(q): round(float(np.percentile(gaps, q)), 2)
                                                 for q in (5, 10, 25, 50, 75, 90, 95)}}, indent=1))
    log(f"null ({draws} draws of {per_year}/yr, twin vs incumbent): median {np.median(gaps):+.2f}, sd {gaps.std():.2f}; "
        f"seed luck on the full walk {seed_luck:+.2f} -> {cache}")
    return gaps


def seed_luck(twin: Path, per_year: int) -> float | None:
    """The twin's full-walk score minus the incumbent's: what two seed sets
    of the same recipe differ by. Gaps inside it are seed noise."""
    cache = Path(twin).parent / f"null_{per_year}.json"
    return json.loads(cache.read_text()).get("seed_luck_full_walk") if cache.exists() else None


def null_percentile(gap: float, gaps: np.ndarray) -> float:
    """The share of null draws below the candidate's gap: P(boost) as a
    calibrated number."""
    return float(np.mean(gaps < gap))


def mean_excess(h: pd.DataFrame) -> float:
    """The three-book average's mean 4-week return over the universe mean,
    in percentage points per hold — the arithmetic read beside the
    compounded score (less path noise; not the score)."""
    return float((books_mean(h) - h["rand_mean"]).mean() * 100)


def run_fast(candidates: list, incumbent: Path, out: Path, store: str = STORE,
             per_year: int = FAST_PER_YEAR, seed: int = FAST_SEED, lo=SELECT[0], hi=SELECT[1],
             log=print, workers: int = 1) -> dict:
    """`stocks-ml challenge-fast`: the candidates on a stratified random
    sample of the selection window at K=FAST_K against the incumbent on
    the same weeks; ranked by the model score; nothing decided."""
    lo, hi, out, incumbent = pd.Timestamp(lo), pd.Timestamp(hi), Path(out), Path(incumbent)
    out.mkdir(parents=True, exist_ok=True)
    inc_rec = incumbent_recipe(incumbent)
    names = {candidate_name(c): c for c in candidates}
    if len(names) != len(candidates):
        raise SystemExit("two candidates have the same recipe")
    sel, ctx, _ = context(store)
    refuse_leaky_features(ctx, list(names.values()), lo, hi, log)
    weeks = stratified_weeks(ctx, lo, hi, per_year, seed)
    desc = f"{len(weeks)} weeks, a stratified random sample ({per_year} per year, seed {seed})"
    log(f"challenge-fast: {len(names)} candidates vs the incumbent {incumbent} ({inc_rec}) on {desc} "
        f"of {lo.date()} -> {hi.date()} at K={FAST_K}; ranks only, decides nothing")
    twin = ensure_twin(incumbent, inc_rec, store, lo, hi, workers=workers, log=log)
    null = null_gaps(sel, ctx, incumbent, twin, per_year, lo, hi, log=log)
    null = null - null.mean()                  # centred: the spread of a zero boost under this design
    thr = float(np.percentile(null, FLAG_PERCENTILE))
    luck = seed_luck(twin, per_year)
    walks = {"incumbent": cut_walk(incumbent, weeks, FAST_K),
             "incumbent (twin seeds)": cut_walk(twin, weeks, FAST_K, copies=twin_copies())}
    for n, c in names.items():
        log(f"fast: {n}")
        p = walk(store, lo, hi, c["label"], c["train_years"], FAST_K, out / n / f"fast_{per_year}x_s{seed}",
                 log=log, features=c.get("features", ()), params=c.get("params"),
                 weeks=weeks, sample=desc, workers=workers)
        walks[n] = load_preds([p])
    M, H, n_common = metrics_on_common(sel, ctx, walks, FAST_K, lo, hi)
    order = rank_by_metric(M, names)
    inc_score = model_score(M["incumbent"])
    ref = max(inc_score, model_score(M["incumbent (twin seeds)"]))   # the luckier seed set is the bar
    rows = {}
    for n in ["incumbent", "incumbent (twin seeds)", *order]:
        gap = None if n.startswith("incumbent") else model_score(M[n]) - ref
        rows[n] = {"metric": M[n], "score": round(model_score(M[n]), 2),
                   "gap": None if gap is None else round(gap, 2),
                   "gap_vs_incumbent_seeds_1_16": None if gap is None else round(model_score(M[n]) - inc_score, 2),
                   "null_percentile": None if gap is None else round(null_percentile(gap, null), 3),
                   "flagged": False if gap is None else bool(gap > thr),
                   "mean_excess_pp": round(mean_excess(H[n]), 2),
                   "copies": copy_metrics(sel, ctx, walks[n], FAST_K, set(H[n].week), lo, hi),
                   "paired_t_vs_incumbent": (None if n == "incumbent"
                                             else round(paired_t_books(H[n], H["incumbent"]), 2))}
    md = [f"| model | top-3 | top-6 | top-10 | score | gap vs the better seed set | P(boost) = null percentile | "
          f"flagged (> {FLAG_PERCENTILE}th) | 3-book avg − universe, pp/hold | paired t |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in rows.items():
        md.append(f"| {n} | " + " | ".join(f"{r['metric'].get(b, float('nan')):+.2f}" for b in SCORE_BOOKS)
                  + f" | **{r['score']:+.2f}** | {'—' if r['gap'] is None else f'{r['gap']:+.2f}'}"
                  + f" | {'—' if r['null_percentile'] is None else f'{r['null_percentile']:.0%}'}"
                  + f" | {'—' if r['gap'] is None else ('YES' if r['flagged'] else 'no')}"
                  + f" | {r['mean_excess_pp']:+.2f}"
                  + f" | {'—' if r['paired_t_vs_incumbent'] is None else f'{r['paired_t_vs_incumbent']:+.2f}'} |")
    flagged = [n for n in order if rows[n]["flagged"]]
    log(f"challenge-fast ({n_common} common weeks), model score (mean over top-3/6/10): incumbent {inc_score:+.2f}, "
        f"its twin seeds {model_score(M['incumbent (twin seeds)']):+.2f} (seed luck on the full walk "
        f"{'n/a' if luck is None else f'{luck:+.2f}'}: gaps inside this are seed noise); "
        + ", ".join(f"{n} {rows[n]['score']:+.2f} (gap vs the better seed set {rows[n]['gap']:+.2f}, "
                    f"null pct {rows[n]['null_percentile']:.0%})" for n in order)
        + f"; flag threshold {thr:+.2f} above the better seed set (the centred null's {FLAG_PERCENTILE}th percentile)")
    log("\n".join(md))
    if flagged:
        log(f"flagged (gap above the null's {FLAG_PERCENTILE}th percentile; a ranking, not a decision): "
            + ", ".join(flagged))
        log("promote with: stocks-ml challenge --out <dir> " +
            " ".join(f"--candidate '{candidate_text(names[n], inc_rec)}'" for n in flagged) + " --k16")
    else:
        log("nothing flagged: no candidate's gap clears the null")
    res = {"out": str(out), "store": store, "selection_window": [str(lo.date()), str(hi.date())],
           "sample": {"per_year": per_year, "seed": seed, "weeks": [str(w.date()) for w in weeks]},
           "k": FAST_K, "incumbent": {"preds": str(incumbent), "recipe": inc_rec},
           "null": {"twin": str(twin), "draws": int(len(null)), "flag_percentile": FLAG_PERCENTILE,
                    "threshold": round(thr, 2), "centred": True, "seed_luck_full_walk": luck,
                    "reference": "the higher-scoring of the incumbent's two seed sets on the draw",
                    "percentiles": {str(q): round(float(np.percentile(null, q)), 2) for q in (10, 50, 90, 95)}},
           "candidates": names, "rows": rows, "order": order, "flagged": flagged, "md": md}
    (out / f"fast_{per_year}x_s{seed}.json").write_text(json.dumps(res, indent=1, default=str))
    for n in order:
        record_trials([{"kind": "challenge_fast", "name": f"challenge_fast_{out.name}_{per_year}x_s{seed}_{n}",
                        "cv_metric": rows[n]["score"], "config": names[n],
                        "notes": json.dumps({**rows[n], "sample": desc, "flag_threshold": round(thr, 2)},
                                            default=str)}])
    log(f"challenge-fast -> {out / f'fast_{per_year}x_s{seed}.json'}")
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
        lo=SELECT[0], hi=SELECT[1], log=print, workers: int = 1) -> dict:
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
    refuse_leaky_features(ctx, list(names.values()), lo, hi, log)
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
                 every=SAMPLE_EVERY, log=log, features=c.get("features", ()), params=c.get("params"),
                 workers=workers)
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
    twin = twin_dir(incumbent) / "preds.parquet"
    if twin.exists() and FULL_K == FINAL_K:     # the incumbent is a recipe, not one seed set: both count
        walks["incumbent (twin seeds)"] = cut_walk(twin, [t for t in ctx.weeks if lo <= t <= hi], FULL_K,
                                                   copies=twin_copies())
    for n in advance:
        c = names[n]
        log(f"stage 2: {n} — every week at K={FULL_K}")
        p = walk(store, lo, hi, c["label"], c["train_years"], FULL_K, out / n / "select",
                 log=log, features=c.get("features", ()), params=c.get("params"), workers=workers)
        walks[n] = load_preds([p])
    M2, H2, n2 = metrics_on_common(sel, ctx, walks, FULL_K, lo, hi)
    incumbents = [n for n in walks if n.startswith("incumbent")]
    detail = {}
    for n in [*incumbents, *advance]:
        preds = walks[n]
        detail[n] = {"metric": M2[n],
                     "copies": copy_metrics(sel, ctx, preds, FULL_K, set(H2[n].week), lo, hi),
                     "paired_t_vs_incumbent": (None if n == "incumbent"
                                               else round(paired_t_books(H2[n], H2["incumbent"]), 2))}
        if n not in incumbents:
            la = audit_segments(store, [str(out / n / "select" / "preds.parquet")], ctx)
            detail[n]["leak_audit"] = {"verdict": la["VERDICT"], "line": leak_line(la)}
    eligible = incumbents + [n for n in advance if detail[n]["leak_audit"]["verdict"] == "PASS"]
    order2 = rank_by_metric(M2, eligible)
    winner = order2[0]
    if winner in incumbents:
        winner = "incumbent"
    if len(incumbents) > 1:
        log(f"seed luck: the incumbent's two seed sets score {model_score(M2['incumbent']):+.2f} and "
            f"{model_score(M2['incumbent (twin seeds)']):+.2f} on the full walk — a candidate gap inside "
            f"{abs(model_score(M2['incumbent (twin seeds)']) - model_score(M2['incumbent'])):.2f} is seed noise; "
            f"the argmax must beat both")
    res["stage2"] = {"weeks": n2, "detail": detail, "order": order2, "winner": winner}
    md = [f"| model | top-3 | top-6 | top-10 | score (mean of the three) | copies 1-{FULL_K} | paired t vs incumbent | leak audit |",
          "|---|---|---|---|---|---|---|---|"]
    for n in [*incumbents, *advance]:
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
        # the stage-2 walk is the select segment when it already has FINAL_K copies
        wdir = out / winner if FULL_K == FINAL_K else out / f"{winner}_k{FINAL_K}"
        for seg, (a, b) in {"select": (lo, hi),
                            "extend": (hi + pd.Timedelta(days=1), HOLDOUT_START - pd.Timedelta(days=1))}.items():
            log(f"stage 3: {winner} {seg} {a.date()} -> {b.date()} at K={FINAL_K}")
            walk(store, a, b, c["label"], c["train_years"], FINAL_K, wdir / seg, log=log,
                 features=c.get("features", ()), params=c.get("params"), workers=workers)
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
