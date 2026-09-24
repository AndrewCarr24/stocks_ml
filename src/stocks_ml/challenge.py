"""`stocks-ml challenge`: the challenger protocol as one process.

A candidate is a recipe — label, training window, extra panel columns,
overrides of selection.MODEL_PARAMS — anything `stocks-ml train` can walk. A
new label, a shorter window, a set of engineered columns, a learning rate:
each is a recipe, walked and judged the same way. The rules are the
registered ones (reports/clean_improvement_registration.md: rules 2, 5, 7,
8) and every constant lives at the top of this file:

  (no screen) Until 2026-09-24 a stage 1 ranked every candidate on every
           4th week at K=4 and sent the top two on. The owner dropped it: the
           same model's four seed sets spread +4.4 .. +16.6 on it, more than the
           band stage 2 decides by, so it ranked mostly luck — its leader that
           day (+23.6, ten points clear) tied the incumbent in stage 2, and a
           candidate it placed 24th had tied the incumbent in a full test. At
           the screening cadence a full walk costs ~4 screens, so every
           candidate now walks the full comparison: keep menus small.
  stage 2  every candidate, every week of the selection window at K=FULL_K (16, the deployed
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
FULL_K = 16               # stage 2: copies, every week — the deployed ensemble size (K=4 could not
                          # separate models that K=16 does: the 2026-09-14 resampling; set 2026-09-15)
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
RECIPE_KEYS = ("label", "train_years", "features", "params", "drop", "store", "train_top")
ADJUDICATE = (pd.Timestamp("2016-01-01"), pd.Timestamp("2019-12-31"))
# The head-to-head window (owner's rule 2026-09-18): the incumbent is the argmax
# of a long search on 2006-2015, so its score there is inflated by selection and
# a single fresh candidate's is not. Both recipes are chosen on 2006-2015 alone
# (a challenger's variants by stage 2); the challenger's winner then meets the
# incumbent once on 2016-2019, neither having been selected there, at K=16 on
# the same weeks, each at its own procedure-decided settings, with the
# incumbent's seed twin on the window for the noise band. Used for final
# head-to-heads only; 2020 -> the holdout stays untouched.


def parse_candidate(text: str, base: dict) -> dict:
    """'label=...,train_years=5,features=a+b,params=learning_rate:0.01+max_depth:4'
    over the incumbent's recipe as the base: unspecified fields are inherited."""
    r = {"label": base["label"], "train_years": base["train_years"],
         "features": list(base.get("features") or []), "params": dict(base.get("params") or {}),
         "drop": list(base.get("drop") or []), "train_top": base.get("train_top")}
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
        elif k == "drop":
            r["drop"] = [f for f in v.split("+") if f]
        elif k == "store":
            r["store"] = v                       # another world (universe): the walk and its scoring run there
        elif k == "train_top":
            r["train_top"] = int(v)              # fit on the largest N names by market cap; score every member
        else:
            r["params"] = {}
            for kv in [x for x in v.split("+") if x]:
                if ":" not in kv:
                    raise ValueError(f"param {kv!r} is not name:value")
                pk, pv = kv.split(":", 1)
                r["params"][pk.strip()] = pv.strip()
    out = recipe(r["label"], r["train_years"], r["features"], r["params"], r["drop"], r.get("train_top"))
    if r.get("store"):
        out["store"] = r["store"]
    return out


def candidate_name(rec: dict) -> str:
    """A directory name for a recipe: label and window, plus a short hash of
    any features or params."""
    name = f"{rec['label']}_{rec['train_years']}y"
    extra = {k: rec[k] for k in ("features", "params", "drop", "store", "train_top") if rec.get(k)}
    if extra:
        name += "_" + hashlib.sha256(json.dumps(extra, sort_keys=True).encode()).hexdigest()[:8]
    return name


def refuse_leaky_features(ctx, candidates: list, lo, hi, log=print) -> None:
    """Every feature a candidate names is checked against the future split
    factor on every scan window (leak_audit.feature_factor_worst: the
    selection window and the pre-holdout extension, since 2026-09-21) before
    a single copy is fit; a failing feature ends the run."""
    from stocks_ml.leak_audit import FEATURE_FACTOR_LIMIT, feature_factor_worst
    feats = sorted({f for c in candidates for f in (c.get("features") or [])})
    if not feats:
        return
    missing = [f for f in feats if f not in ctx.pan.columns]
    if missing:
        raise SystemExit(f"the panel lacks the recipe's feature columns {missing}")
    from stocks_ml.features.panel import SR_PREFIX
    parents = {f: f[len(SR_PREFIX):] for f in feats if f.startswith(SR_PREFIX) and f[len(SR_PREFIX):] in ctx.pan.columns}
    res = feature_factor_worst(ctx, sorted(set(feats) | set(parents.values())))
    failed = []
    for f in feats:
        r = res.get(f, {"corr_with_future_split_factor": float("nan"), "verdict": "PASS"})
        verdict = r["verdict"]
        if f in parents:
            # a within-sector re-ranking of a panel column cannot add split information: it is
            # judged against its parent (2026-09-19: several admitted columns sit at 0.15-0.21
            # for economic reasons — value and quality names split less)
            p = res.get(parents[f], {}).get("corr_with_future_split_factor", float("nan"))
            if p != p:      # a parent constant within the week (macro, market, flags): no split information to inherit
                p = 0.0
            verdict = "PASS" if abs(r["corr_with_future_split_factor"]) <= max(FEATURE_FACTOR_LIMIT, abs(p) + 0.05) else "FAIL"
            log(f"feature leak check: {f} corr with the future split factor {r['corr_with_future_split_factor']:+.3f} "
                f"(parent {parents[f]} {p:+.3f}; {r.get('window', '-')}) -> {verdict}")
        else:
            log(f"feature leak check: {f} corr with the future split factor {r['corr_with_future_split_factor']:+.3f} "
                f"({r.get('window', '-')}) -> {verdict}")
        if verdict == "FAIL":
            failed.append(f)
    if failed:
        raise SystemExit(f"feature(s) correlate with the FUTURE split factor beyond {FEATURE_FACTOR_LIMIT} "
                         f"(or beyond their parent's): {failed} — a look-ahead; not walked")


class Worlds:
    """The contexts a challenge scores on: the incumbent's store, and a
    candidate's own when its recipe names one (`store=`: another universe).
    Loaded on first use so a second world costs memory only while it is
    scored; a candidate's walk builds and releases its own."""

    def __init__(self, sel, ctx, store: str):
        self.sel, self.base, self.store = sel, ctx, store
        self._ctx = {store: ctx}

    def store_of(self, cand: dict) -> str:
        return cand.get("store") or self.store

    def ctx_of(self, cand: dict):
        s = self.store_of(cand)
        if s not in self._ctx:
            _, self._ctx[s], _ = context(s)
        return self._ctx[s]


def refuse_leaky_features_by_world(worlds: Worlds, candidates: list, lo, hi, log=print) -> None:
    by_store = {}
    for c in candidates:
        by_store.setdefault(worlds.store_of(c), []).append(c)
    for s, cs in by_store.items():
        if any(c.get("features") for c in cs):
            refuse_leaky_features(worlds.ctx_of(cs[0]), cs, lo, hi, log)


def incumbent_recipe(preds_path: Path, spec_path: Path | None = None, allow_other: bool = False) -> dict:
    """The incumbent walk's recipe — refused unless it is the champion's
    (the spec's) recipe, since 2026-09-20: for two days the screens graded
    clean candidates against the champion's OLD walks, fitted before the
    dollar-volume fix, whose leak is worth 4-5 points on 2006-2015 (the
    "leaky yardstick"). `allow_other` is for a deliberate comparison
    against another recipe."""
    from stocks_ml.procedure import SPEC_PATH
    rec = json.loads((Path(preds_path).parent / "spec.json").read_text())
    if not rec.get("recipe"):
        raise SystemExit(f"{preds_path} has no recipe in its record; the challenge needs the "
                         "incumbent's label and window")
    r = rec["recipe"]
    if not allow_other:
        spec = json.loads(Path(spec_path or SPEC_PATH).read_text())
        want = {"label": spec["horizon"]["label"], "train_years": int(spec["training_window_years"]),
                "features": list(spec.get("features") or []), "drop": list(spec.get("drop_features") or []),
                "params": {}, "train_top": spec.get("train_top")}
        have = {"label": r["label"], "train_years": int(r["train_years"]), "features": list(r.get("features") or []),
                "drop": list(r.get("drop") or []), "params": dict(r.get("params") or {}), "train_top": r.get("train_top")}
        if have != want:
            raise SystemExit(f"the incumbent {preds_path} was walked with {have}, not the champion's recipe {want} "
                             f"(models/champion_spec.json): a stale yardstick. Pass --incumbent-recipe-ok to compare "
                             f"against another recipe on purpose.")
    return r


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


def stratified_blocks(ctx, lo, hi, per_year: int = FAST_PER_YEAR, seed: int = FAST_SEED, block: int = 4) -> list:
    """`per_year // block` runs of `block` consecutive rank weeks drawn from
    each calendar year (the year cut into consecutive runs; runs drawn
    without replacement), seeded: the screening cadence's sample — each run
    is one fit serving `block` weeks."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    rng = np.random.default_rng(seed)
    weeks = pd.DatetimeIndex([t for t in ctx.weeks if lo <= t <= hi])
    out = []
    for y in sorted(set(weeks.year)):
        ys = list(weeks[weeks.year == y])
        runs = [ys[i:i + block] for i in range(0, len(ys) - block + 1, block)]
        take = min(max(1, per_year // block), len(runs))
        for i in sorted(rng.choice(len(runs), size=take, replace=False)):
            out += runs[i]
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


def metrics_on_common(sel, ctx, walks: dict, k, lo=SELECT[0], hi=SELECT[1], ctxs: dict | None = None):
    """The selection metric per book for every walk on the rank weeks they
    all share (same weeks; the same-bar rule); `ctxs` names another world's
    context for a walk made there; `k` is the copies per walk (an int, or
    {name: int} — the incumbent carries both seed sets, 32). Returns
    ({name: {book: %/yr}}, {name: holdings on the common weeks}, n_common)."""
    k_of = k if isinstance(k, dict) else {n: k for n in walks}
    H = {n: holdings(sel, (ctxs or {}).get(n, ctx), p, range(1, k_of[n] + 1)) for n, p in walks.items()}
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


BAND_DRAWS = 40           # random half-ensembles scored per walk for its seed band
BAND_Z = 2.0              # a gap is decided when it exceeds BAND_Z x the two walks' combined seed sd ...
PAIRED_T = 2.0            # ... AND the paired weekly t of the three-book average is at least this (2026-09-20: two
                          # walks of the same recipe still differed by 4 points at K=64 — a one-name perturbation
                          # moves every copy the same way; only the weeks can say whether a gap is real)


def seed_band(sel, ctx, preds: pd.DataFrame, copies, common, lo=SELECT[0], hi=SELECT[1], draws: int = BAND_DRAWS) -> dict:
    """How much a walk's model score moves on seed luck alone: the scores
    of `draws` random half-ensembles of its copies (the 2026-09-19 sanity
    check: two walks of the SAME recipe scored +12.4 and +7.3 at K=16 while
    their half-ensembles spread 3 points each). `sd16` scales the half-
    ensemble spread to the full ensemble (var halves with twice the
    copies); the verdict reads gaps against it."""
    copies = list(copies)
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(draws):
        sub = sorted(rng.choice(copies, max(2, len(copies) // 2), replace=False))
        h = holdings(sel, ctx, preds, sub)
        h = h[h.week.isin(common)]
        scores.append(model_score(selection_metric(sel, h, lo, hi)))
    scores = np.array(scores)
    return {"half_mean": round(float(scores.mean()), 2), "half_sd": round(float(scores.std(ddof=1)), 2),
            "sd16": round(float(scores.std(ddof=1) / np.sqrt(2)), 2), "draws": draws, "copies": len(copies)}


def decide(score_c: float, score_i: float, band_c: dict, band_i: dict, paired_t: float | None = None) -> tuple[str, float, float]:
    """The symmetric verdict: a gap is decided only when it beats BAND_Z x
    the two walks' combined seed sd AND (when given) the paired weekly t
    of the three-book average is at least PAIRED_T in magnitude in the same
    direction; the candidate wins on a positive decided gap, the incumbent
    stands on a negative one; otherwise a tie (the incumbent keeps its
    place, nothing is claimed). Returns (verdict, gap, threshold)."""
    gap = score_c - score_i
    thr = BAND_Z * float(np.sqrt(band_c["sd16"] ** 2 + band_i["sd16"] ** 2))
    weeks_agree = paired_t is None or (abs(paired_t) >= PAIRED_T and np.sign(paired_t) == np.sign(gap))
    if gap > thr and weeks_agree:
        return "candidate", gap, thr
    if gap < -thr and weeks_agree:
        return "incumbent", gap, thr
    return "tie", gap, thr


def merged_copies(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """One walk with both seed sets' copies (c1..c16 + c17..c32) on the
    common (week, ticker) rows: the incumbent as a 32-copy ensemble instead
    of a race between its two sets (the owner's bias finding, 2026-09-19:
    "beat both" lets the incumbent stand two times in three on a tie)."""
    ca = [c for c in a.columns if c.startswith("c") and c[1:].isdigit()]
    cb = [c for c in b.columns if c.startswith("c") and c[1:].isdigit()]
    b = b.rename(columns={c: f"c{len(ca) + i}" for i, c in enumerate(cb, 1)})     # c1..c16 + c17..c32
    return a[["week", "ticker", *ca]].merge(b[["week", "ticker", *b.columns[2:]]], on=["week", "ticker"], how="inner")


def twin_dir(incumbent: Path) -> Path:
    """The seed twin lives beside the incumbent's walk: <walk>/twin/."""
    return Path(incumbent).parent.parent / "twin"


def twin_copies() -> range:
    return range(FINAL_K + 1, 2 * FINAL_K + 1)


def ensure_twin(incumbent: Path, inc_rec: dict, store: str, lo, hi, workers: int = 1, log=print,
                refit_every: int = 1) -> Path:
    """Walk the incumbent's recipe with fresh seeds on every week of the
    selection window (resumes if partly done; instant if complete)."""
    d = twin_dir(incumbent)
    log(f"seed twin: {inc_rec} copies {twin_copies().start}..{twin_copies().stop - 1} -> {d}")
    walk(store, lo, hi, inc_rec["label"], inc_rec["train_years"], FINAL_K, d, log=log,
         features=inc_rec.get("features", ()), params=inc_rec.get("params"), workers=workers,
         copies=twin_copies(), drop=inc_rec.get("drop", ()), train_top=inc_rec.get("train_top"), refit_every=refit_every)
    return d / "preds.parquet"


def null_gaps(sel, ctx, incumbent: Path, twin: Path, per_year: int, lo=SELECT[0], hi=SELECT[1],
              draws: int = NULL_DRAWS, log=print, k: int = FINAL_K, block: int = 1) -> np.ndarray:
    """The model-score gap of the seed twin over the incumbent on `draws`
    stratified samples (seeds 1..draws; the candidates' draw is seed 0) —
    what a boost of zero looks like under this sample design. Cached at
    <twin dir>/null_<per_year>.json."""
    cache = Path(twin).parent / (f"null_{per_year}" + (f"_b{block}" if block > 1 else "") + (f"_k{k}" if k != FINAL_K else "") + ".json")
    if cache.exists():
        rec = json.loads(cache.read_text())
        if rec["draws"] == draws and rec["k"] == k:
            return np.array(rec["gaps"])
    hi_ = holdings(sel, ctx, load_preds([incumbent]), range(1, k + 1))
    ht = holdings(sel, ctx, load_preds([twin]), range(twin_copies().start, twin_copies().start + k))
    common = set(hi_.week) & set(ht.week)
    hi_, ht = hi_[hi_.week.isin(common)], ht[ht.week.isin(common)]
    gaps = []
    for sd in range(1, draws + 1):
        w = set(stratified_blocks(ctx, lo, hi, per_year, sd, block) if block > 1 else stratified_weeks(ctx, lo, hi, per_year, sd))
        gaps.append(model_score(selection_metric(sel, ht[ht.week.isin(w)], lo, hi))
                    - model_score(selection_metric(sel, hi_[hi_.week.isin(w)], lo, hi)))
    gaps = np.array(gaps)
    seed_luck = model_score(selection_metric(sel, ht, lo, hi)) - model_score(selection_metric(sel, hi_, lo, hi))
    cache.write_text(json.dumps({"draws": draws, "k": k, "block": block, "per_year": per_year, "weeks": len(common),
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
             log=print, workers: int = 1, refit_every: int = 1, k: int = FAST_K, incumbent_recipe_ok: bool = False) -> dict:
    """`stocks-ml challenge-fast`: the candidates on a stratified random
    sample of the selection window at K=FAST_K against the incumbent on
    the same weeks; ranked by the model score; nothing decided."""
    lo, hi, out, incumbent = pd.Timestamp(lo), pd.Timestamp(hi), Path(out), Path(incumbent)
    out.mkdir(parents=True, exist_ok=True)
    inc_rec = incumbent_recipe(incumbent, allow_other=incumbent_recipe_ok)
    names = {candidate_name(c): c for c in candidates}
    if len(names) != len(candidates):
        raise SystemExit("two candidates have the same recipe")
    sel, ctx, _ = context(store)
    worlds = Worlds(sel, ctx, store)
    refuse_leaky_features_by_world(worlds, list(names.values()), lo, hi, log)
    refit_every = max(1, int(refit_every))
    if refit_every > 1:
        weeks = stratified_blocks(ctx, lo, hi, per_year, seed, refit_every)
        desc = (f"{len(weeks)} weeks in runs of {refit_every}, a stratified random sample ({per_year} per year, "
                f"seed {seed}), one fit per run (screening cadence)")
    else:
        weeks = stratified_weeks(ctx, lo, hi, per_year, seed)
        desc = f"{len(weeks)} weeks, a stratified random sample ({per_year} per year, seed {seed})"
    log(f"challenge-fast: {len(names)} candidates vs the incumbent {incumbent} ({inc_rec}) on {desc} "
        f"of {lo.date()} -> {hi.date()} at K={k}; ranks only, decides nothing")
    twin = ensure_twin(incumbent, inc_rec, store, lo, hi, workers=workers, log=log, refit_every=refit_every)
    null = null_gaps(sel, ctx, incumbent, twin, per_year, lo, hi, log=log, k=k, block=refit_every)
    null = null - null.mean()                  # centred: the spread of a zero boost under this design
    thr = float(np.percentile(null, FLAG_PERCENTILE))
    luck = seed_luck(twin, per_year)
    tw = range(twin_copies().start, twin_copies().start + k)
    walks = {"incumbent": cut_walk(incumbent, weeks, k),
             "incumbent (twin seeds)": cut_walk(twin, weeks, k, copies=tw)}
    for n, c in names.items():
        log(f"fast: {n}" + (f" on {c['store']}" if c.get("store") else ""))
        tag = f"fast_{per_year}x_s{seed}" + (f"_r{refit_every}" if refit_every > 1 else "") + (f"_k{k}" if k != FAST_K else "")
        p = walk(worlds.store_of(c), lo, hi, c["label"], c["train_years"], k, out / n / tag,
                 log=log, features=c.get("features", ()), params=c.get("params"), drop=c.get("drop", ()), train_top=c.get("train_top"),
                 weeks=weeks, sample=desc, workers=workers, refit_every=refit_every)
        walks[n] = load_preds([p])
    ctxs = {n: worlds.ctx_of(c) for n, c in names.items()}
    M, H, n_common = metrics_on_common(sel, ctx, walks, k, lo, hi, ctxs=ctxs)
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
                   "copies": copy_metrics(sel, ctxs.get(n, ctx), walks[n], k, set(H[n].week), lo, hi),
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
           "k": k, "refit_every": refit_every, "incumbent": {"preds": str(incumbent), "recipe": inc_rec},
           "null": {"twin": str(twin), "draws": int(len(null)), "flag_percentile": FLAG_PERCENTILE,
                    "threshold": round(thr, 2), "centred": True, "seed_luck_full_walk": luck,
                    "reference": "the higher-scoring of the incumbent's two seed sets on the draw",
                    "percentiles": {str(q): round(float(np.percentile(null, q)), 2) for q in (10, 50, 90, 95)}},
           "candidates": names, "rows": rows, "order": order, "flagged": flagged, "md": md}
    tag = f"fast_{per_year}x_s{seed}" + (f"_r{refit_every}" if refit_every > 1 else "") + (f"_k{k}" if k != FAST_K else "")
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1, default=str))
    for n in order:
        record_trials([{"kind": "challenge_fast", "name": f"challenge_fast_{out.name}_{tag}_{n}",
                        "cv_metric": rows[n]["score"], "config": names[n],
                        "notes": json.dumps({**rows[n], "sample": desc, "flag_threshold": round(thr, 2)},
                                            default=str)}])
    log(f"challenge-fast -> {out / f'{tag}.json'}")
    return res


def window_table(sel, ctx, ctx_c, name: str, inc_select: Path, cand_select: Path, H: dict, lo, hi) -> dict:
    """Each model's strategy on the window at the settings the procedure
    decides on ITS OWN 2006-2015 walk, plus the sp500 row. Returns
    {rows, md, settings}."""
    from stocks_ml.backtest import own_settings, row_vs_spy, settings_label, simulate_holdings, table_md
    span = f"{lo.year}-{hi.year}"
    spans = {span: (lo, hi + pd.Timedelta(days=1))}
    sel_weeks = lambda c: [t for t in c.weeks if SELECT[0] <= t <= SELECT[1]]  # noqa: E731
    st_inc = own_settings(sel, ctx, holdings(sel, ctx, cut_walk(inc_select, sel_weeks(ctx), FINAL_K), range(1, FINAL_K + 1)))
    st_c = own_settings(sel, ctx_c, holdings(sel, ctx_c, load_preds([cand_select]), range(1, FINAL_K + 1)))
    r_inc = simulate_holdings(sel, ctx, H["incumbent"], st_inc)
    r_c = simulate_holdings(sel, ctx_c, H[name], st_c)
    spy = ctx.wret["SPY"]
    rows = {f"incumbent ({settings_label(st_inc)})": row_vs_spy(sel, r_inc, spy, spans),
            f"{name} ({settings_label(st_c)})": row_vs_spy(sel, r_c, spy, spans),
            "sp500": {span: sel.metrics(spy.reindex(r_inc.index), *spans[span])}}
    return {"rows": rows, "md": table_md(rows, (span,)), "settings": {"incumbent": st_inc, name: st_c}}


def adjudicate(name: str, cand: dict, incumbent: Path, out: Path, worlds: "Worlds", log=print,
               workers: int = 1, lo=ADJUDICATE[0], hi=ADJUDICATE[1], refit_every: int = 1) -> dict:
    """The head-to-head on the adjudication window (see ADJUDICATE): the
    incumbent's own extend walk cut to the window, its seed twin walked on
    the window, the candidate walked on the window; the model score on the
    common weeks decides — the candidate must beat both seed sets."""
    sel, ctx, store = worlds.sel, worlds.base, worlds.store
    inc_rec = incumbent_recipe(incumbent, allow_other=True)      # run() checked it
    ext = incumbent.parent.parent / "extend" / "preds.parquet"
    if not ext.exists():
        raise SystemExit(f"the incumbent has no extend walk at {ext}: the adjudication window needs it")
    weeks = [t for t in ctx.weeks if lo <= t <= hi]
    log(f"adjudication {lo.date()} -> {hi.date()} ({len(weeks)} weeks, K={FINAL_K}): {name} vs the incumbent, "
        f"neither selected on this window; each at its own procedure-decided settings")
    walks = {"incumbent": cut_walk(ext, weeks, FINAL_K)}
    twin = walk(store, lo, hi, inc_rec["label"], inc_rec["train_years"], FINAL_K,
                incumbent.parent.parent / "twin_adjudicate", log=log, features=inc_rec.get("features", ()),
                params=inc_rec.get("params"), workers=workers, copies=twin_copies(), drop=inc_rec.get("drop", ()), train_top=inc_rec.get("train_top"),
                refit_every=refit_every)
    walks["incumbent"] = merged_copies(walks["incumbent"], cut_walk(twin, weeks, FINAL_K, copies=twin_copies()))   # 32 copies
    p = walk(worlds.store_of(cand), lo, hi, cand["label"], cand["train_years"], FINAL_K, out / name / "adjudicate",
             log=log, features=cand.get("features", ()), params=cand.get("params"), drop=cand.get("drop", ()), train_top=cand.get("train_top"),
             workers=workers, refit_every=refit_every)
    walks[name] = load_preds([p])
    ctx_c = worlds.ctx_of(cand)
    k_of = {"incumbent": 2 * FINAL_K, name: FINAL_K}
    M, H, n = metrics_on_common(sel, ctx, walks, k_of, lo, hi, ctxs={name: ctx_c})
    common = set(H["incumbent"].week)
    scores = {k: round(model_score(M[k]), 2) for k in walks}
    bands = {"incumbent": seed_band(sel, ctx, walks["incumbent"], range(1, 2 * FINAL_K + 1), common, lo, hi),
             name: seed_band(sel, ctx_c, walks[name], range(1, FINAL_K + 1), common, lo, hi)}
    t_pair = round(paired_t_books(H[name], H["incumbent"]), 2)
    verdict, gap, thr = decide(scores[name], scores["incumbent"], bands[name], bands["incumbent"], t_pair)
    winner = name if verdict == "candidate" else "incumbent"
    tab = window_table(sel, ctx, ctx_c, name, incumbent, out / name / "select" / "preds.parquet", H, lo, hi)
    md = ["| model | top-3 | top-6 | top-10 | score (mean of the three) | seed band (sd of the score at K) | paired t vs incumbent |", "|---|---|---|---|---|---|---|"]
    for k in walks:
        md.append(f"| {'**' + k + '**' if k == winner else k} | "
                  + " | ".join(f"{M[k].get(b, float('nan')):+.2f}" for b in SCORE_BOOKS)
                  + f" | **{scores[k]:+.2f}** | ±{bands[k]['sd16']:.2f} (K={k_of[k]}; half-ensembles {bands[k]['half_mean']:+.2f} ± {bands[k]['half_sd']:.2f})"
                  + f" | {'—' if k == 'incumbent' else f'{paired_t_books(H[k], H['incumbent']):+.2f}'} |")
    log(f"adjudication ({n} common weeks): the incumbent (both seed sets, K={2 * FINAL_K}) {scores['incumbent']:+.2f} ± {bands['incumbent']['sd16']:.2f}; "
        f"{name} {scores[name]:+.2f} ± {bands[name]['sd16']:.2f}; gap {gap:+.2f}, decided beyond ±{thr:.2f} with paired t {t_pair:+.2f} (|t| >= {PAIRED_T} needed) -> "
        + (f"{name} wins" if verdict == "candidate" else "the incumbent stands" if verdict == "incumbent" else "a TIE (inside seed luck)"))
    log("\n".join(md))
    log("strategy on the window, each at its own settings:\n" + "\n".join(tab["md"]))
    res = {"window": [str(lo.date()), str(hi.date())], "weeks": n, "metric": M, "scores": scores, "bands": bands,
           "verdict": verdict, "gap": round(gap, 2), "threshold": round(thr, 2),
           "winner": winner, "md": md, "strategy": tab, "twin": str(twin), "walk": str(p)}
    _ledger(out, "adjudicate", name, cand, {"metric": M[name], "scores": scores, "winner": winner,
                                             "window": res["window"], "weeks": n})
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
    if list(rec.get("drop") or []) != list(base.get("drop") or []):
        parts.append("drop=" + "+".join(rec.get("drop") or []))
    if rec.get("store"):
        parts.append(f"store={rec['store']}")
    if rec.get("train_top") != base.get("train_top"):
        parts.append(f"train_top={rec.get('train_top')}")
    return ",".join(parts)


def _ledger(out: Path, stage: str, name: str, rec: dict, note: dict) -> None:
    record_trials([{"kind": "challenge", "name": f"challenge_{out.name}_{stage}_{name}",
                    "cv_metric": model_score(note.get("metric", {})) if note.get("metric") else None,
                    "config": rec, "notes": json.dumps(note, default=str)}])


def run(candidates: list, incumbent: Path, out: Path, store: str = STORE, k16: bool = False,
        lo=SELECT[0], hi=SELECT[1], log=print, workers: int = 1, adjudicate_window: bool = False,
        refit_every: int = 1, incumbent_recipe_ok: bool = False) -> dict:
    lo, hi, out, incumbent = pd.Timestamp(lo), pd.Timestamp(hi), Path(out), Path(incumbent)
    out.mkdir(parents=True, exist_ok=True)
    inc_rec = incumbent_recipe(incumbent, allow_other=incumbent_recipe_ok)
    names = {candidate_name(c): c for c in candidates}
    if len(names) != len(candidates):
        raise SystemExit("two candidates have the same recipe")
    for n, c in names.items():
        if c == recipe(inc_rec["label"], inc_rec["train_years"], inc_rec.get("features", ()),
                       inc_rec.get("params"), inc_rec.get("drop", ()), inc_rec.get("train_top")):
            raise SystemExit(f"candidate {n} is the incumbent's own recipe")
    sel, ctx, _ = context(store)
    worlds = Worlds(sel, ctx, store)
    refuse_leaky_features_by_world(worlds, list(names.values()), lo, hi, log)
    res = {"out": str(out), "store": store, "selection_window": [str(lo.date()), str(hi.date())],
           "rules": {"screen": None, "full_k": FULL_K, "score_books": SCORE_BOOKS, "final_k": FINAL_K,
                     "refit_every": int(refit_every)},
           "incumbent": {"preds": str(incumbent), "recipe": inc_rec}, "candidates": names}
    log(f"challenge: {len(names)} candidates vs the incumbent {incumbent} ({inc_rec}); "
        f"selection window {lo.date()} -> {hi.date()}; rules {res['rules']}")

    # the incumbent at the candidates' cadence: a weekly walk against cadence-4 candidates (or the reverse)
    # would mix two fitting schedules in one table
    inc_cadence = int(json.loads((incumbent.parent / "spec.json").read_text()).get("refit_every", 1))
    if inc_cadence != int(refit_every):
        raise SystemExit(f"the incumbent {incumbent} was walked with a refit every {inc_cadence} week(s), the candidates "
                         f"would be walked every {int(refit_every)}: pass --refit-every {inc_cadence}, or an incumbent "
                         f"walked at the candidates' cadence")
    ctxs = {n: worlds.ctx_of(c) for n, c in names.items()}

    # ---- stage 2: every week, the frozen comparison
    walks = {"incumbent": cut_walk(incumbent, [t for t in ctx.weeks if lo <= t <= hi], FULL_K)}
    if FULL_K == FINAL_K:     # the incumbent is a recipe, not one seed set: both count (walked if missing —
        twin = ensure_twin(incumbent, inc_rec, store, lo, hi, workers=workers, log=log, refit_every=refit_every)
        walks["incumbent (twin seeds)"] = cut_walk(twin, [t for t in ctx.weeks if lo <= t <= hi], FULL_K,
                                                   copies=twin_copies())    # until 2026-09-24 only if present)
    for n in names:
        c = names[n]
        log(f"stage 2: {n} — every week at K={FULL_K}" + (f" on {c['store']}" if c.get("store") else ""))
        p = walk(worlds.store_of(c), lo, hi, c["label"], c["train_years"], FULL_K, out / n / "select",
                 log=log, features=c.get("features", ()), params=c.get("params"), drop=c.get("drop", ()), train_top=c.get("train_top"), workers=workers,
                 refit_every=refit_every)
        walks[n] = load_preds([p])
    if "incumbent (twin seeds)" in walks:            # one 32-copy incumbent, not a race between two seed sets
        walks["incumbent"] = merged_copies(walks["incumbent"], walks.pop("incumbent (twin seeds)"))
    k_of = {n: (2 * FULL_K if n == "incumbent" and walks[n].shape[1] > FULL_K + 2 else FULL_K) for n in walks}
    M2, H2, n2 = metrics_on_common(sel, ctx, walks, k_of, lo, hi, ctxs=ctxs)
    common2 = set(H2["incumbent"].week)
    incumbents = ["incumbent"]
    detail = {}
    for n in [*incumbents, *names]:
        preds = walks[n]
        detail[n] = {"metric": M2[n], "k": k_of[n],
                     "copies": copy_metrics(sel, ctxs.get(n, ctx), preds, k_of[n], set(H2[n].week), lo, hi),
                     "band": seed_band(sel, ctxs.get(n, ctx), preds, range(1, k_of[n] + 1), common2, lo, hi),
                     "paired_t_vs_incumbent": (None if n == "incumbent"
                                               else round(paired_t_books(H2[n], H2["incumbent"]), 2))}
        if n not in incumbents:
            la = audit_segments(worlds.store_of(names[n]), [str(out / n / "select" / "preds.parquet")], ctxs[n])
            detail[n]["leak_audit"] = {"verdict": la["VERDICT"], "line": leak_line(la)}
    eligible = incumbents + [n for n in names if detail[n]["leak_audit"]["verdict"] == "PASS"]
    order2 = rank_by_metric(M2, eligible)
    best = next((n for n in order2 if n not in incumbents), None)
    verdict, gap, thr = ("incumbent", 0.0, 0.0) if best is None else decide(
        model_score(M2[best]), model_score(M2["incumbent"]), detail[best]["band"], detail["incumbent"]["band"],
        detail[best]["paired_t_vs_incumbent"])
    winner = best if verdict == "candidate" else "incumbent"
    res["stage2"] = {"weeks": n2, "detail": detail, "order": order2, "winner": winner, "verdict": verdict,
                     "gap": round(gap, 2), "threshold": round(thr, 2), "best_candidate": best}
    md = [f"| model | top-3 | top-6 | top-10 | score (mean of the three) | seed band (sd of the score at K) | copies | paired t vs incumbent | leak audit |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in [*incumbents, *names]:
        d = detail[n]
        md.append(f"| {'**' + n + '**' if n == winner else n} | " +
                  " | ".join(f"{d['metric'].get(b, float('nan')):+.2f}" for b in SCORE_BOOKS) +
                  f" | **{model_score(d['metric']):+.2f}** | ±{d['band']['sd16']:.2f} (half-ensembles {d['band']['half_mean']:+.2f} ± {d['band']['half_sd']:.2f}) | "
                  f"K={d['k']}: " + ", ".join(f"{v:+.1f}" for v in d["copies"]) +
                  f" | {'—' if d['paired_t_vs_incumbent'] is None else f'{d['paired_t_vs_incumbent']:+.2f}'}"
                  f" | {d.get('leak_audit', {}).get('verdict', '—')} |")
    res["stage2"]["md"] = md
    log(f"stage 2 ({n2} common weeks, every week): "
        + (f"{best} vs the incumbent: gap {gap:+.2f}, decided beyond ±{thr:.2f} ({BAND_Z} x the combined seed sd) "
           f"with paired t {detail[best]['paired_t_vs_incumbent']:+.2f} (|t| >= {PAIRED_T} needed) -> "
           f"{'the candidate wins' if verdict == 'candidate' else 'the incumbent stands' if verdict == 'incumbent' else 'a TIE (inside seed luck; the incumbent keeps its place, nothing is claimed)'}"
           if best else "no eligible candidate"))
    log("\n".join(md))
    for n in names:
        _ledger(out, "stage2", n, names[n], {**detail[n], "weeks": n2, "winner": n == winner})
    (out / "challenge.json").write_text(json.dumps(res, indent=1, default=str))

    # ---- the head-to-head on the adjudication window (owner's rule 2026-09-18)
    if adjudicate_window:
        ranked = [n for n in order2 if n not in incumbents]        # the challenger's winner, chosen on 2006-2015
        if not ranked:
            log("adjudication: no candidate passed the leak audit; nothing to adjudicate")
        else:
            res["adjudication"] = adjudicate(ranked[0], names[ranked[0]], incumbent, out, worlds, log=log, workers=workers,
                                             refit_every=refit_every)
            winner = res["adjudication"]["winner"]
            res["stage2"]["winner_before_adjudication"] = res["stage2"]["winner"]
            res["stage2"]["winner"] = winner
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
            walk(worlds.store_of(c), a, b, c["label"], c["train_years"], FINAL_K, wdir / seg, log=log,
                 features=c.get("features", ()), params=c.get("params"), drop=c.get("drop", ()), train_top=c.get("train_top"), workers=workers)
        res["stage3"] = {"walk": str(wdir),
                         "eval": eval_run(wdir, incumbent.parent.parent, store=worlds.store_of(c), k=FINAL_K, log=log)}
        (out / "challenge.json").write_text(json.dumps(res, indent=1, default=str))
        log(f"stage 3 done: adopt with `stocks-ml procedure --preds {wdir / 'select' / 'preds.parquet'}` "
            "on the owner's go")
    elif winner != "incumbent":
        log(f"next: `stocks-ml challenge ... --k16` (or train {winner} at K={FINAL_K} on both segments "
            f"and run `stocks-ml eval --walk <walk> --incumbent {incumbent.parent.parent}`)")
    log(f"challenge -> {out / 'challenge.json'}")
    return res
