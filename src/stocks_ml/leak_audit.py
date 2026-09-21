"""The leak audit — run by `stocks-ml eval`, required before any adoption.

A walk's scores are tested against known future-information proxies on
pre-holdout weeks only. Per walk segment:

1. FACTOR (report-only): the vendor adjustment factor log(close/closeunadj)
   at the rank date — the restatement a look-ahead basis leak rides on (the
   2026-09 split leak). Reported: mean weekly Spearman(score, factor); the
   score's weekly IC; the IC after residualizing score ranks on factor
   ranks each week; the retention ratio. Report-only since 2026-09-12: a
   model that favours strong, rising companies favours future splitters for
   legitimate reasons, and the future split factor is partly future return,
   so residualizing it removes real skill too. The number is kept so it
   stays comparable (the leaky champion retained 0.49).
2. DELISTING (report-only): names whose last print falls within 8 weeks of
   the rank date — their rate among the top-15 vs the member universe.
3. IDENTITY (the gate): the mechanism that removed the leak, checked where
   it matters most. The panel's f_sf per-share ratios divide a restated
   per-share value by the SPLIT-adjusted close (close_split), so the
   restatement cancels and the ratio equals what the live job computes
   (features/sharadar_fundamentals.py, data/world.py). At the segment's
   middle rank week the audit recomputes book-to-market as
   bvps_asof / close_split for every member, ranks it within the week, and
   requires the stored panel column to agree to the float; the three
   members with the largest in-window future split factors are reported
   with their inputs. Exact; the only gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.selection import HOLDOUT_START

RETENTION_REFERENCE = 0.75   # report-only since 2026-09-12; kept so the number stays comparable
DELIST_DAYS = 56


def nw_t(x: pd.Series, lags: int = 4) -> float:
    """Newey-West t of the mean of a weekly series."""
    x = x.dropna().to_numpy(float)
    n = len(x)
    if n < 10:
        return float("nan")
    d = x - x.mean()
    v = float(d @ d) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * float(d[:-k] @ d[k:]) / n
    return float(x.mean() / np.sqrt(v / n))


def identity_check(ctx, prices: pd.DataFrame, fund: pd.DataFrame, week: pd.Timestamp) -> dict:
    """Recompute f_sf_book_to_market at `week` from the store's inputs the way
    the panel builder does (bvps as-of, split-adjusted close, week rank) and
    compare with the stored panel; report the top-3 future splitters."""
    from stocks_ml.features.ranking import rank_normalize
    from stocks_ml.features.sharadar_fundamentals import _asof, prepared_arq
    rows = ctx.pan[ctx.pan["date"] == week]
    if rows.empty or "f_sf_book_to_market" not in rows.columns:
        return {"week": str(week.date()), "ok": False, "reason": "no panel rows / column at this week"}
    base = rows[["date", "ticker"]].copy()
    bvps = _asof(base, prepared_arq(fund), ["bvps"])["bvps"]     # the builder's own ARQ frame
    day = prices[prices["date"] <= week].sort_values("date").groupby("ticker").last()
    close_split = day["close_split"].reindex(base["ticker"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = bvps.to_numpy() / np.where(close_split == 0, np.nan, close_split)
    calc = rank_normalize(pd.DataFrame({"date": base["date"].to_numpy(), "ticker": base["ticker"].to_numpy(),
                                        "f_sf_book_to_market": ratio}), ["f_sf_book_to_market"])
    stored = rows["f_sf_book_to_market"].to_numpy(float)
    got = calc["f_sf_book_to_market"].to_numpy(float)
    diff = np.abs(stored - got)
    ok = bool(np.nanmax(diff) <= 1e-9) if len(diff) else False
    factor = (day["closeunadj"] / day["close_split"]).reindex(base["ticker"])
    top = factor.nlargest(3)
    detail = {}
    for tk, f in top.items():
        i = int(np.flatnonzero(base["ticker"].to_numpy() == tk)[0])
        detail[tk] = {"bvps_asof": None if pd.isna(bvps.iloc[i]) else round(float(bvps.iloc[i]), 4),
                      "close_split": round(float(close_split[i]), 4),
                      "future_split_factor": round(float(f), 3),
                      "stored_rank": round(float(stored[i]), 6), "recomputed_rank": round(float(got[i]), 6)}
    return {"week": str(week.date()), "members": int(len(rows)), "max_abs_rank_diff": float(np.nanmax(diff)),
            "top_splitters": detail, "ok": ok}


def audit(store: str, preds_path: str, ctx=None) -> dict:
    """One walk segment's audit (see the module docstring)."""
    import stocks_ml.selection as sel
    preds = pd.read_parquet(preds_path)
    preds["week"] = pd.to_datetime(preds["week"])
    preds = preds.sort_values(["week", "ticker"])
    if preds["week"].max() >= HOLDOUT_START:
        raise RuntimeError(f"{preds_path} reaches {preds['week'].max().date()}: the audit reads "
                           f"pre-holdout weeks only (< {HOLDOUT_START.date()})")
    copies = [c for c in preds.columns if c.startswith("c") and c[1:].isdigit()]
    preds["score"] = preds[copies].mean(axis=1)

    ctx = ctx or sel.Ctx(store)
    prices = ctx.prices
    if not {"closeunadj", "close_split"} <= set(prices.columns):
        raise RuntimeError("prices table lacks closeunadj/close_split: rebuild the world "
                           "(stocks-ml world) — the IDENTITY check needs them")
    pr = prices.dropna(subset=["closeunadj"])
    fac = np.log(pr["close"] / pr["closeunadj"]).groupby([pr["date"], pr["ticker"]]).last()
    last_print = prices.groupby("ticker")["date"].max()
    censor = prices["date"].max() - pd.Timedelta(days=DELIST_DAYS + 30)

    fwd = ctx.fwd["4w"]
    rows = []
    for wk, g in preds.groupby("week"):
        try:
            f = fac.loc[wk]
        except KeyError:
            continue
        slot = sel.week_slot(fwd.index, wk)
        if slot is None:
            continue
        r = fwd.loc[slot]
        d = pd.DataFrame({"score": g.set_index("ticker")["score"]})
        d["factor"] = f.reindex(d.index)
        d["ret"] = r.reindex(d.index)
        d = d.dropna(subset=["score", "factor"])
        if len(d) < 100:
            continue
        rs, rf = d["score"].rank(), d["factor"].rank()
        sp = float(rs.corr(rf))
        lab = d.dropna(subset=["ret"])
        ic = float(lab["score"].rank().corr(lab["ret"].rank())) if len(lab) > 50 else np.nan
        b = float(np.cov(rs, rf)[0, 1] / np.var(rf)) if float(np.var(rf)) > 0 else 0.0
        d["resid"] = rs - b * rf
        labr = d.dropna(subset=["ret"])
        ic_r = float(labr["resid"].rank().corr(labr["ret"].rank())) if len(labr) > 50 else np.nan
        top15 = g.nlargest(15, "score")["ticker"]
        if wk < censor:
            gone = last_print.reindex(g["ticker"]).lt(wk + pd.Timedelta(days=DELIST_DAYS))
            gone_top = last_print.reindex(top15).lt(wk + pd.Timedelta(days=DELIST_DAYS))
            rows.append([wk, sp, ic, ic_r, float(gone.mean()), float(gone_top.mean())])
        else:
            rows.append([wk, sp, ic, ic_r, np.nan, np.nan])
    t = pd.DataFrame(rows, columns=["week", "spearman_factor", "ic", "ic_resid",
                                    "delist_uni", "delist_top15"]).set_index("week")
    ic, ic_r = t["ic"].mean(), t["ic_resid"].mean()
    retention = float(ic_r / ic) if ic and np.isfinite(ic) and abs(ic) > 1e-6 else float("nan")

    weeks = sorted(preds["week"].unique())
    ident = identity_check(ctx, prices, ctx.fund, pd.Timestamp(weeks[len(weeks) // 2]))
    return {
        "weeks": int(len(t)),
        "span": f"{t.index.min().date()} -> {t.index.max().date()}",
        "spearman_score_vs_factor": round(float(t["spearman_factor"].mean()), 4),
        "spearman_t": round(nw_t(t["spearman_factor"]), 2),
        "ic": round(float(ic), 5), "ic_t": round(nw_t(t["ic"]), 2),
        "ic_resid_factor": round(float(ic_r), 5), "ic_resid_t": round(nw_t(t["ic_resid"]), 2),
        "ic_retention": round(retention, 3),
        "retention_reference": f">= {RETENTION_REFERENCE} (report-only since 2026-09-12)",
        "FACTOR": "report-only",
        "delist_within_8w_universe": round(float(t["delist_uni"].mean()), 5),
        "delist_within_8w_top15": round(float(t["delist_top15"].mean()), 5),
        "identity": ident,
        "IDENTITY": "PASS" if ident["ok"] else "FAIL",
        "VERDICT": "PASS" if ident["ok"] else "FAIL",
    }


FEATURE_FACTOR_LIMIT = 0.15      # |mean weekly Spearman(feature, future split factor)| above this = leaky
LIVE_ARCHIVE = "archive"         # <live store>/archive/features_<date>.parquet: the rows the live job ranked


def archive_live_rows(live_dir, ctx, t, columns=None) -> Path:
    """Save the panel rows the live job scored at rank week t, as they were
    computed THAT week. A later panel build re-derives the same rows with
    more future known (splits, restatements); the drift between the two is
    a leak detector (owner's idea 2026-09-19: "compare live data to
    historic data; if they look different something is awry")."""
    d = Path(live_dir) / LIVE_ARCHIVE
    d.mkdir(parents=True, exist_ok=True)
    rows = ctx.pan[ctx.pan["date"] == pd.Timestamp(t)]
    if columns:
        rows = rows[["date", "ticker", *[c for c in columns if c in rows.columns]]]
    path = d / f"features_{pd.Timestamp(t).date()}.parquet"
    rows.to_parquet(path, index=False)
    return path


def live_vs_rebuilt(live_dir, panel: pd.DataFrame, features=None) -> pd.DataFrame:
    """Per feature: how the archived live rows compare with the same rows in
    `panel` (a later build): the mean weekly Spearman between the two
    versions and the share of names whose value moved by more than 0.1 rank
    units. Ranked features should agree to ~1.0; a feature whose history is
    rewritten by later data (a mixed-basis price, a restated fundamental)
    reads lower. Empty when nothing is archived."""
    files = sorted(Path(live_dir, LIVE_ARCHIVE).glob("features_*.parquet")) if Path(live_dir, LIVE_ARCHIVE).exists() else []
    if not files:
        return pd.DataFrame(columns=["feature", "weeks", "rank_agreement", "share_moved_0.1"])
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    rows = []
    for f in files:
        a = pd.read_parquet(f); a["date"] = pd.to_datetime(a["date"])
        t = a["date"].iloc[0]
        b = panel[panel["date"] == t]
        j = a.merge(b, on=["date", "ticker"], suffixes=("_live", "_rebuilt"))
        if len(j) < 50:
            continue
        cols = [c for c in (features or [c for c in a.columns if c.startswith(("f_", "x_"))]) if f"{c}_live" in j.columns]
        for c in cols:
            x, y = j[f"{c}_live"], j[f"{c}_rebuilt"]
            ok = x.notna() & y.notna()
            if ok.sum() < 50 or x[ok].nunique() < 5 or y[ok].nunique() < 5:
                continue
            rows.append({"feature": c, "week": t, "rank_agreement": float(x[ok].rank().corr(y[ok].rank())),
                         "share_moved_0.1": float(((x[ok] - y[ok]).abs() > 0.1).mean())})
    if not rows:
        return pd.DataFrame(columns=["feature", "weeks", "rank_agreement", "share_moved_0.1"])
    t = pd.DataFrame(rows).groupby("feature").agg(weeks=("week", "nunique"), rank_agreement=("rank_agreement", "mean"),
                                                  **{"share_moved_0.1": ("share_moved_0.1", "mean")}).reset_index()
    return t.sort_values("rank_agreement").reset_index(drop=True)


def feature_factor_check(ctx, features, lo="2006-01-01", hi="2015-12-31") -> dict:
    """Each named extra feature's mean weekly Spearman with the vendor's
    future adjustment factor log(close_split / closeunadj) at the rank date
    on the selection window — the channel of the 2026-09 split leak. The
    panel's own f_ columns sit within ±0.05; a mixed-basis level feature
    reads ±0.6. `challenge` and `challenge-fast` refuse a recipe whose
    feature exceeds FEATURE_FACTOR_LIMIT."""
    from scipy.stats import spearmanr
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    px = ctx.prices[["date", "ticker", "close_split", "closeunadj"]].copy()
    px["factor"] = np.log(px["close_split"] / px["closeunadj"])
    pan = ctx.pan[(ctx.pan.date >= lo) & (ctx.pan.date <= hi)][["date", "ticker", *features]]
    j = pan.merge(px[["date", "ticker", "factor"]], on=["date", "ticker"])
    out = {}
    for c in features:
        ic = j.groupby("date")[[c, "factor"]].apply(
            lambda g: spearmanr(g[c], g["factor"]).correlation
            if g["factor"].nunique() > 5 and g[c].nunique() > 5 else np.nan).dropna()
        out[c] = {"corr_with_future_split_factor": round(float(ic.mean()), 4), "weeks": int(len(ic)),
                  "verdict": "PASS" if abs(ic.mean()) <= FEATURE_FACTOR_LIMIT else "FAIL"}
    out["VERDICT"] = "PASS" if all(v["verdict"] == "PASS" for k, v in out.items() if k != "VERDICT") else "FAIL"
    return out


def scan_windows():
    """The selection window and the pre-holdout extension: a source that
    begins late (FINRA short interest, 2017-12) is invisible on 2006-2015 —
    f_short_dtc sat at +0.35 on 2016-2024 unseen until 2026-09-21."""
    import stocks_ml.selection as sel
    return (("2006-01-01", "2015-12-31"), ("2016-01-01", str(sel.label_end(sel.HOLDOUT_START).date())))


def feature_factor_worst(ctx, features) -> dict:
    """feature_factor_check on every scan window; per feature the
    correlation of largest magnitude and the window it came from."""
    out = {}
    for lo, hi in scan_windows():
        res = feature_factor_check(ctx, list(features), lo, hi)
        for f in features:
            if f not in res:
                continue
            c = res[f]["corr_with_future_split_factor"]
            if c != c:
                continue
            if f not in out or abs(c) > abs(out[f]["corr_with_future_split_factor"]):
                out[f] = {"corr_with_future_split_factor": c, "window": f"{lo[:4]}-{hi[:4]}", "weeks": res[f].get("weeks"),
                          "verdict": "PASS" if abs(c) <= FEATURE_FACTOR_LIMIT else "FAIL"}
    return out


def fundamentals_file(store) -> Path:
    """The fundamentals table a frozen panel was built from: the newest
    `fundamentals.frozen_<date>.parquet` beside it when one exists, else the
    store's live table. (2026-09-18: the S&P store's table is refetched by
    the extras pull and the vendor restates per-share history at a split —
    APH, 2x in 2026-09 — so the live table no longer reproduces a frozen
    panel; the identity gate must read the vintage the panel saw.)"""
    frozen = sorted(Path(store).glob("fundamentals.frozen_*.parquet"))
    return frozen[-1] if frozen else Path(store) / "fundamentals.parquet"


def model_features(preds_path, ctx) -> list[str]:
    """The columns the walk's model saw: the panel's admitted features, minus
    the record's `drop`, plus its `features`."""
    from stocks_ml.features.panel import feature_cols
    rec_path = Path(preds_path).parent / "spec.json"
    rec = json.loads(rec_path.read_text()).get("recipe", {}) if rec_path.exists() else {}
    fc = [c for c in feature_cols(ctx.pan) if c not in set(rec.get("drop") or [])]
    return fc + [c for c in (rec.get("features") or []) if c not in fc and c in ctx.pan.columns]


def feature_scan(ctx, features) -> dict:
    """The split-factor check on the MODEL'S OWN features (owner's rule
    2026-09-19: standing, every audit — until then it was only ever run on
    proposed extras, and f_dollar_vol sat at -0.42 unnoticed), on every scan
    window (scan_windows), the worst reported. Report-only: the identity
    gate decides; features beyond FEATURE_FACTOR_LIMIT are listed for a
    mechanism check (value/quality names split less, so a price-free ratio
    can sit at 0.15-0.21 for economic reasons)."""
    res = feature_factor_worst(ctx, list(features))
    beyond = {f: res[f]["corr_with_future_split_factor"] for f in features
              if f in res and abs(res[f]["corr_with_future_split_factor"]) > FEATURE_FACTOR_LIMIT}
    return {"limit": FEATURE_FACTOR_LIMIT, "n_features": len(features), "windows": [f"{lo[:4]}-{hi[:4]}" for lo, hi in scan_windows()],
            "beyond_limit": dict(sorted(beyond.items(), key=lambda kv: -abs(kv[1]))),
            "all": {f: res[f]["corr_with_future_split_factor"] for f in features if f in res}}


MISSING_T_LIMIT = 3.0        # |NW t| of the blank-minus-filled weekly return gap above this = the blanks predict
MISSING_SURVIVOR_GAP = 0.10  # blank share among names that later left minus names still here (within weeks): flagged above this
MISSING_SURVIVOR_GAP_ALONE = 0.25   # ...and the audit FAILS above this on its own (or above MISSING_SURVIVOR_GAP with |t| beyond MISSING_T_LIMIT)


def missingness_scan(ctx, features, lo="2006-01-01", hi=None) -> dict:
    """Does a feature's ABSENCE carry information? The 2026-09-21 leak: the
    EDGAR tables held only today's index members, so a neutral-filled
    fundamental in 2016 meant "gone by 2026" — every value was point in
    time, the blank was not. Per feature, on member rows of the window: the
    blank share (the week's modal value: the neutral fill, or a no-activity
    zero that ranking turns into a tie group), the blank
    share among names gone by the panel's last week vs names still members
    — compared WITHIN each week and averaged, so a source that begins late
    (short interest, 2017-12) is not mistaken for one keyed to survival —
    and the weekly mean 4-week return of blank rows minus filled rows with
    its Newey-West t. Report-only; a feature beyond either limit
    is listed for a mechanism check. Indicators (three or fewer distinct
    values) are skipped: their zero is a value."""
    import stocks_ml.selection as sel
    lo = pd.Timestamp(lo)
    hi = pd.Timestamp(hi) if hi is not None else sel.label_end(sel.HOLDOUT_START)
    pan = ctx.pan[(ctx.pan.date >= lo) & (ctx.pan.date <= hi)]
    pan = pan[pan.set_index(["date", "ticker"]).index.isin(
        [(d, t) for d in ctx.weeks if lo <= d <= hi for t in ctx.members[d]])]
    stayers = set(ctx.members[ctx.weeks[-1]])
    left = ~pan["ticker"].isin(stayers)
    out = {}
    for c in features:
        if c not in pan.columns:
            continue
        # "blank": the week's modal value — the neutral fill (0.0 after ranking) or a no-activity zero,
        # which ranking turns into a tie group at a week-specific value (f_insider_net_13w: 70% of
        # the names that later left sat in it vs 27% of those that stayed; SEC Form 4 pulled by the
        # current-ticker map, 2026-09-21)
        mode = pan.groupby("date")[c].transform(lambda v: v.mode().iloc[0] if len(v) else np.nan)
        blank = pan[c] == mode
        share = float(blank.mean())
        if share < 0.01 or pan[c].nunique() <= 3:            # no tie group to speak of, or an indicator whose zero is a value
            out[c] = {"blank_share": round(share, 4), "skipped": True}
            continue
        wk = pan.assign(_b=blank).groupby(["date", "_b"])["fwd_ret_4w"].mean().unstack()
        gap = (wk.get(True) - wk.get(False)).dropna() if True in wk.columns and False in wk.columns else pd.Series(dtype=float)
        by_week = (pd.DataFrame({"date": pan["date"], "left": left, "blank": blank.astype(float)})
                     .groupby(["date", "left"])["blank"].mean().unstack())
        both = by_week.dropna() if {True, False} <= set(by_week.columns) else pd.DataFrame()
        out[c] = {"blank_share": round(share, 4),
                  "blank_share_left": round(float(both[True].mean()), 4) if len(both) else None,
                  "blank_share_stayed": round(float(both[False].mean()), 4) if len(both) else None,
                  "return_gap_pp": round(100 * float(gap.mean()), 3) if len(gap) else None,
                  "t": round(nw_t(gap), 2) if len(gap) > 8 else None}
    def beyond(v):
        return (not v.get("skipped")) and ((v["t"] is not None and abs(v["t"]) > MISSING_T_LIMIT) or
                (v["blank_share_left"] is not None and v["blank_share_stayed"] is not None and
                 v["blank_share_left"] - v["blank_share_stayed"] > MISSING_SURVIVOR_GAP))
    flagged = {c: v for c, v in out.items() if beyond(v)}
    def keyed(v):     # the blanks mark the names that leave: a wide gap alone, or a gap with a return difference
        g = (v["blank_share_left"] - v["blank_share_stayed"]) if v["blank_share_left"] is not None and v["blank_share_stayed"] is not None else 0.0
        return g > MISSING_SURVIVOR_GAP_ALONE or (g > MISSING_SURVIVOR_GAP and v["t"] is not None and abs(v["t"]) > MISSING_T_LIMIT)
    survival_keyed = sorted(c for c, v in flagged.items() if keyed(v))
    return {"t_limit": MISSING_T_LIMIT, "survivor_gap_limit": MISSING_SURVIVOR_GAP, "window": [str(lo.date()), str(hi.date())],
            "n_features": len(out), "beyond_limit": flagged, "survival_keyed": survival_keyed,
            "VERDICT": "FAIL" if survival_keyed else "PASS", "all": out}


COVERAGE_TABLES = {"fundamentals": "date", "insiders": "date", "form4": "filed", "shortint": "publication_date",
                   "edgar": "filed", "sec8k": "filed", "holdings": "date", "prices_hl": "date"}
COVERAGE_GAP = 0.10          # leavers' coverage below stayers' by more than this in a year = the table was pulled by survival


def coverage_by_survival(store: str, years=(2008, 2012, 2016, 2019, 2023), tables=None) -> dict:
    """Per raw table and year: the share of that year's members with a row
    in the trailing 12 months, names gone by the membership's last date vs
    names still members. A table pulled for today's members only (the SEC
    tables of the S&P store before 2026-09-21: edgar 0% vs 0% in 2008, then
    leavers far below stayers; Form 4 25-68% vs 90-94%) shows a gap that no
    economics produces. Report-only; tables beyond COVERAGE_GAP in any year
    are listed."""
    from stocks_ml.data.membership import members_asof
    root = Path(store)
    mem = pd.read_parquet(root / "membership.parquet")
    cur = set(mem[mem["end_date"].isna()]["ticker"])
    out, beyond = {}, {}
    for tbl, dc in (tables or COVERAGE_TABLES).items():
        path = root / f"{tbl}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["ticker", dc])
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        rows = {}
        for y in years:
            d = pd.Timestamp(f"{y}-06-30")
            m = members_asof(mem, d)
            has = set(df.loc[(df[dc] <= d) & (df[dc] > d - pd.Timedelta(days=365)), "ticker"])
            left = [t for t in m if t not in cur]
            stay = [t for t in m if t in cur]
            rows[y] = {"left": round(sum(t in has for t in left) / max(len(left), 1), 3),
                       "stayed": round(sum(t in has for t in stay) / max(len(stay), 1), 3)}
        out[tbl] = rows
        worst = max((r["stayed"] - r["left"]) for r in rows.values())
        if worst > COVERAGE_GAP:
            beyond[tbl] = round(worst, 3)
    return {"limit": COVERAGE_GAP, "tables": out, "beyond_limit": beyond}


def audit_segments(store: str, preds_paths, ctx=None) -> dict:
    """One audit PER segment; the verdict is the identity check on every
    segment and the missingness scan's (no survival-keyed blanks). The factor numbers are reported per segment, never pooled:
    the leaky 2026-09 champion showed retention 0.49 on its selection
    window yet ~1 with 2016-2024 pooled in. Every audit also scans the
    model's own features against the future split factor (feature_scan)."""
    import stocks_ml.selection as sel
    ctx = ctx or sel.Ctx(store)
    if not hasattr(ctx, "fund"):
        ctx.fund = pd.read_parquet(fundamentals_file(store))
    segs = {Path(p).parent.name: audit(store, p, ctx) for p in preds_paths}
    for s in segs.values():
        s["fundamentals_file"] = str(fundamentals_file(store))
    feats = model_features(preds_paths[0], ctx)
    scan = feature_scan(ctx, feats)
    missing = missingness_scan(ctx, feats)
    finite = [s["ic_retention"] for s in segs.values() if np.isfinite(s["ic_retention"])]
    # the verdict: the identity gate on every segment, and no feature whose blanks are keyed to
    # survival (missingness_scan: both limits) — the 2026-09-21 EDGAR leak passed the identity gate
    return {"segments": segs, "worst_retention": round(float(min(finite)), 3) if finite else None,
            "feature_scan": scan, "missingness_scan": missing,
            "VERDICT": "PASS" if all(s["VERDICT"] == "PASS" for s in segs.values()) and missing["VERDICT"] == "PASS" else "FAIL"}


def leak_line(la: dict) -> str:
    """One line: the verdict plus the report-only numbers per segment."""
    segs = la.get("segments", {})
    parts = [f"{name}: identity {s['IDENTITY']}, score-vs-split-factor {s['spearman_score_vs_factor']:+.3f} "
             f"(t {s['spearman_t']:+.1f}), IC {s['ic']:+.4f} (t {s['ic_t']:+.1f}), retention {s['ic_retention']}"
             for name, s in segs.items()]
    scan = la.get("feature_scan")
    if scan:
        b = scan["beyond_limit"]
        parts.append(f"feature scan: {len(b)} of {scan['n_features']} beyond {scan['limit']}"
                     + (" (" + ", ".join(f"{f} {v:+.2f}" for f, v in b.items()) + ")" if b else ""))
    miss = la.get("missingness_scan")
    if miss:
        b = miss["beyond_limit"]
        parts.append(f"missingness scan: {len(b)} of {miss['n_features']} whose blanks predict"
                     + (" (" + ", ".join(f"{f} blank {v['blank_share']:.0%}, left {v['blank_share_left']:.0%} vs stayed "
                                          f"{v['blank_share_stayed']:.0%}, gap {v['return_gap_pp']:+.2f} pp t {v['t']:+.1f}"
                                          for f, v in b.items()) + ")" if b else ""))
    return f"{la['VERDICT']} — " + "; ".join(parts) + "." if parts else f"{la['VERDICT']}."


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="data/sharadar_world2000_nominal_dl")
    ap.add_argument("--preds", nargs="*", default=[])
    ap.add_argument("--coverage", action="store_true", help="the raw tables' coverage by survival (coverage_by_survival)")
    a = ap.parse_args()
    if a.coverage:
        print(json.dumps(coverage_by_survival(a.store), indent=1))
    if a.preds:
        print(json.dumps(audit_segments(a.store, a.preds), indent=1))
