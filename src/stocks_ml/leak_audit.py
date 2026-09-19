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


def feature_scan(ctx, features, lo="2006-01-01", hi="2015-12-31") -> dict:
    """The split-factor check on the MODEL'S OWN features (owner's rule
    2026-09-19: standing, every audit — until then it was only ever run on
    proposed extras, and f_dollar_vol sat at -0.42 unnoticed). Report-only:
    the identity gate decides; features beyond FEATURE_FACTOR_LIMIT are
    listed for a mechanism check (value/quality names split less, so a
    price-free ratio can sit at 0.15-0.21 for economic reasons)."""
    res = feature_factor_check(ctx, list(features), lo, hi)
    beyond = {f: res[f]["corr_with_future_split_factor"] for f in features
              if f in res and abs(res[f]["corr_with_future_split_factor"]) > FEATURE_FACTOR_LIMIT}
    return {"limit": FEATURE_FACTOR_LIMIT, "n_features": len(features),
            "beyond_limit": dict(sorted(beyond.items(), key=lambda kv: -abs(kv[1]))),
            "all": {f: res[f]["corr_with_future_split_factor"] for f in features if f in res}}


def audit_segments(store: str, preds_paths, ctx=None) -> dict:
    """One audit PER segment; the verdict is the identity check on every
    segment. The factor numbers are reported per segment, never pooled:
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
    scan = feature_scan(ctx, model_features(preds_paths[0], ctx))
    finite = [s["ic_retention"] for s in segs.values() if np.isfinite(s["ic_retention"])]
    return {"segments": segs, "worst_retention": round(float(min(finite)), 3) if finite else None,
            "feature_scan": scan,
            "VERDICT": "PASS" if all(s["VERDICT"] == "PASS" for s in segs.values()) else "FAIL"}


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
    return f"{la['VERDICT']} — " + "; ".join(parts) + "." if parts else f"{la['VERDICT']}."


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="data/sharadar_world2000_nominal_dl")
    ap.add_argument("--preds", nargs="+", required=True)
    a = ap.parse_args()
    print(json.dumps(audit_segments(a.store, a.preds), indent=1))
