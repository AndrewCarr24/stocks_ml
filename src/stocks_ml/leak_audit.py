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
    from stocks_ml.features.sharadar_fundamentals import _asof
    rows = ctx.pan[ctx.pan["date"] == week]
    if rows.empty or "f_sf_book_to_market" not in rows.columns:
        return {"week": str(week.date()), "ok": False, "reason": "no panel rows / column at this week"}
    base = rows[["date", "ticker"]].copy()
    bvps = _asof(base, fund[fund["dimension"] == "ARQ"], ["bvps"])["bvps"]
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


def audit_segments(store: str, preds_paths, ctx=None) -> dict:
    """One audit PER segment; the verdict is the identity check on every
    segment. The factor numbers are reported per segment, never pooled:
    the leaky 2026-09 champion showed retention 0.49 on its selection
    window yet ~1 with 2016-2024 pooled in."""
    import stocks_ml.selection as sel
    ctx = ctx or sel.Ctx(store)
    if not hasattr(ctx, "fund"):
        ctx.fund = pd.read_parquet(Path(store) / "fundamentals.parquet")
    segs = {Path(p).parent.name: audit(store, p, ctx) for p in preds_paths}
    finite = [s["ic_retention"] for s in segs.values() if np.isfinite(s["ic_retention"])]
    return {"segments": segs, "worst_retention": round(float(min(finite)), 3) if finite else None,
            "VERDICT": "PASS" if all(s["VERDICT"] == "PASS" for s in segs.values()) else "FAIL"}


def leak_line(la: dict) -> str:
    """One line: the verdict plus the report-only numbers per segment."""
    segs = la.get("segments", {})
    parts = [f"{name}: identity {s['IDENTITY']}, score-vs-split-factor {s['spearman_score_vs_factor']:+.3f} "
             f"(t {s['spearman_t']:+.1f}), IC {s['ic']:+.4f} (t {s['ic_t']:+.1f}), retention {s['ic_retention']}"
             for name, s in segs.items()]
    return f"{la['VERDICT']} — " + "; ".join(parts) + "." if parts else f"{la['VERDICT']}."


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="data/sharadar_world2000_nominal_dl")
    ap.add_argument("--preds", nargs="+", required=True)
    a = ap.parse_args()
    print(json.dumps(audit_segments(a.store, a.preds), indent=1))
