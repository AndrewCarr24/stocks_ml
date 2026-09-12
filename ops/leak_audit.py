"""Pre-adoption leak audit — REQUIRED before any champion adoption
(AGENTS.md standing rule, owner's mandate 2026-09-10).

A candidate's scores are tested against known future-information proxies on
pre-holdout weeks only:

1. FACTOR: the vendor adjustment factor log(closeadj/closeunadj) at the rank
   date — the restatement a look-ahead basis leak rides on (the 2026-09
   split leak). Reported: mean weekly Spearman(score, factor); the score's
   weekly IC; the IC after residualizing score ranks on factor ranks per
   week; the retention ratio. REPORT-ONLY since 2026-09-12 (owner's
   ruling): a model that favours strong, rising companies favours future
   splitters for legitimate reasons, so the correlation cannot separate a
   leak from a proxy; and the future split factor is partly future return,
   so residualizing it removes real skill too. Until then the retention
   gate (>= 75%; the leaky champion retained 0.49) was the verdict — it
   failed the clean-program package on 2006-2015 at retention -1.31, a
   ratio of two zeros (IC 0.0002, t 0.02).
2. DELISTING: names whose last print falls within 8 weeks after the rank
   date. Reported: their rate in the top-15 vs the member universe (the
   backtest's slice_row cannot rank them; a large gap flags survivorship
   pressure on the scores). Report-only.
3. IDENTITY: the store's prices table must carry closeunadj/close_split,
   and for the three names with the largest in-window future-split factors
   the nominal r_sf_bvps must equal stored x factor — the mechanism by
   which the leak was removed, checked where it matters most. GATE: exact;
   the only gate.

Usage:
  PYTHONPATH=src:. .venv/bin/python ops/leak_audit.py \
      --store data/sharadar_world2000_nominal \
      --preds data/experiments/nominal_clean_2006_2015/preds.parquet \
              data/experiments/nominal_clean_2016_2024/preds.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.selection import HOLDOUT_START

RETENTION_GATE = 0.75   # report-only since 2026-09-12; kept so the number stays comparable
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


def _rank(v: pd.Series) -> pd.Series:
    return v.rank()


def audit(store: str, preds_paths: list[str]) -> dict:
    import stocks_ml.selection as sel
    frames = []
    for p in preds_paths:
        df = pd.read_parquet(p)
        df["week"] = pd.to_datetime(df["week"])
        frames.append(df)
    preds = pd.concat(frames, ignore_index=True).sort_values(["week", "ticker"])
    if preds["week"].max() >= HOLDOUT_START:
        raise RuntimeError(f"preds reach {preds['week'].max().date()}: the audit reads "
                           f"pre-holdout weeks only (< {HOLDOUT_START.date()})")
    copies = [c for c in preds.columns if c.startswith("c")]
    preds["score"] = preds[copies].mean(axis=1)

    prices = pd.read_parquet(Path(store) / "prices.parquet")
    if not {"closeunadj", "close_split"} <= set(prices.columns):
        raise RuntimeError("prices table lacks closeunadj/close_split: regenerate it "
                           "(world.prices_from_sep) — IDENTITY check impossible")
    pr = prices.dropna(subset=["closeunadj"])
    fac = (np.log(pr["close"] / pr["closeunadj"])
           .groupby([pr["date"], pr["ticker"]]).last())
    last_print = prices.groupby("ticker")["date"].max()
    censor = prices["date"].max() - pd.Timedelta(days=DELIST_DAYS + 30)

    ctx = sel.Ctx(store)
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
        rs, rf = _rank(d["score"]), _rank(d["factor"])
        sp = float(rs.corr(rf))
        lab = d.dropna(subset=["ret"])
        ic = float(_rank(lab["score"]).corr(_rank(lab["ret"]))) if len(lab) > 50 else np.nan
        # residualize score ranks on factor ranks (per week, OLS on ranks)
        b = float(np.cov(rs, rf)[0, 1] / np.var(rf)) if float(np.var(rf)) > 0 else 0.0
        d["resid"] = rs - b * rf
        labr = d.dropna(subset=["ret"])
        ic_r = float(_rank(labr["resid"]).corr(_rank(labr["ret"]))) if len(labr) > 50 else np.nan
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
    res = {
        "weeks": int(len(t)),
        "span": f"{t.index.min().date()} -> {t.index.max().date()}",
        "spearman_score_vs_factor": round(float(t["spearman_factor"].mean()), 4),
        "spearman_t": round(nw_t(t["spearman_factor"]), 2),
        "ic": round(float(ic), 5), "ic_t": round(nw_t(t["ic"]), 2),
        "ic_resid_factor": round(float(ic_r), 5), "ic_resid_t": round(nw_t(t["ic_resid"]), 2),
        "ic_retention": round(retention, 3),
        "retention_reference": f">= {RETENTION_GATE} (report-only since 2026-09-12)",
        "FACTOR": "report-only",
        "delist_within_8w_universe": round(float(t["delist_uni"].mean()), 5),
        "delist_within_8w_top15": round(float(t["delist_top15"].mean()), 5),
    }

    # IDENTITY: largest in-window future-split names — nominal per-share = stored x factor
    from stocks_ml.features.generated import split_factor_input, sf1_inputs
    lo, hi = preds["week"].min(), preds["week"].max()
    win = pr[(pr["date"] >= lo) & (pr["date"] <= hi)]
    sf = (win["closeunadj"] / win["close_split"]).groupby(win["ticker"]).max()
    names = sf.nlargest(3).index.tolist()
    base = pd.DataFrame({"date": [lo + (hi - lo) / 2] * len(names), "ticker": names})
    factor = split_factor_input(prices, base)
    fund = pd.read_parquet(Path(store) / "fundamentals.parquet")
    nom = sf1_inputs(fund, base, split_factor=factor)
    raw = sf1_inputs(fund, base)
    ok = True
    ident = {}
    for i, tk in enumerate(names):
        a, b_, f_ = nom["r_sf_bvps"].iloc[i], raw["r_sf_bvps"].iloc[i], factor.iloc[i]
        good = (not np.isfinite(a)) or abs(a - b_ * f_) <= 1e-9 * max(1.0, abs(a))
        ident[tk] = {"stored": None if pd.isna(b_) else round(float(b_), 4),
                     "factor": round(float(f_), 3),
                     "nominal": None if pd.isna(a) else round(float(a), 4), "ok": bool(good)}
        ok &= good
    res["identity_top_splitters"] = ident
    res["IDENTITY"] = "PASS" if ok else "FAIL"
    res["VERDICT"] = res["IDENTITY"]
    return res


def audit_segments(store: str, preds_paths: list[str]) -> dict:
    """One audit PER preds file; the verdict is the identity check on every
    segment. The factor numbers are reported per segment, never pooled:
    the leaky 2026-09 champion showed retention 0.49 on its selection
    window yet ~1 with 2016-2024 pooled in, because in that era the
    factor-correlated component aligned with realized returns."""
    segs = {Path(p).parent.name: audit(store, [p]) for p in preds_paths}
    finite = [s["ic_retention"] for s in segs.values() if np.isfinite(s["ic_retention"])]
    out = {"segments": segs, "worst_retention": round(float(min(finite)), 3) if finite else None,
           "VERDICT": "PASS" if all(s["VERDICT"] == "PASS" for s in segs.values()) else "FAIL"}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    a = ap.parse_args()
    out = audit_segments(a.store, a.preds)
    print(json.dumps(out, indent=1))
