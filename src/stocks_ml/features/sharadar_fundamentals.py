"""Fundamentals + insider features from the Sharadar Direct stores.

Point-in-time discipline (mirrors features/fundamentals.py for EDGAR):
every fact becomes usable at its SEC filing date `date` plus one day (an
after-close filing must not enter same-close signals). AR* dimensions only —
guaranteed at ingest (data/world.py keeps ARQ and ART only).

Namespaces: f_sf_* (fundamentals), f_sfi_* (insiders) — deliberately distinct
from the free-EDGAR f_earnings_yield family so ablations can compare sources.
All columns land raw here; the panel builder rank-normalizes per week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SF_RAW_COLS = [
    "f_sf_earnings_yield", "f_sf_book_to_market", "f_sf_fcf_yield",
    "f_sf_sales_to_price", "f_sf_debt_ebitda", "f_sf_current_ratio",
    "f_sf_de", "f_sf_roe", "f_sf_gross_prof", "f_sf_ebitda_margin",
    "f_sf_netinc_yoy", "f_sf_revenue_yoy", "f_sf_issuance",
    "f_sf_neg_ebitda",
]
SFI_RAW_COLS = ["f_sfi_net_13w", "f_sfi_buyers_13w"]


def _asof(base, facts, cols):
    """Latest filed fact per (ticker) at each base date (filed + 1 day)."""
    f = facts.copy()
    f["filed"] = f["date"].dt.normalize() + pd.Timedelta(days=1)
    f = f.sort_values("filed").drop_duplicates(["ticker", "filed"], keep="last")
    left = base[["date", "ticker"]].reset_index().sort_values("date")
    merged = pd.merge_asof(left, f[["filed", "ticker"] + cols].sort_values("filed"),
                           left_on="date", right_on="filed", by="ticker",
                           tolerance=pd.Timedelta(days=400))
    return merged.set_index("index").reindex(base.index)[cols]


def sharadar_fundamental_features(fund: pd.DataFrame, base: pd.DataFrame,
                                  close: pd.Series) -> pd.DataFrame:
    """base: panel rows with date/ticker; close: aligned decision-date close."""
    arq = fund[fund.dimension == "ARQ"].copy()
    art = fund[fund.dimension == "ART"].copy()

    # YoY comparisons on ARQ (prior-year quarter filed long before -> knowable
    # at the current row's filing date)
    arq = arq.sort_values(["ticker", "reportperiod"])
    g = arq.groupby("ticker")
    gap = g["reportperiod"].diff(4).dt.days
    ok = (gap > 330) & (gap < 400)
    arq["netinc_yoy_sc"] = ((arq["netinc"] - g["netinc"].shift(4))
                            / arq["assets"].abs()).where(ok)
    arq["revenue_yoy"] = (arq["revenue"] / g["revenue"].shift(4) - 1).where(
        ok & (g["revenue"].shift(4) > 0))
    arq["issuance"] = (arq["sharesbas"] / g["sharesbas"].shift(4) - 1).where(
        ok & (g["sharesbas"].shift(4) > 0))

    a = _asof(base, arq, ["bvps", "currentratio", "de", "debt", "equity",
                          "assets", "sharesbas", "netinc_yoy_sc",
                          "revenue_yoy", "issuance"])
    t = _asof(base, art, ["epsusd", "revenue", "netinc", "ebitda", "fcf",
                          "gp", "ebitdamargin"])

    out = pd.DataFrame(index=base.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = close.replace(0, np.nan)
        mkt = px * a["sharesbas"]
        out["f_sf_earnings_yield"] = t["epsusd"] / px
        out["f_sf_book_to_market"] = a["bvps"] / px
        out["f_sf_fcf_yield"] = t["fcf"] / mkt
        out["f_sf_sales_to_price"] = t["revenue"] / mkt
        out["f_sf_debt_ebitda"] = (a["debt"] / t["ebitda"]).where(t["ebitda"] > 0)
        out["f_sf_current_ratio"] = a["currentratio"]
        out["f_sf_de"] = a["de"]
        out["f_sf_roe"] = (t["netinc"] / a["equity"]).where(a["equity"] > 0)
        out["f_sf_gross_prof"] = t["gp"] / a["assets"].abs()
        out["f_sf_ebitda_margin"] = t["ebitdamargin"]
        out["f_sf_netinc_yoy"] = a["netinc_yoy_sc"]
        out["f_sf_revenue_yoy"] = a["revenue_yoy"]
        out["f_sf_issuance"] = a["issuance"]
        # economically meaningful absence: negative-EBITDA distress flag
        # (indicator, not ranked; NaN where no ART row is available)
        out["f_sf_neg_ebitda"] = (t["ebitda"] < 0).astype(float).where(
            t["ebitda"].notna())
    return out


def sharadar_insider_features(ins: pd.DataFrame, base: pd.DataFrame,
                              mktcap: pd.Series | None = None) -> pd.DataFrame:
    """Trailing-13-week open-market insider flow, knowable at filing + 1 day."""
    ev = ins.copy()
    ev["filed"] = ev["date"].dt.normalize() + pd.Timedelta(days=1)
    ev["buy"] = (ev["signed_value"] > 0).astype(float)
    ev["sell"] = (ev["signed_value"] < 0).astype(float)
    ev = ev.sort_values("filed")
    cum = ev.groupby("ticker")[["signed_value", "buy", "sell"]].cumsum()
    ev = pd.concat([ev[["ticker", "filed"]], cum], axis=1)

    def trailing(base_dates_shift):
        left = base[["date", "ticker"]].reset_index()
        left["at"] = left["date"] - base_dates_shift
        left = left.sort_values("at")
        m = pd.merge_asof(left, ev.rename(columns={"filed": "at_ev"}),
                          left_on="at", right_on="at_ev", by="ticker")
        return m.set_index("index").reindex(base.index)[["signed_value", "buy", "sell"]]

    now = trailing(pd.Timedelta(days=0))
    ago = trailing(pd.Timedelta(days=91))
    net = now["signed_value"].fillna(0) - ago["signed_value"].fillna(0)
    buyers = (now["buy"].fillna(0) - ago["buy"].fillna(0)) - (
        now["sell"].fillna(0) - ago["sell"].fillna(0))
    out = pd.DataFrame(index=base.index)
    if mktcap is not None:
        net = net / mktcap.replace(0, np.nan)
    out["f_sfi_net_13w"] = net
    out["f_sfi_buyers_13w"] = buyers
    return out
