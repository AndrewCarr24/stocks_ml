from __future__ import annotations

import numpy as np
import pandas as pd

DURATION_CONCEPTS = ["revenues", "net_income", "gross_profit", "ocf"]
INSTANT_CONCEPTS = ["assets", "equity", "liabilities", "shares"]


def _annual_rows(edgar: pd.DataFrame, concept: str) -> pd.DataFrame:
    f = edgar[(edgar.concept == concept) & (edgar.form == "10-K")].copy()
    if concept in DURATION_CONCEPTS:
        days = (f["end"] - f["start"]).dt.days
        f = f[(days > 300) & (days < 400)]
    return f


def _asof_join(base: pd.DataFrame, facts: pd.DataFrame, colname: str) -> pd.Series:
    """Latest fact safely available before each base date, per ticker."""
    if facts.empty:
        return pd.Series(np.nan, index=base.index)
    facts = facts.copy()
    # Company Facts exposes a date but no acceptance time. Delay date-only
    # filings by one day so after-close submissions cannot enter same-close signals.
    facts["filed"] = pd.to_datetime(facts["filed"]).dt.normalize() + pd.Timedelta(days=1)
    facts = (facts.sort_values(["filed", "end"])
                  .drop_duplicates(subset=["ticker", "filed"], keep="last"))
    left = base[["date", "ticker"]].reset_index().sort_values("date")
    merged = pd.merge_asof(left, facts[["filed", "ticker", "val"]].sort_values("filed"),
                           left_on="date", right_on="filed", by="ticker")
    return merged.set_index("index")["val"].reindex(base.index).rename(colname)


def fundamental_features(edgar: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    vals = {}

    for concept in DURATION_CONCEPTS:
        vals[concept] = _asof_join(out, _annual_rows(edgar, concept), concept)

    for concept in INSTANT_CONCEPTS:
        facts = edgar[edgar.concept == concept]
        vals[concept] = _asof_join(out, facts, concept)

    # asset growth: consecutive annual (10-K) assets, 200-550 days apart by period end
    annual_assets = _annual_rows(edgar, "assets").sort_values(["ticker", "end"]).copy()
    annual_assets["prev_val"] = annual_assets.groupby("ticker")["val"].shift(1)
    annual_assets["prev_filed"] = annual_assets.groupby("ticker")["filed"].shift(1)
    gap = annual_assets.groupby("ticker")["end"].diff().dt.days
    annual_assets["growth"] = np.where((gap > 200) & (gap < 550),
                                       annual_assets["val"] / annual_assets["prev_val"] - 1.0,
                                       np.nan)
    # Growth is not knowable until both period values have been filed.
    annual_assets["filed"] = annual_assets[["filed", "prev_filed"]].max(axis=1)
    growth_facts = annual_assets.rename(columns={"growth": "gval"})[
        ["ticker", "filed", "end", "gval"]].rename(columns={"gval": "val"}).dropna(subset=["val"])
    vals["asset_growth"] = _asof_join(out, growth_facts, "asset_growth")

    mktcap = vals["shares"] * out["close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["f_earnings_yield"] = vals["net_income"] / mktcap
        out["f_book_to_market"] = vals["equity"] / mktcap
        out["f_sales_to_price"] = vals["revenues"] / mktcap
        out["f_cf_to_price"] = vals["ocf"] / mktcap
        out["f_roe"] = vals["net_income"] / vals["equity"]
        out["f_gross_profitability"] = vals["gross_profit"] / vals["assets"]
        out["f_ocf_to_assets"] = vals["ocf"] / vals["assets"]
        out["f_asset_growth"] = vals["asset_growth"]
        out["f_leverage"] = vals["liabilities"] / vals["assets"]
        out["f_log_mktcap"] = np.log(mktcap.where(mktcap > 0))
    return out.replace([np.inf, -np.inf], np.nan)
