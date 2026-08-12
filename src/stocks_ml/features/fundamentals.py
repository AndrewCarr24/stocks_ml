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


def _quarterly_rows(edgar: pd.DataFrame, concept: str) -> pd.DataFrame:
    """True quarterly durations from 10-Q/10-K facts (~70-110 day periods).

    Q4 often appears only as the annual duration inside the 10-K, so seasonal
    chains can have gaps — a structural absence handled by neutral-fill, never
    by relaxing the join."""
    f = edgar[(edgar.concept == concept) & (edgar.form.isin(["10-Q", "10-K"]))].copy()
    days = (f["end"] - f["start"]).dt.days
    return f[(days > 70) & (days < 110)]


def _seasonal_pairs(rows: pd.DataFrame, tolerance_days: int = 45) -> pd.DataFrame:
    """Match each quarterly fact with the same fiscal quarter one year earlier.

    Knowable-when: max(filed, filed_prev) — a seasonal difference exists only
    once both quarters are on file (same convention as asset growth)."""
    rows = (rows.sort_values(["ticker", "end", "filed"])
                .drop_duplicates(subset=["ticker", "end"], keep="last"))
    left = rows.assign(target=rows["end"] - pd.Timedelta(days=365)).sort_values("target")
    right = rows[["ticker", "end", "val", "filed"]].sort_values("end")
    m = pd.merge_asof(left, right, left_on="target", right_on="end", by="ticker",
                      tolerance=pd.Timedelta(days=tolerance_days),
                      direction="nearest", suffixes=("", "_prev"))
    m = m.dropna(subset=["val_prev"])
    m["sdiff"] = m["val"] - m["val_prev"]
    m["filed_eff"] = m[["filed", "filed_prev"]].max(axis=1)
    return m.sort_values(["ticker", "end"])


def earnings_quality_features(edgar: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """SUE, consecutive-earnings-increases, and net share issuance.

    docs/research recommendation #2 (Bernard-Thomas 1989; Green-Hand-Zhang 2017;
    Hou-Xue-Zhang 2020 survivors). All filing-dated with the next-calendar-day
    availability shift applied by _asof_join; NaN where history is short."""
    out = base.copy()

    pairs = _seasonal_pairs(_quarterly_rows(edgar, "net_income"))
    if not pairs.empty:
        # SUE: seasonal surprise scaled by the std of the last 8 surprises
        # (min 4). The rolling window sees only rows already computed, and the
        # feature's filed date is the current pair's filed_eff.
        pairs["sigma"] = (pairs.groupby("ticker")["sdiff"]
                          .transform(lambda s: s.rolling(8, min_periods=4).std()))
        sue = pairs[pairs["sigma"] > 0].copy()
        sue["val"] = sue["sdiff"] / sue["sigma"]
        sue = (sue.drop(columns=["filed", "filed_prev", "val_prev"], errors="ignore")
                  .rename(columns={"filed_eff": "filed"}))
        out["f_sue"] = _asof_join(out, sue[["ticker", "filed", "end", "val"]], "sue")

        # nincr: length of the current run of positive seasonal surprises
        # (Green-Hand-Zhang's survivor), capped at 8 like the literature.
        def _runs(s):
            run, acc = 0, []
            for pos in (s > 0):
                run = run + 1 if pos else 0
                acc.append(min(run, 8))
            return pd.Series(acc, index=s.index, dtype=float)

        pairs["nincr"] = pairs.groupby("ticker")["sdiff"].transform(_runs)
        nincr = (pairs.drop(columns=["filed", "filed_prev", "val", "val_prev"],
                            errors="ignore")
                      .rename(columns={"filed_eff": "filed", "nincr": "val"}))
        out["f_nincr"] = _asof_join(out, nincr[["ticker", "filed", "end", "val"]],
                                    "nincr")
    else:
        out["f_sue"] = np.nan
        out["f_nincr"] = np.nan

    shares = edgar[edgar.concept == "shares"].copy()
    ipairs = _seasonal_pairs(shares.assign(start=shares["end"]), tolerance_days=60) \
        if not shares.empty else pd.DataFrame()
    if not ipairs.empty:
        iss = ipairs[ipairs["val_prev"] > 0].copy()
        iss["val"] = iss["val"] / iss["val_prev"] - 1.0
        iss = (iss.drop(columns=["filed", "filed_prev", "val_prev"], errors="ignore")
                  .rename(columns={"filed_eff": "filed"}))
        out["f_net_issuance"] = _asof_join(
            out, iss[["ticker", "filed", "end", "val"]], "net_issuance")
    else:
        out["f_net_issuance"] = np.nan
    return out
