"""Feature probe, round 2: both tails of the 2006-2024 OOS record.

Round 1 (decline_features_probe_v1) tested nine decline/distress ratios; none
cleared the bar, and the two sent to the model exam (leverage_exam_v1) added
nothing. This round starts from the record instead of from ratios:

* the model's holdings are U-shaped in calendar-year return: it holds the
  year's worst-decile names 2.1 weeks/yr (3.4x the 0.62 base rate) and the
  best decile 1.0 weeks/yr -- it buys dips, and half its dips are secular
  declines (LUMN, BIGGQ, ENDPQ, GTW, GNRC, SEDG, FSLR, PCG);
* the weekly top-10 4-week gainers sit at a median model rank of 254/490;
  the recurring misses are cyclicals and product-cycle names (MU, AMD, NVDA,
  NFLX, MRO, NEM, FCX, BIIB, AMAT) and the March-April 2009 junk rally.

The candidates below carry information the panel does not: SEC 8-K item codes
(restructuring, impairments, auditor changes, officer departures, 8.01 floods),
the market's reaction on past *earnings* days (f_pead uses 10-Q/10-K filing
dates), the expected earnings window, dividend cuts and dividend cover,
accruals, book-value erosion, equity and revenue trends, margin trough and
margin peak against the firm's own history, cash runway, a partial Altman Z,
insider sale dollars over 26 weeks, one-day jumps (lottery), nominal price, own
seasonality, point-in-time peer groups (the 20 most return-correlated members
over the trailing year: the model has no sector or peer feature at all --
sector maps are current-snapshot and excluded), and two interactions the
bivariate probe cannot see from the parts (leverage x peer momentum, beta x
market 4-week return). Everything is computed
on the champion's panel base (S&P members each Friday), SF1 facts usable at
filing date + 1 day, 8-Ks usable the day after SEC acceptance, prices as of the
Friday close.

The never-ablated PENDING_ABLATION_FEATURES already in the panel ride along as
a separate block: they are free to probe and were never tested.

Pre-registered tails_features_probe_v1 (2026-09-04), rule fixed before the
numbers -- dense features (defined and non-zero on >= 10% of member-weeks):
keep iff universe |NW t| >= 2 on 2002-01 -> 2024-06-14 AND the same IC sign in
2002-2012 and 2013-2024. Sparse flags (< 10%): keep iff the weekly mean
label_4w of flagged names (weeks with >= 3 flagged) has |NW t| >= 2 AND the
same sign in both eras. With ~20 candidates at |t| >= 2 about one false keeper
is expected; the sampled paired model exam that follows (owner's go) is the
second gate. Nothing here changes the champion.

  PYTHONPATH=src:. .venv/bin/python ops/tails_features_probe.py --register
  PYTHONPATH=src:. .venv/bin/python ops/tails_features_probe.py            # ~5 min; reports/tails_features_probe.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ops.decline_features_probe import HOLD, HOLDOUT, MIN_NAMES, MIN_PICKS, NW_LAG, SPLIT, START, T_BAR, WORLD, \
    nw_t, summarize, weekly_ic
from stocks_ml.data.store import DataStore
from stocks_ml.features.panel import PENDING_ABLATION_FEATURES
from stocks_ml.features.sharadar_fundamentals import _asof
from stocks_ml.models.trials import record_trials

NAME = "tails_features_probe_v1"
REPORT = Path("reports/tails_features_probe.md")
SPARSE = 0.10          # defined-and-non-zero share below this -> flag rule
MIN_FLAGGED = 3        # flagged names a week needs to count
N_PEERS = 20
CORR_DAYS, MIN_CORR_DAYS = 250, 200
DISTRESS_ITEMS = ("2.04", "2.05", "2.06", "3.01", "4.01", "1.02")

CANDIDATES = {
    # --- risk: the dip that is a decline
    "c_8k_distress_26w": "count of 8-Ks in the last 26 weeks with items 2.04 triggering event / 2.05 exit or disposal costs / 2.06 impairment / 3.01 delisting notice / 4.01 auditor change / 1.02 termination of a material agreement",
    "c_8k_officer_26w": "count of 8-Ks in the last 26 weeks with item 5.02 (officer or director departure / appointment)",
    "c_8k_other_26w": "count of 8-Ks in the last 26 weeks with item 8.01 'other events' (the PG&E flood of fire filings)",
    "c_ins_sell_26w": "Form-4 open-market sales (code S) over the last 26 weeks, dollars / market cap (Mozilo, Ahearn, the homebuilders 2005-06)",
    "c_accrual": "(net income - operating cash flow) / assets, trailing 12m: profits the cash never confirmed (GNRC, SEDG, BIG, the homebuilders)",
    "c_bvps_chg_2q": "book value per share vs two quarters earlier: impairments and OCI losses hit the book before the P&L",
    "c_div_cover": "trailing FCF / dividends paid (payers only): the dividend the cash flow cannot carry (LUMN, BIG, FCX, PCG)",
    "c_dps_yoy": "trailing 12m dividends per share vs a year earlier (payers a year ago): a cut is the board conceding",
    "c_equity_yoy": "book equity vs a year earlier (ARQ): write-downs and losses eating the equity",
    "c_rev_streak": "consecutive quarters of year-over-year revenue decline (0-8)",
    "c_cash_runway": "cash / trailing cash burn (years) when FCF < 0, +inf otherwise",
    "c_partial_z": "Altman Z without the working-capital and retained-earnings terms: 3.3 EBIT/assets + 0.6 mcap/liabilities + revenue/assets",
    "c_jump_dn_4w": "worst single-day return in the last 20 trading days (a news crash vs a drift)",
    "c_jump_up_4w": "best single-day return in the last 20 trading days (lottery / MAX effect)",
    "c_price_level": "log nominal share price: single-digit prices in the S&P 500 are distress",
    # --- growth: the surge the model ranked at the bottom
    "c_rev_accel": "revenue YoY growth minus the prior quarter's YoY growth: the cycle turning",
    "c_rev_accel_4q": "revenue YoY growth minus the YoY growth four quarters earlier: the slower deceleration (PYPL, GNRC, SEDG)",
    "c_margin_z_5y": "EBITDA margin (trailing 12m) standardized against the firm's own trailing 5 years: the cyclical peak (MOS 2008, homebuilders 2006)",
    "c_gm_chg": "gross margin (trailing 12m) minus a year earlier: pricing power arriving or leaving",
    "c_earn_react_last": "market-adjusted 2-day return around the last earnings 8-K (item 2.02): the true post-earnings drift",
    "c_earn_react_mean4": "mean of that reaction over the last four earnings days: earnings-announcement-return momentum",
    "c_earn_window": "1 if the next earnings 8-K is expected inside the 4-week label window (last 2.02 date + the ticker's median gap)",
    "c_seasonal_3y": "the stock's own return over the same 4-week calendar window in each of the prior 3 years, averaged",
    # --- peers: point-in-time peer groups (20 most correlated members, trailing 250 days)
    "c_peer_mom_4w": "mean 4-week return of the peer group: industry momentum without a sector map",
    "c_peer_mom_12w": "mean 12-week return of the peer group",
    "c_own_vs_peer_4w": "own 4-week return minus the peer group's: the within-industry reversal",
    "c_peer_rev_gap": "own revenue YoY minus the peer group's median: shrinking while peers grow",
    # --- interactions the trees cannot see from the parts (bivariate probe needs the product)
    "c_lev_x_peer_mom12": "debt / market cap times the peer group's 12-week return: the levered cyclical when its industry turns (THC 2009, FCX 2016, APA 2020)",
    "c_beta_x_mkt4w": "250-day beta to SPY times SPY's 4-week return: high beta after the market turns (the March-2009 junk rally)",
    "c_margin_trough": "EBITDA margin (trailing 12m) minus its own 12-quarter high: depth of the margin depression (MU, AMD, oil 2020)",
}
PENDING = sorted(PENDING_ABLATION_FEATURES)
REFERENCE = ["f_short_dtc", "f_mom_4w", "f_log_mktcap", "f_pead", "f_sf_fcf_yield", "f_evt_earnings_8k_7d",
             "f_days_since_earnings_8k"]
SPEC = (f"{len(CANDIDATES)} candidates ({', '.join(CANDIDATES)}) plus the {len(PENDING)} never-ablated "
        f"PENDING_ABLATION_FEATURES as a separate block; universe = champion panel base, {START.date()} -> "
        f"labels ending before {HOLDOUT.date()}; IC = mean weekly Spearman vs label_4w, NW t lag {NW_LAG}; "
        f"dense rule (defined & non-zero >= {SPARSE:.0%}): keep iff |t| >= {T_BAR} AND same IC sign in "
        f"{START.year}-{SPLIT.year - 1} and {SPLIT.year}-2024; sparse rule: weekly mean label_4w of flagged "
        f"names (>= {MIN_FLAGGED} flagged), keep iff |NW t| >= {T_BAR} AND same sign both eras. Keepers -> "
        f"sampled paired model exam (owner's go); nothing changes the champion.")


def register():
    record_trials([{"kind": "preregistration", "name": NAME, "notes": SPEC}])
    print("registered", NAME)


# ----------------------------------------------------------------------------- helpers
def _lookup(wide: pd.DataFrame, base: pd.DataFrame) -> np.ndarray:
    """wide (dates x tickers) values at base's (date, ticker) rows."""
    s = wide.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    return s.reindex(pd.MultiIndex.from_frame(base[["date", "ticker"]])).to_numpy()


def _asof_wide(wide: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return wide.reindex(wide.index.union(dates)).ffill().reindex(dates)


def _ret_between(close: pd.DataFrame, base: pd.DataFrame, d0: pd.Series, d1: pd.Series) -> np.ndarray:
    """close.asof(d1) / close.asof(d0) - 1 per base row (d0, d1 are per-row dates)."""
    cal = close.index.values
    i0 = np.searchsorted(cal, d0.values, side="right") - 1
    i1 = np.searchsorted(cal, d1.values, side="right") - 1
    col = base["ticker"].map({t: i for i, t in enumerate(close.columns)})
    ok = col.notna().to_numpy() & (i0 >= 0) & (i1 >= 0)
    out = np.full(len(base), np.nan)
    c = col.fillna(0).astype(int).to_numpy()
    arr = close.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out[ok] = arr[i1[ok], c[ok]] / arr[i0[ok], c[ok]] - 1
    return out


# ----------------------------------------------------------------------------- price features
def price_features(close: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(sorted(base["date"].unique()))
    r = close.pct_change(fill_method=None)
    out = pd.DataFrame(index=base.index)
    out["c_jump_dn_4w"] = _lookup(_asof_wide(r.rolling(20, min_periods=15).min(), dates), base)
    out["c_jump_up_4w"] = _lookup(_asof_wide(r.rolling(20, min_periods=15).max(), dates), base)
    px = _lookup(_asof_wide(close, dates), base)
    out["c_price_level"] = np.log(np.where(px > 0, px, np.nan))
    seas = []
    for k in (1, 2, 3):
        d0 = base["date"] - pd.Timedelta(days=365 * k)
        seas.append(_ret_between(close, base, d0, d0 + pd.Timedelta(days=28)))
    seas = np.vstack(seas)
    out["c_seasonal_3y"] = np.where(np.isfinite(seas).sum(0) >= 2, np.nanmean(seas, 0), np.nan)
    return out


# ----------------------------------------------------------------------------- SF1 features
def sf1_features(world: DataStore, base: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    fund = world.read("fundamentals")
    arq = fund[fund.dimension == "ARQ"].sort_values(["ticker", "reportperiod"]).copy()
    g = arq.groupby("ticker")
    gap4 = g["reportperiod"].diff(4).dt.days
    ok4 = (gap4 > 330) & (gap4 < 400)
    prev_rev, prev_eq = g["revenue"].shift(4), g["equity"].shift(4)
    arq["revenue_yoy"] = (arq["revenue"] / prev_rev - 1).where(ok4 & (prev_rev > 0))
    arq["equity_yoy"] = (arq["equity"] / prev_eq - 1).where(ok4 & (prev_eq > 0))
    gap1 = g["reportperiod"].diff(1).dt.days
    ok1 = (gap1 > 60) & (gap1 < 120)
    arq["rev_accel"] = (arq["revenue_yoy"] - g["revenue_yoy"].shift(1)).where(ok1)
    arq["rev_accel_4q"] = (arq["revenue_yoy"] - g["revenue_yoy"].shift(4)).where(ok4)
    gap2 = g["reportperiod"].diff(2).dt.days
    prev_bvps = g["bvps"].shift(2)
    arq["bvps_chg_2q"] = (arq["bvps"] / prev_bvps - 1).where((gap2 > 150) & (gap2 < 220) & (prev_bvps > 0))
    dec = (arq["revenue_yoy"] < 0).astype(int)
    run_id = (dec == 0).groupby(arq["ticker"]).cumsum()
    arq["rev_streak"] = dec.groupby([arq["ticker"], run_id]).cumsum().clip(upper=8).astype(float)
    arq.loc[arq["revenue_yoy"].isna(), "rev_streak"] = np.nan

    art = fund[fund.dimension == "ART"].sort_values(["ticker", "reportperiod"]).copy()
    ga = art.groupby("ticker")
    gapa = ga["reportperiod"].diff(4).dt.days
    oka = (gapa > 330) & (gapa < 400)
    prev_dps = ga["dps"].shift(4)
    art["dps_yoy"] = (art["dps"] / prev_dps - 1).where(oka & (prev_dps > 0))
    art["gm_chg"] = (art["grossmargin"] - ga["grossmargin"].shift(4)).where(oka)
    art["margin_trough"] = art["ebitdamargin"] - ga["ebitdamargin"].transform(
        lambda x: x.rolling(12, min_periods=8).max())
    mu = ga["ebitdamargin"].transform(lambda x: x.rolling(20, min_periods=12).mean())
    sd = ga["ebitdamargin"].transform(lambda x: x.rolling(20, min_periods=12).std())
    art["margin_z_5y"] = ((art["ebitdamargin"] - mu) / sd.where(sd > 0)).clip(-5, 5)
    art["accrual"] = art["netinc"] - art["ncfo"]

    a = _asof(base, arq, ["revenue_yoy", "equity_yoy", "rev_accel", "rev_accel_4q", "rev_streak", "bvps_chg_2q",
                          "cashneq", "assets", "liabilities", "sharesbas", "debt"])
    t = _asof(base, art, ["dps_yoy", "gm_chg", "margin_trough", "margin_z_5y", "accrual", "fcf", "ebit",
                          "revenue", "dps"])
    out = pd.DataFrame(index=base.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["c_dps_yoy"] = t["dps_yoy"]
        out["c_equity_yoy"] = a["equity_yoy"]
        out["c_rev_streak"] = a["rev_streak"]
        out["c_rev_accel"] = a["rev_accel"]
        out["c_rev_accel_4q"] = a["rev_accel_4q"]
        out["c_bvps_chg_2q"] = a["bvps_chg_2q"]
        out["c_margin_z_5y"] = t["margin_z_5y"]
        out["c_accrual"] = t["accrual"] / a["assets"].where(a["assets"] > 0)
        divs = (t["dps"] * a["sharesbas"]).where(t["dps"] > 0)
        out["c_div_cover"] = t["fcf"] / divs
        out["c_gm_chg"] = t["gm_chg"]
        fcf, cash = t["fcf"].to_numpy(), a["cashneq"].to_numpy()
        out["c_cash_runway"] = np.where(np.isnan(fcf) | np.isnan(cash), np.nan,
                                        np.where(fcf < 0, cash / -fcf, np.inf))
        mcap = px * a["sharesbas"]
        out["c_partial_z"] = (3.3 * t["ebit"] / a["assets"] + 0.6 * mcap / a["liabilities"]
                              + t["revenue"] / a["assets"])
        out["c_margin_trough"] = t["margin_trough"]
        out["lev"] = a["debt"] / mcap                 # consumed by the peer block, not a candidate
    out["mcap"] = mcap                                # consumed by the Form-4 block, not a candidate
    out["revenue_yoy"] = a["revenue_yoy"]          # consumed by the peer block, not a candidate
    return out


# ----------------------------------------------------------------------------- 8-K features
def sec8k_features(sec8k: pd.DataFrame, close: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    f = sec8k.copy()
    accepted = pd.to_datetime(f["accepted"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    f["anchor"] = accepted.fillna(pd.to_datetime(f["filed"]).dt.normalize())
    f["available"] = f["anchor"] + pd.Timedelta(days=1)
    items = f["items"].fillna("").astype(str)

    def has(code):
        return items.str.contains(rf"(?:^|,\s*){code.replace('.', r'\.')}(?:,|$)", regex=True)

    f["distress"] = np.logical_or.reduce([has(c) for c in DISTRESS_ITEMS])
    f["officer"] = has("5.02")
    f["other"] = has("8.01")
    f["earn"] = has("2.02") & ~f["is_amendment"].fillna(False).astype(bool)
    f = f.dropna(subset=["available"]).sort_values("available")

    # earnings-day reaction: close[d+1] / close[d-1] - 1 minus SPY, d = trading day on/before acceptance
    cal = close.index.values
    e = f[f["earn"]].drop_duplicates(["ticker", "anchor"]).copy()
    d = np.searchsorted(cal, e["anchor"].values, side="right") - 1
    col = e["ticker"].map({t: i for i, t in enumerate(close.columns)})
    spy = close.columns.get_loc("SPY") if "SPY" in close.columns else None
    ok = col.notna().to_numpy() & (d >= 1) & (d + 1 < len(cal))
    arr = close.to_numpy()
    react = np.full(len(e), np.nan)
    c = col.fillna(0).astype(int).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        own = arr[d[ok] + 1, c[ok]] / arr[d[ok] - 1, c[ok]] - 1
        mkt = arr[d[ok] + 1, spy] / arr[d[ok] - 1, spy] - 1 if spy is not None else 0.0
        react[ok] = own - mkt
    e["react"] = react
    e["react_avail"] = pd.DatetimeIndex(cal[np.clip(d + 1, 0, len(cal) - 1)])  # the day close[d+1] is known
    e = e.sort_values("react_avail")

    out = pd.DataFrame(index=base.index)
    for name, mask in (("c_8k_distress_26w", f["distress"]), ("c_8k_officer_26w", f["officer"]),
                       ("c_8k_other_26w", f["other"])):
        ev = f.loc[mask, ["ticker", "available"]]
        cnt = np.zeros(len(base))
        for tk, grp in ev.groupby("ticker"):
            rows = np.flatnonzero((base["ticker"] == tk).to_numpy())
            if not len(rows):
                continue
            av = grp["available"].values
            t = base["date"].values[rows]
            cnt[rows] = (np.searchsorted(av, t, side="right")
                         - np.searchsorted(av, t - np.timedelta64(182, "D"), side="right"))
        out[name] = cnt

    last = np.full(len(base), np.nan)
    mean4 = np.full(len(base), np.nan)
    window = np.zeros(len(base))
    bt = base[["date", "ticker"]].reset_index(drop=True)
    for tk, grp in e.groupby("ticker"):
        rows = np.flatnonzero((bt["ticker"] == tk).to_numpy())
        if not len(rows):
            continue
        t = bt["date"].values[rows]
        ra, rv = grp["react_avail"].values, grp["react"].to_numpy()
        k = np.searchsorted(ra, t, side="right")           # reactions known by t
        has_k = k >= 1
        last[rows[has_k]] = rv[k[has_k] - 1]
        csum = np.concatenate([[0.0], np.nancumsum(rv)])
        ccnt = np.concatenate([[0], np.cumsum(np.isfinite(rv))])
        lo = np.maximum(k - 4, 0)
        n = ccnt[k] - ccnt[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = (csum[k] - csum[lo]) / n
        mean4[rows[n >= 2]] = m[n >= 2]
        # expected next earnings 8-K: last anchor + the ticker's median gap (60-120 d), default 91
        an = np.sort(grp["anchor"].values)
        gaps = np.diff(an).astype("timedelta64[D]").astype(float)
        gaps = gaps[(gaps > 60) & (gaps < 120)]
        med = np.median(gaps) if len(gaps) >= 2 else 91.0
        ka = np.searchsorted(an, t, side="right")
        has_a = ka >= 1
        expected = an[np.maximum(ka - 1, 0)] + np.timedelta64(int(round(med)), "D")
        delta = (expected - t).astype("timedelta64[D]").astype(float)
        window[rows[has_a & (delta > -14) & (delta <= 28)]] = 1.0
    out["c_earn_react_last"] = last
    out["c_earn_react_mean4"] = mean4
    out["c_earn_window"] = window
    return out


# ----------------------------------------------------------------------------- Form 4
def insider_sales(form4: pd.DataFrame, base: pd.DataFrame, mcap: pd.Series) -> pd.Series:
    """Open-market sale dollars (code S) filed in the last 182 days / market cap. Filed + 1 day, as in the panel."""
    f4 = form4[form4["code"] == "S"].copy()
    f4["available"] = pd.to_datetime(f4["filed"]).dt.normalize() + pd.Timedelta(days=1)
    f4 = f4.dropna(subset=["available", "value"]).sort_values("available")
    sold = np.zeros(len(base))
    for tk, grp in f4.groupby("ticker"):
        rows = np.flatnonzero((base["ticker"] == tk).to_numpy())
        if not len(rows):
            continue
        av, cum = grp["available"].values, np.concatenate([[0.0], np.cumsum(grp["value"].to_numpy(dtype=float))])
        t = base["date"].values[rows]
        hi = np.searchsorted(av, t, side="right")
        lo = np.searchsorted(av, t - np.timedelta64(182, "D"), side="right")
        sold[rows] = cum[hi] - cum[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(sold / mcap.where(mcap > 0).to_numpy(), index=base.index)


# ----------------------------------------------------------------------------- peer features
def peer_features(close: pd.DataFrame, base: pd.DataFrame, revenue_yoy: pd.Series, lev: pd.Series) -> pd.DataFrame:
    r = close.pct_change(fill_method=None)
    mom4 = close / close.shift(20) - 1
    mom12 = close / close.shift(60) - 1
    idx = pd.MultiIndex.from_frame(base[["date", "ticker"]])
    ry = pd.Series(revenue_yoy.to_numpy(), index=idx)
    lv = pd.Series(lev.to_numpy(), index=idx)
    cal = close.index.values
    pos = {t: i for i, t in enumerate(close.columns)}
    spy = r["SPY"].to_numpy()
    spy4 = mom4["SPY"].to_numpy()
    out = {k: np.full(len(base), np.nan) for k in ("c_peer_mom_4w", "c_peer_mom_12w", "c_own_vs_peer_4w",
                                                   "c_peer_rev_gap", "c_lev_x_peer_mom12", "c_beta_x_mkt4w")}
    for t, rows in base.groupby("date").indices.items():
        i = np.searchsorted(cal, np.datetime64(t), side="right") - 1
        if i < CORR_DAYS:
            continue
        tks = base["ticker"].to_numpy()[rows]
        keep = np.array([tk in pos for tk in tks])
        if keep.sum() < N_PEERS + 5:
            continue
        rows, tks = rows[keep], tks[keep]
        cols = np.array([pos[tk] for tk in tks])
        w = r.to_numpy()[i - CORR_DAYS + 1:i + 1][:, cols]
        n_ok = np.isfinite(w).sum(0)
        good = n_ok >= MIN_CORR_DAYS
        if good.sum() < N_PEERS + 5:
            continue
        rows, tks, cols, w = rows[good], tks[good], cols[good], w[:, good]
        w = np.where(np.isfinite(w), w, np.nan)
        w = w - np.nanmean(w, 0)
        w = np.where(np.isfinite(w), w, 0.0)
        sd = np.sqrt((w ** 2).sum(0))
        sd[sd == 0] = np.nan
        c = (w.T @ w) / np.outer(sd, sd)
        np.fill_diagonal(c, -np.inf)
        c = np.where(np.isfinite(c), c, -np.inf)
        peers = np.argpartition(-c, N_PEERS, axis=1)[:, :N_PEERS]
        m4 = mom4.to_numpy()[i, cols]
        m12 = mom12.to_numpy()[i, cols]
        key = pd.MultiIndex.from_arrays([np.repeat(t, len(tks)), tks])
        rev = ry.reindex(key).to_numpy(dtype=float)
        levs = lv.reindex(key).to_numpy(dtype=float)
        m = spy[i - CORR_DAYS + 1:i + 1]
        mk = np.isfinite(m)
        mm = m[mk] - m[mk].mean()
        beta = (w[mk].T @ mm) / (mm @ mm)                     # w is demeaned, NaN -> 0
        with np.errstate(invalid="ignore"):
            p4 = np.nanmean(m4[peers], 1)
            p12 = np.nanmean(m12[peers], 1)
            out["c_peer_mom_4w"][rows] = p4
            out["c_peer_mom_12w"][rows] = p12
            out["c_own_vs_peer_4w"][rows] = m4 - p4
            out["c_peer_rev_gap"][rows] = rev - np.nanmedian(rev[peers], 1)
            out["c_lev_x_peer_mom12"][rows] = levs * p12
            out["c_beta_x_mkt4w"][rows] = beta * spy4[i]
    return pd.DataFrame(out, index=base.index)


def candidate_features(world: DataStore, base: pd.DataFrame) -> pd.DataFrame:
    """All candidates on base rows (date, ticker); base.index kept."""
    prices = world.read("prices").sort_values("date")
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    dates = pd.DatetimeIndex(sorted(base["date"].unique()))
    px = pd.Series(_lookup(_asof_wide(close, dates), base), index=base.index).replace(0, np.nan)
    t0 = time.time()
    parts = [price_features(close, base)]
    print(f"prices {time.time() - t0:.0f}s", flush=True)
    sf = sf1_features(world, base, px)
    parts.append(sf.drop(columns=["revenue_yoy", "lev", "mcap"]))
    print(f"sf1 {time.time() - t0:.0f}s", flush=True)
    parts.append(sec8k_features(world.read("sec8k"), close, base))
    print(f"8-K {time.time() - t0:.0f}s", flush=True)
    parts.append(insider_sales(world.read("form4"), base, sf["mcap"]).rename("c_ins_sell_26w").to_frame())
    print(f"form4 {time.time() - t0:.0f}s", flush=True)
    parts.append(peer_features(close, base, sf["revenue_yoy"], sf["lev"]))
    print(f"peers {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(parts, axis=1)[list(CANDIDATES)]
    return out.replace([np.inf, -np.inf], [1e300, -1e300])


# ----------------------------------------------------------------------------- probe
def flagged_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Weekly mean label_4w of names with col != 0 (weeks with >= MIN_FLAGGED)."""
    d = df.loc[df[col].fillna(0) != 0, ["date", "label_4w"]].dropna()
    g = d.groupby("date")["label_4w"]
    return g.mean().where(g.size() >= MIN_FLAGGED)


def probe(pan: pd.DataFrame, picks: pd.DataFrame, col: str, candidate: bool) -> dict:
    cov = pan[col].notna().mean()
    active = (pan[col].fillna(0) != 0).mean()
    sparse = candidate and active < SPARSE
    u = summarize(flagged_series(pan, col) if sparse else weekly_ic(pan, col, MIN_NAMES))
    p = summarize(weekly_ic(picks, col, MIN_PICKS))
    keep = (abs(u["t"]) >= T_BAR) and u["same_sign"]
    return {"feature": col, "candidate": candidate, "coverage": cov, "active": active, "sparse": sparse,
            **u, "picks_ic": p["ic"], "picks_t": p["t"], "keep": keep}


def run():
    world = DataStore(str(WORLD))
    cols = ["date", "ticker", "label_4w", "label_end_date_4w"] + PENDING + REFERENCE
    pan = pd.read_parquet(WORLD / "panel_sf.parquet", columns=cols)
    pan = pan[(pan["date"] >= START) & (pan["label_end_date_4w"] < HOLDOUT)].reset_index(drop=True)
    feats = candidate_features(world, pan[["date", "ticker"]])
    pan = pd.concat([pan, feats], axis=1)
    pan.to_parquet(Path("data/experiments") / f"{NAME}_features.parquet")

    hold = pd.read_parquet(HOLD)
    hold["week"] = pd.to_datetime(hold["week"])
    picks = hold[["week", "top15"]].assign(ticker=hold["top15"].str.split(",")).explode("ticker")
    picks = picks.rename(columns={"week": "date"}).merge(
        pan[["date", "ticker", "label_4w"] + list(CANDIDATES) + PENDING + REFERENCE], on=["date", "ticker"])

    res = pd.DataFrame([probe(pan, picks, c, True) for c in CANDIDATES]
                       + [probe(pan, picks, c, False) for c in PENDING + REFERENCE])
    res["block"] = ["candidate"] * len(CANDIDATES) + ["pending"] * len(PENDING) + ["reference"] * len(REFERENCE)

    def fmt(r):
        stat = "flag mean" if r.sparse else "IC"
        verdict = ("KEEP" if r.keep else "drop") if r.block != "reference" else "in the model"
        return (f"| {r.feature} | {r.coverage:.0%} / {r.active:.0%} | {stat} | {r.ic:+.4f} | {r.t:+.1f} | "
                f"{r.ic_a:+.4f} | {r.ic_b:+.4f} | {r.picks_ic:+.3f} | {r.picks_t:+.1f} | {verdict} |")

    head = ["| feature | defined / active | stat | value | NW t | 2002-12 | 2013-24 | picks IC | picks t | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    kept = res[(res.block == "candidate") & res.keep].feature.tolist()
    kept_pending = res[(res.block == "pending") & res.keep].feature.tolist()
    lines = ["# Tails feature probe: risk factors and growth the model misses", "",
             f"Pre-registered `{NAME}` (2026-09-04). {SPEC}", "",
             f"Universe: {len(pan):,} member-weeks over {pan['date'].nunique()} weeks, {pan['date'].min().date()} -> "
             f"{pan['date'].max().date()}. Picks: the champion's cached top-15 per week ({picks['date'].nunique()} "
             f"weeks). Dense features: value = mean weekly Spearman IC vs label_4w (0.01 real, 0.02 good). Sparse "
             f"flags: value = mean weekly label_4w (return minus the member median, 4 weeks) of flagged names. "
             f"NW t = Newey-West, lag {NW_LAG}.", "",
             "## Candidates", "", *head, *[fmt(r) for r in res[res.block == "candidate"].itertuples()], "",
             "## Pending-ablation panel features (never tested; same rule)", "", *head,
             *[fmt(r) for r in res[res.block == "pending"].itertuples()], "",
             "## Reference: features already in the model", "", *head,
             *[fmt(r) for r in res[res.block == "reference"].itertuples()], "",
             "## Verdict", "",
             (f"Keep (candidates): {', '.join(kept)}." if kept else "Keep (candidates): none.") + " " +
             (f"Keep (pending block): {', '.join(kept_pending)}." if kept_pending else "Keep (pending block): none.") +
             " Keepers go to the sampled paired model exam (champion params, same weeks, top-6 4-week return "
             "primary, t > 2) on the owner's go. Nothing here changes the champion.", "",
             "## Definitions", "", *[f"- `{k}`: {v}" for k, v in CANDIDATES.items()], ""]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))

    entries = [{"kind": "feature_probe", "name": f"{NAME}:{r.feature}",
                "config": {"probe": NAME, "block": r.block, "sparse": bool(r.sparse)},
                "ic": r.ic, "nw_t": r.t, "ic_2002_2012": r.ic_a, "ic_2013_2024": r.ic_b,
                "picks_ic": r.picks_ic, "picks_t": r.picks_t, "coverage": r.coverage, "active": r.active,
                "keep": bool(r.keep) if r.block != "reference" else None,
                "notes": f"{REPORT}"} for r in res.itertuples()]
    entries.append({"kind": "feature_probe", "name": NAME,
                    "notes": f"result: keep {kept or 'none'} of {len(CANDIDATES)} candidates and "
                             f"{kept_pending or 'none'} of {len(PENDING)} pending features by the registered "
                             f"rule; {REPORT}"})
    record_trials(entries)
    return res


if __name__ == "__main__":
    register() if "--register" in sys.argv else run()
