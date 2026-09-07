"""Candidate features: computed for every panel row, admitted by the screen.

The panel carries these as ``x_`` columns. ``feature_cols`` only admits ``f_``
columns, so a candidate is invisible to the model until a selection run's
feature screen (stocks_ml.feature_screen) admits it and it reaches the walk
through ``extra_features``. That keeps the champion's inputs fixed while the
candidates ride along in every panel — research and live alike (data/world.py
appends them the same way), so a run that admits some is emulated live.

Provenance (the asterisk): the thirty stories below were written after reading
both tails of the champion's 2006-2024 record — its secular-decline dips and
the cyclical surges it ranked at the bottom — including the 2016-2024 window the
nested test grades on (ops/tails_features_probe.py; AGENTS.md). Any nested
result that uses them is a separate "+ engineered features" line: the screen
bounds the statistics to the selection window, not the ideas. Candidates added
later must be written from the selection window's tails only.

Conventions match the panel's: SF1 facts usable at filing date + 1 day
(sharadar_fundamentals._asof), 8-Ks the day after SEC acceptance (filing date
when acceptance is missing), Form 4s at filing + 1 day, prices as of the Friday
close. Each candidate is ranked cross-sectionally per week to (-1, 1] and
neutral-filled, like every ranked feature (features/ranking.py).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from stocks_ml.features.ranking import rank_normalize
from stocks_ml.features.sharadar_fundamentals import _asof

PREFIX = "x_"
N_PEERS = 20
CORR_DAYS, MIN_CORR_DAYS = 250, 200
DISTRESS_ITEMS = ("2.04", "2.05", "2.06", "3.01", "4.01", "1.02")
BIG = 1e300                      # stands in for +-inf: rank_normalize needs finite values

CANDIDATES = {
    # --- risk: the dip that is a decline
    "x_8k_distress_26w": "count of 8-Ks in the last 26 weeks with items 2.04 triggering event / 2.05 exit or disposal costs / 2.06 impairment / 3.01 delisting notice / 4.01 auditor change / 1.02 termination of a material agreement",
    "x_8k_officer_26w": "count of 8-Ks in the last 26 weeks with item 5.02 (officer or director departure / appointment)",
    "x_8k_other_26w": "count of 8-Ks in the last 26 weeks with item 8.01 'other events'",
    "x_ins_sell_26w": "Form-4 open-market sales (code S) over the last 26 weeks, dollars / market cap",
    "x_accrual": "(net income - operating cash flow) / assets, trailing 12m: profits the cash never confirmed",
    "x_bvps_chg_2q": "book value per share vs two quarters earlier: impairments and OCI losses hit the book before the P&L",
    "x_div_cover": "trailing FCF / dividends paid (payers only): the dividend the cash flow cannot carry",
    "x_dps_yoy": "trailing 12m dividends per share vs a year earlier (payers a year ago): a cut is the board conceding",
    "x_equity_yoy": "book equity vs a year earlier (ARQ): write-downs and losses eating the equity",
    "x_rev_streak": "consecutive quarters of year-over-year revenue decline (0-8)",
    "x_cash_runway": "cash / trailing cash burn (years) when FCF < 0, +inf otherwise",
    "x_partial_z": "Altman Z without the working-capital and retained-earnings terms: 3.3 EBIT/assets + 0.6 mcap/liabilities + revenue/assets",
    "x_jump_dn_4w": "worst single-day return in the last 20 trading days (a news crash vs a drift)",
    "x_jump_up_4w": "best single-day return in the last 20 trading days (lottery / MAX effect)",
    "x_price_level": "log nominal share price: single-digit prices in the S&P 500 are distress",
    # --- growth: the surge the model ranked at the bottom
    "x_rev_accel": "revenue YoY growth minus the prior quarter's YoY growth: the cycle turning",
    "x_rev_accel_4q": "revenue YoY growth minus the YoY growth four quarters earlier: the slower deceleration",
    "x_margin_z_5y": "EBITDA margin (trailing 12m) standardized against the firm's own trailing 5 years: the cyclical peak",
    "x_gm_chg": "gross margin (trailing 12m) minus a year earlier: pricing power arriving or leaving",
    "x_earn_react_last": "market-adjusted 2-day return around the last earnings 8-K (item 2.02): the true post-earnings drift",
    "x_earn_react_mean4": "mean of that reaction over the last four earnings days: earnings-announcement-return momentum",
    "x_earn_window": "1 if the next earnings 8-K is expected inside the 4-week label window (last 2.02 date + the ticker's median gap)",
    "x_seasonal_3y": "the stock's own return over the same 4-week calendar window in each of the prior 3 years, averaged",
    # --- peers: point-in-time peer groups (20 most correlated members, trailing 250 days)
    "x_peer_mom_4w": "mean 4-week return of the peer group: industry momentum without a sector map",
    "x_peer_mom_12w": "mean 12-week return of the peer group",
    "x_own_vs_peer_4w": "own 4-week return minus the peer group's: the within-industry reversal",
    "x_peer_rev_gap": "own revenue YoY minus the peer group's median: shrinking while peers grow",
    # --- interactions the trees cannot see from the parts
    "x_lev_x_peer_mom12": "debt / market cap times the peer group's 12-week return: the levered cyclical when its industry turns",
    "x_beta_x_mkt4w": "250-day beta to SPY times SPY's 4-week return: high beta after the market turns",
    "x_margin_trough": "EBITDA margin (trailing 12m) minus its own 12-quarter high: depth of the margin depression",
}


def candidate_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(PREFIX)]


# ----------------------------------------------------------------------------- helpers
def _lookup(wide: pd.DataFrame, base: pd.DataFrame) -> np.ndarray:
    """wide (dates x tickers) values at base's (date, ticker) rows."""
    s = wide.stack(future_stack=True)
    s.index.names = ["date", "ticker"]
    return s.reindex(pd.MultiIndex.from_frame(base[["date", "ticker"]])).to_numpy()


def _asof_wide(wide: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return wide.reindex(wide.index.union(dates)).ffill().reindex(dates)


def _rows_by_ticker(base: pd.DataFrame) -> dict:
    """ticker -> positional rows of base (base.index order)."""
    return base.reset_index(drop=True).groupby("ticker").indices


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
    out["x_jump_dn_4w"] = _lookup(_asof_wide(r.rolling(20, min_periods=15).min(), dates), base)
    out["x_jump_up_4w"] = _lookup(_asof_wide(r.rolling(20, min_periods=15).max(), dates), base)
    px = _lookup(_asof_wide(close, dates), base)
    out["x_price_level"] = np.log(np.where(px > 0, px, np.nan))
    seas = np.vstack([_ret_between(close, base, d0, d0 + pd.Timedelta(days=28))
                      for d0 in (base["date"] - pd.Timedelta(days=365 * k) for k in (1, 2, 3))])
    n = np.isfinite(seas).sum(0)
    with np.errstate(invalid="ignore"):
        out["x_seasonal_3y"] = np.where(n >= 2, np.nansum(seas, 0) / np.maximum(n, 1), np.nan)
    return out


# ----------------------------------------------------------------------------- SF1 features
def sf1_features(fund: pd.DataFrame, base: pd.DataFrame, px: pd.Series) -> pd.DataFrame:
    """The SF1 candidates plus three columns the other blocks consume
    (mcap, revenue_yoy, lev), not candidates themselves."""
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
        out["x_dps_yoy"] = t["dps_yoy"]
        out["x_equity_yoy"] = a["equity_yoy"]
        out["x_rev_streak"] = a["rev_streak"]
        out["x_rev_accel"] = a["rev_accel"]
        out["x_rev_accel_4q"] = a["rev_accel_4q"]
        out["x_bvps_chg_2q"] = a["bvps_chg_2q"]
        out["x_margin_z_5y"] = t["margin_z_5y"]
        out["x_accrual"] = t["accrual"] / a["assets"].where(a["assets"] > 0)
        divs = (t["dps"] * a["sharesbas"]).where(t["dps"] > 0)
        out["x_div_cover"] = t["fcf"] / divs
        out["x_gm_chg"] = t["gm_chg"]
        fcf, cash = t["fcf"].to_numpy(), a["cashneq"].to_numpy()
        out["x_cash_runway"] = np.where(np.isnan(fcf) | np.isnan(cash), np.nan,
                                        np.where(fcf < 0, cash / -fcf, np.inf))
        mcap = px * a["sharesbas"]
        out["x_partial_z"] = (3.3 * t["ebit"] / a["assets"] + 0.6 * mcap / a["liabilities"]
                              + t["revenue"] / a["assets"])
        out["x_margin_trough"] = t["margin_trough"]
        out["lev"] = a["debt"] / mcap
    out["mcap"] = mcap
    out["revenue_yoy"] = a["revenue_yoy"]
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

    by_ticker = _rows_by_ticker(base)
    dates = base["date"].values
    out = pd.DataFrame(index=base.index)
    for name, mask in (("x_8k_distress_26w", f["distress"]), ("x_8k_officer_26w", f["officer"]),
                       ("x_8k_other_26w", f["other"])):
        cnt = np.zeros(len(base))
        for tk, grp in f.loc[mask, ["ticker", "available"]].groupby("ticker"):
            rows = by_ticker.get(tk)
            if rows is None:
                continue
            av, t = grp["available"].values, dates[rows]
            cnt[rows] = (np.searchsorted(av, t, side="right")
                         - np.searchsorted(av, t - np.timedelta64(182, "D"), side="right"))
        out[name] = cnt

    last = np.full(len(base), np.nan)
    mean4 = np.full(len(base), np.nan)
    window = np.zeros(len(base))
    for tk, grp in e.groupby("ticker"):
        rows = by_ticker.get(tk)
        if rows is None:
            continue
        t = dates[rows]
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
    out["x_earn_react_last"] = last
    out["x_earn_react_mean4"] = mean4
    out["x_earn_window"] = window
    return out


# ----------------------------------------------------------------------------- Form 4
def insider_sales(form4: pd.DataFrame, base: pd.DataFrame, mcap: pd.Series) -> pd.Series:
    """Open-market sale dollars (code S) filed in the last 182 days / market cap."""
    f4 = form4[form4["code"] == "S"].copy()
    f4["available"] = pd.to_datetime(f4["filed"]).dt.normalize() + pd.Timedelta(days=1)
    f4 = f4.dropna(subset=["available", "value"]).sort_values("available")
    by_ticker = _rows_by_ticker(base)
    dates = base["date"].values
    sold = np.zeros(len(base))
    for tk, grp in f4.groupby("ticker"):
        rows = by_ticker.get(tk)
        if rows is None:
            continue
        av, cum = grp["available"].values, np.concatenate([[0.0], np.cumsum(grp["value"].to_numpy(dtype=float))])
        t = dates[rows]
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
    out = {k: np.full(len(base), np.nan) for k in ("x_peer_mom_4w", "x_peer_mom_12w", "x_own_vs_peer_4w",
                                                   "x_peer_rev_gap", "x_lev_x_peer_mom12", "x_beta_x_mkt4w")}
    tickers = base["ticker"].to_numpy()
    for t, rows in base.reset_index(drop=True).groupby("date").indices.items():
        i = np.searchsorted(cal, np.datetime64(t), side="right") - 1
        if i < CORR_DAYS:
            continue
        tks = tickers[rows]
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
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)    # an all-NaN peer slice is NaN by design
            p4 = np.nanmean(m4[peers], 1)
            p12 = np.nanmean(m12[peers], 1)
            out["x_peer_mom_4w"][rows] = p4
            out["x_peer_mom_12w"][rows] = p12
            out["x_own_vs_peer_4w"][rows] = m4 - p4
            out["x_peer_rev_gap"][rows] = rev - np.nanmedian(rev[peers], 1)
            out["x_lev_x_peer_mom12"][rows] = levs * p12
            out["x_beta_x_mkt4w"][rows] = beta * spy4[i]
    return pd.DataFrame(out, index=base.index)


# ----------------------------------------------------------------------------- assembly
def candidate_features(world, base: pd.DataFrame, log=None) -> pd.DataFrame:
    """Raw candidates on base rows (date, ticker); base.index kept. `world` is
    the DataStore holding prices, fundamentals, sec8k and form4."""
    log = log or (lambda msg: None)
    prices = world.read("prices").sort_values("date")
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    dates = pd.DatetimeIndex(sorted(base["date"].unique()))
    px = pd.Series(_lookup(_asof_wide(close, dates), base), index=base.index).replace(0, np.nan)
    parts = [price_features(close, base)]
    log("candidates: prices")
    sf = sf1_features(world.read("fundamentals"), base, px)
    parts.append(sf.drop(columns=["revenue_yoy", "lev", "mcap"]))
    log("candidates: sf1")
    parts.append(sec8k_features(world.read("sec8k"), close, base))
    log("candidates: 8-K")
    parts.append(insider_sales(world.read("form4"), base, sf["mcap"]).rename("x_ins_sell_26w").to_frame())
    log("candidates: form4")
    parts.append(peer_features(close, base, sf["revenue_yoy"], sf["lev"]))
    log("candidates: peers")
    out = pd.concat(parts, axis=1)[list(CANDIDATES)]
    return out.replace([np.inf, -np.inf], [BIG, -BIG])


def add_candidates(panel: pd.DataFrame, world, log=None) -> pd.DataFrame:
    """The panel plus every candidate as a ranked, neutral-filled ``x_`` column
    (existing ``x_`` columns are rebuilt). Row order and the other columns are
    untouched."""
    base = panel[["date", "ticker"]]
    feats = candidate_features(world, base, log=log)
    out = panel.drop(columns=candidate_cols(panel))
    out = pd.concat([out, feats], axis=1)
    return rank_normalize(out, list(CANDIDATES))
