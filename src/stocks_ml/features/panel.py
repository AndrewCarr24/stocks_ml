from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_WEEK = 5
ANNUALIZER = np.sqrt(252)
_WEEK_ANCHOR = {0: "W-MON", 1: "W-TUE", 2: "W-WED", 3: "W-THU", 4: "W-FRI"}

# Generated and retained for research/coverage diagnostics, but excluded from
# production model matrices after the identical-calendar ablation documented in
# reports/feature_ablation.md.  SEC 8-K features were admitted; these were not.
REJECTED_MODEL_FEATURES = frozenset({
    "f_rev_resid_mkt_1w",
    "f_amihud_4w",
    "f_amihud_12w",
    "f_resid_ret_lag1w",
    "f_resid_ret_lag2w",
    "f_resid_ret_lag3w",
})

# Audited against quarterly ALFRED vintages on 2026-07-21. T10Y2Y had zero
# changes across 204,839 common comparisons (archive coverage starts 2014);
# FEDFUNDS had zero across 12,043 comparisons back to 2004. Other configured
# FRED series showed revisions and remain diagnostic-only. FEDFUNDS also needs
# the conservative 35-day observation-date lag configured in config.yaml.
POINT_IN_TIME_MACRO_SERIES = frozenset({"T10Y2Y", "FEDFUNDS"})

# Generated but NOT yet admitted to production matrices: candidate families
# awaiting their paired-ΔIC ablation (models/ablation.py, adopt at t >= 3).
# Remove from this set only with an ablation report showing t >= 3.
PENDING_ABLATION_FEATURES = frozenset({
    # momentum block, docs/research recommendation #1
    "f_mom_12w_skip1w", "f_mom_52w_skip4w", "f_mom_interm",
    # EDGAR earnings-quality bundle, recommendation #2
    "f_sue", "f_nincr", "f_net_issuance",
})


def _admitted_macro_feature(col: str) -> bool:
    return any(col == f"f_macro_{sid}" or col == f"f_macro_{sid}_chg"
               for sid in POINT_IN_TIME_MACRO_SERIES)


def trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.sort(prices["date"].unique()))


def rebalance_dates(calendar: pd.DatetimeIndex, start, end, weekday: int = 4) -> pd.DatetimeIndex:
    anchors = pd.date_range(start, end, freq=_WEEK_ANCHOR[weekday])
    idx = calendar.searchsorted(anchors, side="right") - 1
    idx = idx[idx >= 0]
    dates = pd.DatetimeIndex(pd.unique(calendar[idx]))
    return dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def all_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Return only features admitted to production model matrices."""
    return [c for c in all_feature_cols(df)
            if c not in REJECTED_MODEL_FEATURES
            and c not in PENDING_ABLATION_FEATURES
            # Only explicitly audited FRED series are admitted. Current
            # Wikipedia sectors are not effective-dated and remain excluded.
            and (not c.startswith("f_macro_") or _admitted_macro_feature(c))
            and not c.startswith("f_sec_")
            and not c.endswith("_sect")]


def _feature_family(col: str) -> str:
    if col.startswith("f_macro_"):
        return "macro"
    if col.startswith("f_mkt_"):
        return "market"
    if col.startswith("f_sec_") or col.endswith("_sect"):
        return "sector"
    if col.startswith("f_evt_") or col in {"f_days_since_filing", "f_pead",
                                             "f_days_since_earnings_8k"}:
        return "filing_events"
    if col.startswith("f_insider_"):
        return "insiders"
    if col.startswith("f_short_"):
        return "short_interest"
    if col in {"f_earnings_yield", "f_book_to_market", "f_sales_to_price",
               "f_cf_to_price", "f_roe", "f_gross_profitability", "f_ocf_to_assets",
               "f_asset_growth", "f_leverage", "f_log_mktcap"}:
        return "fundamentals"
    if col.startswith(("f_mom_", "f_rev_")) or col in {"f_hi_52w", "f_lo_52w",
                                             "f_overnight_4w", "f_intraday_4w"}:
        return "price_trend"
    if col.startswith("f_vol_") or col in {"f_downside_dev", "f_beta_60d",
                                             "f_idio_vol_60d"}:
        return "risk"
    if col in {"f_dollar_vol", "f_abn_volume", "f_amihud_4w", "f_amihud_12w"}:
        return "liquidity"
    if col in {"f_month", "f_woq"}:
        return "calendar"
    return "other"


def feature_coverage(panel: pd.DataFrame) -> dict:
    """Return weekly and cross-sectional coverage before neutral filling.

    Structural absence before a feature first appears is separated from complete
    outages after that date. This keeps a late source start from masking retrieval
    failures during the source's usable history.
    """
    fcols = all_feature_cols(panel)

    def _weekly_summary(weekly: pd.Series) -> dict:
        observed_dates = weekly[weekly > 0].index
        if len(observed_dates):
            first, last = observed_dates.min(), observed_dates.max()
            after_start = weekly.loc[weekly.index >= first]
            pre_source = int((weekly.index < first).sum())
            missing_after_start = int((after_start == 0).sum())
            median_after_start = float(after_start.median())
            first_s = str(pd.Timestamp(first).date())
            last_s = str(pd.Timestamp(last).date())
        else:
            pre_source = int(len(weekly))
            missing_after_start = 0
            median_after_start = 0.0
            first_s = last_s = None
        return {
            "weeks_any": int((weekly > 0).sum()),
            "weeks_complete": int((weekly == 1).sum()),
            "weeks_pre_source": pre_source,
            "weeks_all_missing_after_start": missing_after_start,
            "median_weekly_coverage": float(weekly.median()),
            "median_weekly_coverage_after_start": median_after_start,
            "first_observed": first_s,
            "last_observed": last_s,
        }

    features = {}
    for col in fcols:
        observed = panel[col].notna()
        weekly = observed.groupby(panel["date"]).mean()
        features[col] = {
            "family": _feature_family(col),
            "cell_coverage": float(observed.mean()),
            **_weekly_summary(weekly),
        }

    families = {}
    for family in sorted(set(v["family"] for v in features.values())):
        cols = [c for c, values in features.items() if values["family"] == family]
        cells = panel[cols].notna()
        weekly = cells.groupby(panel["date"]).mean().mean(axis=1)
        families[family] = {
            "features": cols,
            "cell_coverage": float(cells.to_numpy().mean()),
            **_weekly_summary(weekly),
        }
    return {"n_rows": int(len(panel)), "n_weeks": int(panel["date"].nunique()),
            "features": features, "families": families}


def _wide(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values=field).sort_index()


def price_features(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    close, volume = _wide(prices, "close"), _wide(prices, "volume")
    open_ = _wide(prices, "open")
    ret = close.pct_change(fill_method=None)
    weeks = {"1w": 5, "4w": 20, "12w": 60, "26w": 130, "52w": 252}

    out = {}
    for name, days in weeks.items():
        out[f"f_mom_{name}"] = close.pct_change(days, fill_method=None)
    out["f_vol_4w"] = ret.rolling(20).std() * ANNUALIZER
    out["f_vol_12w"] = ret.rolling(60).std() * ANNUALIZER
    out["f_downside_dev"] = ret.clip(upper=0).rolling(60).std() * ANNUALIZER
    dollar = (close * volume).rolling(20).mean()
    out["f_dollar_vol"] = np.log(dollar.where(dollar > 0))
    out["f_abn_volume"] = volume.rolling(20).mean() / volume.rolling(120).mean() - 1
    out["f_hi_52w"] = close / close.rolling(252).max() - 1
    out["f_lo_52w"] = close / close.rolling(252).min() - 1
    out["aux_vol"] = out["f_vol_12w"]

    # Skip-adjusted and intermediate momentum (docs/research recommendation #1;
    # Jegadeesh-Titman 1993 skip-week, Novy-Marx 2012 intermediate horizon).
    # The plain lookbacks above include the most recent days, which carry the
    # OPPOSITE (reversal) signal; these hand the model the clean decomposition:
    #   12w skipping the last week, the classic 12-1 (52w skipping 4w), and
    #   Novy-Marx's t-12m..t-7m return that excludes recent months entirely.
    out["f_mom_12w_skip1w"] = close.shift(5) / close.shift(60) - 1
    out["f_mom_52w_skip4w"] = close.shift(20) / close.shift(252) - 1
    out["f_mom_interm"] = close.shift(130) / close.shift(252) - 1

    # Cleaner one-week reversal: remove the market component using a beta
    # estimated strictly before the five-day return window. Both the beta and
    # realized return end no later than t, so the feature is point-in-time.
    mkt_ret = ret["SPY"] if "SPY" in ret.columns else pd.Series(np.nan, index=ret.index)
    beta_pre_week = ret.rolling(60, min_periods=40).cov(mkt_ret).div(
        mkt_ret.rolling(60, min_periods=40).var(), axis=0
    ).shift(5)
    stock_1w = close.pct_change(5, fill_method=None)
    mkt_1w = (close["SPY"].pct_change(5, fill_method=None) if "SPY" in close
              else pd.Series(np.nan, index=close.index))
    out["f_rev_resid_mkt_1w"] = stock_1w.sub(beta_pre_week.mul(mkt_1w, axis=0))

    # Non-overlapping weekly residual-return lags relative to the next-week
    # forecast. lag1 is the just-completed week ending at t; lag2 ends five
    # trading days earlier, etc. Each beta estimate ends before its return
    # window starts, so no realization helps estimate its own market exposure.
    for lag in range(1, 5):
        end_shift = 5 * (lag - 1)
        start_shift = 5 * lag
        stock_week = close.shift(end_shift).div(close.shift(start_shift)) - 1.0
        market_week = (close["SPY"].shift(end_shift) / close["SPY"].shift(start_shift) - 1.0
                       if "SPY" in close else pd.Series(np.nan, index=close.index))
        beta_before = ret.rolling(60, min_periods=40).cov(mkt_ret).div(
            mkt_ret.rolling(60, min_periods=40).var(), axis=0
        ).shift(start_shift)
        out[f"f_resid_ret_lag{lag}w"] = stock_week.sub(
            beta_before.mul(market_week, axis=0)
        )

    # Amihud illiquidity: absolute daily return per dollar traded. Scale by 1e6
    # for numerical readability; cross-sectional ranking makes the scale neutral.
    dollar_volume = close * volume
    amihud_daily = ret.abs().div(dollar_volume.where(dollar_volume > 0)) * 1e6
    out["f_amihud_4w"] = amihud_daily.rolling(20, min_periods=15).mean()
    out["f_amihud_12w"] = amihud_daily.rolling(60, min_periods=40).mean()

    # Overnight (close_{d-1} -> open_d) vs intraday (open_d -> close_d) split,
    # compounded over the trailing 20 trading days ending at t. Built as
    # cumulative-product "index" series (closed-form, no rolling().apply): the
    # single leading NaN (no close_{d-1} on day 0) is filled to 1x so it cancels
    # exactly out of every ratio, matching close.pct_change(20)'s NaN-until-warmup
    # behavior for windows that don't yet have 20 trading days of history.
    overnight = open_ / close.shift(1) - 1
    intraday = close / open_ - 1
    on_cum = (1 + overnight.fillna(0)).cumprod()
    ia_cum = (1 + intraday.fillna(0)).cumprod()
    on_valid = overnight.notna().rolling(20).sum().eq(20)
    ia_valid = intraday.notna().rolling(20).sum().eq(20)
    out["f_overnight_4w"] = (on_cum / on_cum.shift(20) - 1).where(on_valid)
    out["f_intraday_4w"] = (ia_cum / ia_cum.shift(20) - 1).where(ia_valid)

    # Beta / idiosyncratic vol vs SPY over a trailing 60-trading-day window.
    # Closed-form rolling cov/var (no python loops); SPY absent -> NaN throughout.
    cov = ret.rolling(60).cov(mkt_ret)
    var_m = mkt_ret.rolling(60).var()
    var_i = ret.rolling(60).var()
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = cov.div(var_m, axis=0)
    idio_var = var_i.sub(beta.pow(2).multiply(var_m, axis=0)).clip(lower=0)
    out["f_beta_60d"] = beta
    out["f_idio_vol_60d"] = np.sqrt(idio_var) * ANNUALIZER

    frames = []
    for col, wide_df in out.items():
        sub = wide_df.reindex(dates).stack(future_stack=True).rename(col)
        frames.append(sub)
    feats = pd.concat(frames, axis=1).reset_index()
    feats.columns = ["date", "ticker", *out.keys()]
    return feats


def market_macro_features(prices: pd.DataFrame, fred_lagged: pd.DataFrame,
                          dates: pd.DatetimeIndex) -> pd.DataFrame:
    spy = _wide(prices[prices.ticker == "SPY"], "close")["SPY"]
    ret = spy.pct_change()
    # Cross-sectional dispersion: std across tickers of the 5-trading-day
    # return ending at t (close_{t-5} -> close_t). Backward-looking only.
    dispersion = _wide(prices, "close").pct_change(5, fill_method=None).std(axis=1, ddof=1)
    mkt = pd.DataFrame({
        "f_mkt_mom_4w": spy.pct_change(20),
        "f_mkt_mom_26w": spy.pct_change(130),
        "f_mkt_vol_4w": ret.rolling(20).std() * ANNUALIZER,
        "f_mkt_dispersion": dispersion,
    }).reindex(dates)

    macro = fred_lagged.reindex(dates.union(fred_lagged.index)).ffill().reindex(dates)
    chg = macro.diff(13)  # 13 rebalance rows ~ one quarter
    macro.columns = [f"f_macro_{c}" for c in macro.columns]
    chg.columns = [f"{c}_chg" for c in macro.columns]
    return pd.concat([mkt, macro, chg], axis=1).rename_axis("date").reset_index()


def sector_relative_momentum(panel: pd.DataFrame) -> pd.DataFrame:
    """Raw f_mom_4w/f_mom_12w minus their same-date same-sector median.

    Must run BEFORE rank normalization (operates on raw momentum values).
    Requires 'date', 'sector', 'f_mom_4w', 'f_mom_12w' columns. Rows with an
    unknown ('sector' is NaN) sector get NaN (groupby drops NaN keys, so their
    transformed result is NaN).
    """
    out = pd.DataFrame(index=panel.index)
    for name in ("f_mom_4w", "f_mom_12w"):
        med = panel.groupby(["date", "sector"])[name].transform("median")
        out[f"{name}_sect"] = panel[name] - med
    return out


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    d = pd.DatetimeIndex(dates)
    week_of_quarter = ((d.dayofyear - 1) % 91) // 7
    return pd.DataFrame({"date": d, "f_month": d.month.astype(float),
                         "f_woq": week_of_quarter.astype(float)})


# The 4-week label spans ~29 calendar days of future prices; anything training
# or splitting on label_4w must purge at least this many calendar days.
LABEL_4W_PURGE_DAYS = 42


def make_labels(prices: pd.DataFrame, dates: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    open_ = _wide(prices, "open")
    cal = open_.index
    entry_idx = cal.searchsorted(dates, side="right")        # first trading day after t
    exit_idx = entry_idx + horizon

    rows = []
    for t, ei, xi in zip(dates, entry_idx, exit_idx):
        if xi >= len(cal):
            entry = open_.iloc[ei] if ei < len(cal) else pd.Series(dtype=float)
            fwd = pd.Series(np.nan, index=entry.index if not entry.empty else open_.columns)
            label_end_date = pd.NaT
        else:
            fwd = open_.iloc[xi] / open_.iloc[ei] - 1.0
            label_end_date = cal[xi]
        grp = pd.DataFrame({"date": t, "ticker": fwd.index, "fwd_ret": fwd.values,
                            "label_end_date": label_end_date})
        rows.append(grp)
    labels = pd.concat(rows, ignore_index=True)
    med = labels.groupby("date")["fwd_ret"].transform("median")
    labels["label"] = labels["fwd_ret"] - med
    return labels


def build_panel(store, cfg) -> pd.DataFrame:
    from stocks_ml.data.fred import load_fred_lagged
    from stocks_ml.data.membership import members_asof
    from stocks_ml.data.prices import drop_corrupt_series
    from stocks_ml.features.events import filing_features, sec8k_features
    from stocks_ml.features.fundamentals import (earnings_quality_features,
                                                 fundamental_features)
    from stocks_ml.features.insiders import insider_features, shares_outstanding_asof, short_features
    from stocks_ml.features.ranking import RANK_EXEMPT_PREFIXES, rank_normalize

    prices = store.read("prices")
    prices, corrupt = drop_corrupt_series(prices)
    if corrupt:
        store.set_manifest("corrupt_tickers", corrupt)
    membership = store.read("membership")
    edgar = store.read("edgar")
    form4 = (store.read("form4") if store.exists("form4") else
            pd.DataFrame(columns=["ticker", "filed", "trans_date", "code", "shares", "value"]))
    shortint = (store.read("shortint") if store.exists("shortint") else
               pd.DataFrame(columns=["ticker", "settlement_date", "publication_date",
                                     "short_interest"]))
    sec8k = (store.read("sec8k") if store.exists("sec8k") else
             pd.DataFrame(columns=["ticker", "accession", "accepted", "filed", "items",
                                   "primary_document", "is_amendment"]))
    fred_lagged = load_fred_lagged(store, cfg.fred_series)

    cal = trading_calendar(prices)
    warmup_start = cfg.backtest_start - pd.Timedelta(days=450)
    dates = rebalance_dates(cal, max(warmup_start, cal.min()), cal.max(),
                            cfg.rebalance_weekday)

    base_rows = []
    for t in dates:
        for ticker in members_asof(membership, t):
            base_rows.append((t, ticker))
    base = pd.DataFrame(base_rows, columns=["date", "ticker"])
    if corrupt:
        base = base[~base["ticker"].isin(corrupt)]

    pfeats = price_features(prices, dates)
    panel = base.merge(pfeats, on=["date", "ticker"], how="left")

    close_wide = _wide(prices, "close").reindex(dates)
    close_wide.index.name = "date"
    close = close_wide.stack(future_stack=True).rename("close")
    close_df = close.reset_index()
    close_df.columns = ["date", "ticker", "close"]
    panel = panel.merge(close_df, on=["date", "ticker"], how="left")
    panel = fundamental_features(edgar, panel)
    panel = earnings_quality_features(edgar, panel)
    panel = panel.merge(filing_features(edgar, prices, dates), on=["date", "ticker"], how="left")
    panel = panel.merge(sec8k_features(sec8k, sorted(prices["ticker"].unique()), dates),
                        on=["date", "ticker"], how="left")

    # dollar volume / raw volume on the FULL daily calendar (prices.index), not
    # reindexed to `dates`: insider_features needs the full trading calendar
    # (via this wide frame's index) to size its trading-day event window.
    volume_wide = _wide(prices, "volume")
    dollar_wide = (_wide(prices, "close") * volume_wide).rolling(20).mean()
    panel = panel.merge(insider_features(form4, dates, dollar_wide), on=["date", "ticker"], how="left")
    # Short-interest features (FINRA data begins 2017-12) are included because a
    # controlled A/B test found ~+0.002 IC. Structural pre-source absence is measured
    # in feature_coverage and neutral-filled after ranking; it must not alter folds.
    shares_out = shares_outstanding_asof(edgar, panel)
    panel = panel.merge(short_features(shortint, shares_out, volume_wide), on=["date", "ticker"], how="left")

    panel = panel.merge(market_macro_features(prices, fred_lagged, dates), on="date", how="left")
    panel = panel.merge(calendar_features(dates), on="date", how="left")
    panel = panel.merge(make_labels(prices, dates, cfg.horizon_days),
                        on=["date", "ticker"], how="left")
    # make_labels computes raw returns for every stored price series. Recenter
    # after the point-in-time membership merge so departed/nonmember tickers
    # (including stale post-acquisition price records) cannot affect targets.
    member_median = panel.groupby("date")["fwd_ret"].transform("median")
    panel["label"] = panel["fwd_ret"] - member_median

    # Four-week label for the monthly-cadence pipeline: identical open-to-open
    # construction at 4x the horizon, recentered on members like the weekly one.
    # Consumers must purge >= its ~4-week calendar span (see pipelines.py).
    labels_4w = make_labels(prices, dates, cfg.horizon_days * 4).rename(columns={
        "fwd_ret": "fwd_ret_4w", "label": "label_4w",
        "label_end_date": "label_end_date_4w"})
    panel = panel.merge(labels_4w, on=["date", "ticker"], how="left")
    member_median_4w = panel.groupby("date")["fwd_ret_4w"].transform("median")
    panel["label_4w"] = panel["fwd_ret_4w"] - member_median_4w

    sector = membership.dropna(subset=["sector"]).drop_duplicates("ticker")
    panel["sector"] = panel["ticker"].map(dict(zip(sector["ticker"], sector["sector"])))
    panel = pd.concat([panel, sector_relative_momentum(panel)], axis=1)
    dummies = pd.get_dummies(panel["sector"], prefix="f_sec", prefix_sep="_", dtype=float)
    dummies.columns = [c.lower().replace(" ", "_") for c in dummies.columns]
    panel = pd.concat([panel.drop(columns=["sector", "close"]), dummies], axis=1)

    panel = panel[panel["date"] >= cfg.backtest_start].reset_index(drop=True)
    # Rank experimental columns too, so future ablations use the exact stored
    # representation that production admission would receive.
    ranked_cols = [c for c in all_feature_cols(panel)
                   if not c.startswith(RANK_EXEMPT_PREFIXES)]
    store.set_manifest("feature_coverage", feature_coverage(panel))
    panel = rank_normalize(panel, ranked_cols)

    store.write("panel", panel)
    return panel
