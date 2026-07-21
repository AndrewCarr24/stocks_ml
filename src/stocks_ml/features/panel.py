from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_WEEK = 5
ANNUALIZER = np.sqrt(252)
_WEEK_ANCHOR = {0: "W-MON", 1: "W-TUE", 2: "W-WED", 3: "W-THU", 4: "W-FRI"}


def trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.sort(prices["date"].unique()))


def rebalance_dates(calendar: pd.DatetimeIndex, start, end, weekday: int = 4) -> pd.DatetimeIndex:
    anchors = pd.date_range(start, end, freq=_WEEK_ANCHOR[weekday])
    idx = calendar.searchsorted(anchors, side="right") - 1
    idx = idx[idx >= 0]
    dates = pd.DatetimeIndex(pd.unique(calendar[idx]))
    return dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]


def _wide(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values=field).sort_index()


def price_features(prices: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    close, volume = _wide(prices, "close"), _wide(prices, "volume")
    open_ = _wide(prices, "open")
    ret = close.pct_change()
    weeks = {"1w": 5, "4w": 20, "12w": 60, "26w": 130, "52w": 252}

    out = {}
    for name, days in weeks.items():
        out[f"f_mom_{name}"] = close.pct_change(days)
    out["f_vol_4w"] = ret.rolling(20).std() * ANNUALIZER
    out["f_vol_12w"] = ret.rolling(60).std() * ANNUALIZER
    out["f_downside_dev"] = ret.clip(upper=0).rolling(60).std() * ANNUALIZER
    dollar = (close * volume).rolling(20).mean()
    out["f_dollar_vol"] = np.log(dollar.where(dollar > 0))
    out["f_abn_volume"] = volume.rolling(20).mean() / volume.rolling(120).mean() - 1
    out["f_hi_52w"] = close / close.rolling(252).max() - 1
    out["f_lo_52w"] = close / close.rolling(252).min() - 1
    out["aux_vol"] = out["f_vol_12w"]

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
    out["f_overnight_4w"] = on_cum / on_cum.shift(20) - 1
    out["f_intraday_4w"] = ia_cum / ia_cum.shift(20) - 1

    # Beta / idiosyncratic vol vs SPY over a trailing 60-trading-day window.
    # Closed-form rolling cov/var (no python loops); SPY absent -> NaN throughout.
    mkt_ret = ret["SPY"] if "SPY" in ret.columns else pd.Series(np.nan, index=ret.index)
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
    return feats.dropna(subset=["f_mom_1w"])


def market_macro_features(prices: pd.DataFrame, fred_lagged: pd.DataFrame,
                          dates: pd.DatetimeIndex) -> pd.DataFrame:
    spy = _wide(prices[prices.ticker == "SPY"], "close")["SPY"]
    ret = spy.pct_change()
    # Cross-sectional dispersion: std across tickers of the 5-trading-day
    # return ending at t (close_{t-5} -> close_t). Backward-looking only.
    dispersion = _wide(prices, "close").pct_change(5).std(axis=1, ddof=1)
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
        else:
            fwd = open_.iloc[xi] / open_.iloc[ei] - 1.0
        grp = pd.DataFrame({"date": t, "ticker": fwd.index, "fwd_ret": fwd.values})
        rows.append(grp)
    labels = pd.concat(rows, ignore_index=True)
    med = labels.groupby("date")["fwd_ret"].transform("median")
    labels["label"] = labels["fwd_ret"] - med
    return labels


def build_panel(store, cfg) -> pd.DataFrame:
    from stocks_ml.data.fred import load_fred_lagged
    from stocks_ml.data.membership import members_asof
    from stocks_ml.data.prices import drop_corrupt_series
    from stocks_ml.features.events import filing_features
    from stocks_ml.features.fundamentals import fundamental_features
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

    pfeats = price_features(prices, dates)
    panel = base.merge(pfeats, on=["date", "ticker"], how="inner")

    close_wide = _wide(prices, "close").reindex(dates)
    close_wide.index.name = "date"
    close = close_wide.stack(future_stack=True).rename("close")
    close_df = close.reset_index()
    close_df.columns = ["date", "ticker", "close"]
    panel = panel.merge(close_df, on=["date", "ticker"], how="left")
    panel = fundamental_features(edgar, panel)
    panel = panel.merge(filing_features(edgar, prices, dates), on=["date", "ticker"], how="left")

    # dollar volume / raw volume on the FULL daily calendar (prices.index), not
    # reindexed to `dates`: insider_features needs the full trading calendar
    # (via this wide frame's index) to size its trading-day event window.
    volume_wide = _wide(prices, "volume")
    dollar_wide = (_wide(prices, "close") * volume_wide).rolling(20).mean()
    panel = panel.merge(insider_features(form4, dates, dollar_wide), on=["date", "ticker"], how="left")
    # Short-interest features (FINRA data begins 2017-12) are included: the A/B test
    # showed a consistent ~+0.002 IC contribution. Their pre-2018 all-NaN region made
    # the 2012-15 CV fold prone to degenerate fits, so that fold is excluded from
    # evaluation via cfg.eval_start (2015-03) — see config.yaml.
    shares_out = shares_outstanding_asof(edgar, panel)
    panel = panel.merge(short_features(shortint, shares_out, volume_wide), on=["date", "ticker"], how="left")

    panel = panel.merge(market_macro_features(prices, fred_lagged, dates), on="date", how="left")
    panel = panel.merge(calendar_features(dates), on="date", how="left")
    panel = panel.merge(make_labels(prices, dates, cfg.horizon_days),
                        on=["date", "ticker"], how="left")

    sector = membership.dropna(subset=["sector"]).drop_duplicates("ticker")
    panel["sector"] = panel["ticker"].map(dict(zip(sector["ticker"], sector["sector"])))
    panel = pd.concat([panel, sector_relative_momentum(panel)], axis=1)
    dummies = pd.get_dummies(panel["sector"], prefix="f_sec", prefix_sep="_", dtype=float)
    dummies.columns = [c.lower().replace(" ", "_") for c in dummies.columns]
    panel = pd.concat([panel.drop(columns=["sector", "close"]), dummies], axis=1)

    ranked_cols = [c for c in feature_cols(panel) if not c.startswith(RANK_EXEMPT_PREFIXES)]
    panel = rank_normalize(panel, ranked_cols)

    panel = panel[panel["date"] >= cfg.backtest_start].reset_index(drop=True)
    store.write("panel", panel)
    return panel
