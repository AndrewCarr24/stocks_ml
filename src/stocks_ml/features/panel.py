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
    mkt = pd.DataFrame({
        "f_mkt_mom_4w": spy.pct_change(20),
        "f_mkt_mom_26w": spy.pct_change(130),
        "f_mkt_vol_4w": ret.rolling(20).std() * ANNUALIZER,
    }).reindex(dates)

    macro = fred_lagged.reindex(dates.union(fred_lagged.index)).ffill().reindex(dates)
    chg = macro.diff(13)  # 13 rebalance rows ~ one quarter
    macro.columns = [f"f_macro_{c}" for c in macro.columns]
    chg.columns = [f"{c}_chg" for c in macro.columns]
    return pd.concat([mkt, macro, chg], axis=1).rename_axis("date").reset_index()


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
