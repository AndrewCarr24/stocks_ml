from __future__ import annotations

import numpy as np
import pandas as pd

from stocks_ml.features.panel import trading_calendar


def _wide(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values=field).sort_index()


def filing_features(edgar: pd.DataFrame, prices: pd.DataFrame,
                    dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Filing-window / PEAD features from the edgar store's `filed` dates.

    For each (rebalance date t, ticker), F = the ticker's most recent unique
    `filed` date with filed <= t (any concept/form). All windows are anchored
    to trading days on or before their target date and never look past t.

    Returns a long frame [date, ticker, f_evt_filed_5d, f_days_since_filing,
    f_pead].
    """
    cal = trading_calendar(prices)
    tickers = sorted(prices["ticker"].unique())
    grid = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"]).to_frame(index=False)

    filed = edgar[["ticker", "filed"]].dropna()
    if filed.empty:
        merged = grid.copy()
        merged["F"] = pd.NaT
    else:
        filed = filed.copy()
        filed["F"] = pd.to_datetime(filed["filed"]).dt.normalize()
        filed["available"] = filed["F"] + pd.Timedelta(days=1)
        filed = filed[["ticker", "F", "available"]].drop_duplicates().sort_values("available")
        merged = pd.merge_asof(grid.sort_values("date"), filed,
                       left_on="date", right_on="available", by="ticker",
                               direction="backward")
    merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)

    cal_vals = cal.values
    t_idx = np.searchsorted(cal_vals, merged["date"].to_numpy(), side="left")
    has_f = merged["F"].notna().to_numpy()
    f_idx = np.full(len(merged), -1, dtype=np.int64)
    if has_f.any():
        f_vals = merged.loc[has_f, "F"].to_numpy()
        f_idx[has_f] = np.searchsorted(cal_vals, f_vals, side="right") - 1

    diff = t_idx - f_idx  # trading days between F's anchor day and t

    f_evt_filed_5d = np.where(has_f & (diff >= 0) & (diff <= 5), 1.0, 0.0)
    f_days_since_filing = np.where(has_f, np.minimum(diff, 63).astype(float), np.nan)

    close = _wide(prices, "close").reindex(columns=tickers)
    close_arr = close.to_numpy()
    n_days = close_arr.shape[0]
    ticker_pos = merged["ticker"].map({t: i for i, t in enumerate(tickers)}).to_numpy(dtype=np.int64)

    lo_idx, hi_idx = f_idx - 1, f_idx + 1
    pead_ok = has_f & (diff >= 1) & (lo_idx >= 0) & (hi_idx < n_days)
    safe_lo = np.clip(lo_idx, 0, n_days - 1)
    safe_hi = np.clip(hi_idx, 0, n_days - 1)
    close_lo = close_arr[safe_lo, ticker_pos]
    close_hi = close_arr[safe_hi, ticker_pos]
    with np.errstate(divide="ignore", invalid="ignore"):
        f_pead = np.where(pead_ok, close_hi / close_lo - 1.0, np.nan)

    out = merged[["date", "ticker"]].copy()
    out["f_evt_filed_5d"] = f_evt_filed_5d
    out["f_days_since_filing"] = f_days_since_filing
    out["f_pead"] = f_pead
    return out


def sec8k_features(sec8k: pd.DataFrame, tickers: list[str],
                   dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time 8-K and earnings-announcement recency features.

    Item 2.02 identifies results-of-operations announcements. To avoid timezone
    ambiguity and after-close leakage, every filing becomes usable on the calendar
    day *after* its SEC acceptance timestamp. Amendments are not treated as new
    earnings announcements. Economic event dates are deliberately unused.
    """
    grid = pd.MultiIndex.from_product(
        [dates, sorted(tickers)], names=["date", "ticker"]
    ).to_frame(index=False)
    if sec8k.empty:
        out = grid.copy()
        out["f_evt_8k_7d"] = 0.0
        out["f_evt_earnings_8k_7d"] = 0.0
        out["f_days_since_earnings_8k"] = np.nan
        return out

    filings = sec8k.copy()
    accepted = pd.to_datetime(filings["accepted"], utc=True, errors="coerce")
    filed = pd.to_datetime(filings["filed"], errors="coerce")
    # Filing date is a conservative fallback when old SEC fragments omit the
    # acceptance timestamp; in either case availability starts the next day.
    accepted_date = accepted.dt.tz_convert(None).dt.normalize()
    filings["available"] = accepted_date.fillna(filed.dt.normalize()) + pd.Timedelta(days=1)
    filings["is_earnings"] = (
        filings["items"].fillna("").str.contains(r"(?:^|,\s*)2\.02(?:,|$)", regex=True)
        & ~filings["is_amendment"].fillna(False)
    )
    filings = filings.dropna(subset=["available"])

    def _last_event(mask: pd.Series, name: str) -> pd.DataFrame:
        events = (filings.loc[mask, ["ticker", "available"]]
                         .drop_duplicates()
                         .sort_values("available")
                         .rename(columns={"available": name}))
        if events.empty:
            merged = grid.copy()
            merged[name] = pd.NaT
            return merged
        return pd.merge_asof(grid.sort_values("date"), events,
                             left_on="date", right_on=name, by="ticker",
                             direction="backward").sort_values(["date", "ticker"])

    any_8k = _last_event(pd.Series(True, index=filings.index), "last_8k")
    earnings = _last_event(filings["is_earnings"], "last_earnings_8k")
    out = grid.sort_values(["date", "ticker"]).reset_index(drop=True)
    last_8k = any_8k["last_8k"].reset_index(drop=True)
    last_earnings = earnings["last_earnings_8k"].reset_index(drop=True)
    days_8k = (out["date"] - last_8k).dt.days
    days_earnings = (out["date"] - last_earnings).dt.days
    out["f_evt_8k_7d"] = days_8k.between(0, 6).astype(float)
    out["f_evt_earnings_8k_7d"] = days_earnings.between(0, 6).astype(float)
    out["f_days_since_earnings_8k"] = days_earnings.clip(upper=126).astype(float)
    return out
