from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_DAYS = 91          # ~13 weeks, calendar days
EVT_TRADING_DAYS = 10     # ~2 weeks


def _windowed_asof_sum(events: pd.DataFrame, value_col: str, grid: pd.DataFrame,
                       window_days: int) -> pd.Series:
    """Sum of `events[value_col]` (one row per ticker+filed) with filed in
    (date - window_days, date], for each row of `grid` (columns: date, ticker).
    PIT: an event is only ever visible at dates >= its `filed` date.
    Returns a Series aligned to grid.index.
    """
    if events.empty:
        return pd.Series(0.0, index=grid.index)

    ev = events.sort_values(["ticker", "filed"]).copy()
    ev["cum"] = ev.groupby("ticker")[value_col].cumsum()
    ev_sorted = ev[["ticker", "filed", "cum"]].sort_values("filed")

    def asof_cum(as_of: pd.Series) -> pd.Series:
        left = pd.DataFrame({"ticker": grid["ticker"].to_numpy(), "asof": as_of.to_numpy()},
                            index=grid.index)
        left = left.reset_index().sort_values("asof")
        merged = pd.merge_asof(left, ev_sorted, left_on="asof", right_on="filed",
                               by="ticker", direction="backward")
        return merged.set_index("index")["cum"].reindex(grid.index).fillna(0.0)

    upper = asof_cum(grid["date"])
    lower = asof_cum(grid["date"] - pd.Timedelta(days=window_days))
    return upper - lower


def insider_features(form4: pd.DataFrame, dates: pd.DatetimeIndex,
                     dollar_volume: pd.DataFrame) -> pd.DataFrame:
    """PIT insider Form-4 features, anchored to `filed` dates.

    dollar_volume: wide (date x ticker) 20-day average dollar volume, as
    computed in price_features -- its full DatetimeIndex doubles as the
    trading calendar used for the trading-day event window below (reused
    rather than threading a separate `prices`/calendar argument through).

    "Distinct filings" for f_insider_buyers_13w is approximated as distinct
    (ticker, filed) pairs: the persisted form4 schema (ticker, filed,
    trans_date, code, shares, value) carries no owner/accession identifier, so
    two different insiders filing P trades for the same ticker on the same
    calendar day collapse into one "filing" under this proxy.

    An empty `form4` naturally yields the same defaults as a ticker/window with
    no matching transactions (0.0, not NaN) -- both mean "no insider activity
    observed", not "unknown"; this keeps the corrupt-future no-lookahead
    invariance test consistent regardless of whether the dataset is absent or
    merely empty in a given window.
    """
    tickers = list(dollar_volume.columns)
    grid = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"]).to_frame(index=False)

    f4 = form4.copy()
    # Bulk Form 4 data carries a filing date without a reliable acceptance
    # time. Availability begins the following calendar day.
    f4["filed"] = (pd.to_datetime(f4["filed"]).dt.normalize()
                   + pd.Timedelta(days=1))
    f4["signed_value"] = np.where(f4["code"] == "P", f4["value"], -f4["value"])

    daily_value = f4.groupby(["ticker", "filed"], as_index=False)["signed_value"].sum()
    p_daily = (f4.loc[f4["code"] == "P", ["ticker", "filed"]]
                 .drop_duplicates().assign(n=1.0))

    net_value = _windowed_asof_sum(daily_value, "signed_value", grid, WINDOW_DAYS)
    buyers = _windowed_asof_sum(p_daily, "n", grid, WINDOW_DAYS)

    dv = dollar_volume.reindex(dates).stack(future_stack=True).rename("dv").reset_index()
    dv.columns = ["date", "ticker", "dv"]

    out = grid.copy()
    out["f_insider_net_13w"] = net_value.to_numpy()
    out["f_insider_buyers_13w"] = buyers.to_numpy()
    out = out.merge(dv, on=["date", "ticker"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["f_insider_net_13w"] = out["f_insider_net_13w"] / out["dv"].where(out["dv"] > 0)
    out = out.drop(columns=["dv"])

    # evt flag: most-recent P filing at/before t; recency is monotonic in filed
    # date, so checking only the single latest asof P filing is equivalent to
    # "was ANY P filing seen within the trailing window".
    cal = dollar_volume.index
    p_filed = (f4.loc[f4["code"] == "P", ["ticker", "filed"]].drop_duplicates()
                  .sort_values("filed").rename(columns={"filed": "F"}))
    left = grid.reset_index().sort_values("date")
    merged = (pd.merge_asof(left, p_filed, left_on="date", right_on="F", by="ticker",
                            direction="backward")
                .set_index("index").reindex(grid.index))

    cal_vals = cal.values
    t_idx = np.searchsorted(cal_vals, grid["date"].to_numpy(), side="left")
    has_f = merged["F"].notna().to_numpy()
    f_idx = np.full(len(grid), -1, dtype=np.int64)
    if has_f.any():
        f_idx[has_f] = np.searchsorted(cal_vals, merged.loc[has_f, "F"].to_numpy(), side="right") - 1
    diff = t_idx - f_idx
    out["f_evt_insider_buy_2w"] = np.where(has_f & (diff >= 0) & (diff <= EVT_TRADING_DAYS), 1.0, 0.0)

    return out


def shares_outstanding_asof(edgar: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """PIT asof-join of edgar's 'shares' instant concept onto `base` (date,
    ticker): the latest filed <= t fact per ticker -- the same join
    fundamental_features uses internally for its own 'shares' lookup, exposed
    here as a standalone long frame [date, ticker, shares] for short_features.
    """
    out = base[["date", "ticker"]].copy()
    facts = edgar[edgar["concept"] == "shares"][["ticker", "filed", "val"]].dropna()
    if facts.empty:
        out["shares"] = np.nan
        return out
    facts = facts.copy()
    facts["filed"] = (pd.to_datetime(facts["filed"]).dt.normalize()
                      + pd.Timedelta(days=1))
    facts = facts.sort_values("filed")
    left = out.reset_index().sort_values("date")
    merged = pd.merge_asof(left, facts, left_on="date", right_on="filed", by="ticker",
                           direction="backward")
    out["shares"] = merged.set_index("index")["val"].reindex(out.index)
    return out


def short_features(shortint: pd.DataFrame, shares_outstanding: pd.DataFrame,
                   volume: pd.DataFrame) -> pd.DataFrame:
    """PIT short-interest features, gated on `publication_date <= t`.

    shares_outstanding: long [date, ticker, shares] (see shares_outstanding_asof).
    volume: wide (date x ticker) raw daily share volume (as in price_features'
    input); a trailing 20-trading-day average gives days-to-cover.

    An empty/absent `shortint`, or a ticker/date with no published figure yet,
    both yield NaN via the same merge_asof path (no special-casing needed) --
    "no short-interest figure available" is genuinely unknown, unlike the
    insider count features above where "no activity" is a meaningful zero.
    """
    out = shares_outstanding[["date", "ticker"]].copy()

    si = shortint.copy()
    si["publication_date"] = pd.to_datetime(si["publication_date"])
    # merge_asof requires the right frame globally sorted by the "on" key
    # (publication_date) -- NOT by ["ticker", "publication_date"], which is
    # only sorted within each ticker group and can easily be non-monotonic
    # overall once tickers interleave (real FINRA data: ~3.8M unsorted rows
    # spanning thousands of tickers). Matches the convention already used at
    # every other merge_asof site in this module and in fundamentals.py /
    # events.py (sort by the "on" key alone; `by=` handles grouping).
    si = si.sort_values("publication_date")[
        ["ticker", "publication_date", "short_interest"]]

    left = out.reset_index().sort_values("date")
    merged = pd.merge_asof(left, si, left_on="date", right_on="publication_date",
                           by="ticker", direction="backward")
    out["short_interest"] = merged.set_index("index")["short_interest"].reindex(out.index)

    avg_vol = (volume.rolling(20).mean().stack(future_stack=True)
                    .rename("avg_vol").reset_index())
    avg_vol.columns = ["date", "ticker", "avg_vol"]
    out = out.merge(avg_vol, on=["date", "ticker"], how="left")

    shares = shares_outstanding[["date", "ticker", "shares"]]
    out = out.merge(shares, on=["date", "ticker"], how="left")

    with np.errstate(divide="ignore", invalid="ignore"):
        out["f_short_ratio"] = out["short_interest"] / out["shares"].where(out["shares"] > 0)
        out["f_short_dtc"] = out["short_interest"] / out["avg_vol"].where(out["avg_vol"] > 0)

    return out[["date", "ticker", "f_short_ratio", "f_short_dtc"]]
