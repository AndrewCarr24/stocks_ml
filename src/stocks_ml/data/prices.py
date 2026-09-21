from __future__ import annotations

import pandas as pd

CORRUPT_RATIO = 1.9          # adjusted data should contain no split-sized jumps
CORRUPT_MIN_EVENTS = 2       # one genuine mega-move is possible; repeats are corruption


def drop_corrupt_series(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cut a ticker's rows from the day its adjustment history is shown to
    be broken, to the end of its series: point in time, never the whole
    history.

    Evidence: the total-return close jumps by a split-sized ratio (>1.9x or
    <1/1.9x day over day) while the split-adjusted close (close_split, else
    the raw print closeunadj) does not — a real move shows in both series, a
    broken dividend/spin-off adjustment only in the total-return one. One
    such day is proof (the two series cannot disagree by a split factor).
    The split-adjusted close is the reference because a split and a
    spin-off on one day cancel in the raw print (IAC/Match, 2008-08-21) but
    not in either adjusted series. Without a reference, two adjusted jumps
    are the evidence (one genuine mega-move is possible) and the cut starts
    at the second. The whole-series drop this replaces (2026-09-21) read the future
    and removed real events: Lehman from 1998-2008 for September 2008, 46
    S&P names in the top-2000 build — bankruptcy tails, crisis days, the
    GameStop squeeze — none of them corrupt data. Returns (prices,
    {ticker: first cut date})."""
    p = prices.sort_values(["ticker", "date"])
    ratio = p.groupby("ticker")["close"].pct_change().add(1.0)
    jump = (ratio > CORRUPT_RATIO) | (ratio < 1.0 / CORRUPT_RATIO)
    ref = next((c for c in ("close_split", "closeunadj") if c in p.columns), None)
    if ref is not None:
        r2 = p.groupby("ticker")[ref].pct_change().add(1.0)
        raw_jump = (r2 > CORRUPT_RATIO) | (r2 < 1.0 / CORRUPT_RATIO)
        evidence = jump & ~raw_jump & r2.notna()
        needed = 1
    else:
        evidence = jump
        needed = CORRUPT_MIN_EVENTS
    nth = evidence.groupby(p["ticker"]).cumsum()
    first = p.loc[evidence & (nth == needed)].groupby("ticker")["date"].min()
    cut = {t: pd.Timestamp(d) for t, d in first.items()}
    if not cut:
        return prices.reset_index(drop=True), {}
    cut_from = p["ticker"].map(cut)
    keep = cut_from.isna() | (p["date"] < cut_from)
    return p[keep].reset_index(drop=True), cut
