from __future__ import annotations

import time

import pandas as pd

BATCH = 100
FIELDS = ["open", "high", "low", "close", "volume"]
CORRUPT_RATIO = 1.9          # adjusted data should contain no split-sized jumps
CORRUPT_MIN_EVENTS = 2       # one genuine mega-move is possible; repeats are corruption


def all_tickers_ever(membership: pd.DataFrame) -> list[str]:
    return sorted(set(membership["ticker"]).union({"SPY"}))


def drop_corrupt_series(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop tickers whose ADJUSTED closes jump by a split-sized ratio (>1.9x or
    <1/1.9x day-over-day) two or more times. Properly adjusted series contain no
    split jumps, so repeated occurrences signal a corrupted adjustment history
    (common for delisted tickers on free sources). A single genuine crash/bounce
    (e.g., AIG March 2009) survives the threshold."""
    ratios = (prices.sort_values(["ticker", "date"])
                    .groupby("ticker")["close"].pct_change().add(1.0))
    extreme = (ratios > CORRUPT_RATIO) | (ratios < 1.0 / CORRUPT_RATIO)
    events = extreme.groupby(prices["ticker"]).sum()
    corrupt = sorted(events[events >= CORRUPT_MIN_EVENTS].index)
    return prices[~prices["ticker"].isin(corrupt)].reset_index(drop=True), corrupt


def _fetch_yfinance(tickers: list[str], start, end) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i : i + BATCH]
        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=start, end=end, auto_adjust=True,
                                  group_by="ticker", progress=False, threads=True)
                break
            except Exception:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        if raw is None or raw.empty:  # persistent failure surfaces via failed_tickers
            continue
        if not isinstance(raw.columns, pd.MultiIndex):  # single-ticker shape
            raw = pd.concat({chunk[0]: raw}, axis=1)
        long = (raw.stack(level=0, future_stack=True)
                   .rename_axis(["date", "ticker"]).reset_index())
        long.columns = [str(c).lower() for c in long.columns]
        frames.append(long)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", *FIELDS])
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["close"])[["date", "ticker", *FIELDS]]


def ingest_prices(store, tickers: list[str], start, end=None, fetch_fn=None) -> dict:
    fetch = fetch_fn or _fetch_yfinance
    existing = store.read("prices") if store.exists("prices") else None

    fetched = []
    if existing is not None and not existing.empty:
        # Global-max cursor: all known tickers resume from the store's overall
        # max date (refetch last day; dedupe below), not a per-ticker max, so
        # this does NOT repair per-ticker gaps. New tickers get full history
        # from the caller's start; per-ticker gaps are repaired via `ingest --full`.
        known = set(existing["ticker"].unique())
        known_tickers = [t for t in tickers if t in known]
        new_tickers = [t for t in tickers if t not in known]
        if known_tickers:
            fetched.append(fetch(known_tickers, existing["date"].max(), end))
        if new_tickers:
            fetched.append(fetch(new_tickers, start, end))
    else:
        fetched.append(fetch(tickers, start, end))

    new = pd.concat(fetched, ignore_index=True)
    new["date"] = pd.to_datetime(new["date"])

    parts = [f for f in (existing, new) if f is not None and not f.empty]
    df = pd.concat(parts, ignore_index=True) if parts else new
    df = (df.drop_duplicates(subset=["date", "ticker"], keep="last")
            .sort_values(["ticker", "date"]).reset_index(drop=True))
    store.write("prices", df)

    # Judge success against the final combined frame so an idempotent re-run
    # that fetches zero new rows does not mark stored tickers as failed.
    present = set(df["ticker"].unique())
    failed = sorted(set(tickers) - present)
    summary = {"n_ok": len(set(tickers) & present), "failed_tickers": failed,
               "last_date": str(df["date"].max().date()) if not df.empty else None}
    store.set_manifest("prices", summary)
    return summary
