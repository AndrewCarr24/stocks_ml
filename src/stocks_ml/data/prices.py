from __future__ import annotations

import time

import pandas as pd

BATCH = 100
FIELDS = ["open", "high", "low", "close", "volume"]


def all_tickers_ever(membership: pd.DataFrame) -> list[str]:
    return sorted(set(membership["ticker"]).union({"SPY"}))


def _fetch_yfinance(tickers: list[str], start, end) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i : i + BATCH]
        for attempt in range(3):
            try:
                raw = yf.download(chunk, start=start, end=end, auto_adjust=True,
                                  group_by="ticker", progress=False, threads=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        if raw.empty:
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
    fetch_start = start
    existing = store.read("prices") if store.exists("prices") else None
    if existing is not None and not existing.empty:
        fetch_start = existing["date"].max()  # refetch last day; dedupe below

    new = fetch(tickers, fetch_start, end)
    new["date"] = pd.to_datetime(new["date"])
    got = set(new["ticker"].unique())
    failed = sorted(set(tickers) - got)

    df = pd.concat([existing, new], ignore_index=True) if existing is not None else new
    df = (df.drop_duplicates(subset=["date", "ticker"], keep="last")
            .sort_values(["ticker", "date"]).reset_index(drop=True))
    store.write("prices", df)

    summary = {"n_ok": len(got), "failed_tickers": failed,
               "last_date": str(df["date"].max().date()) if not df.empty else None}
    store.set_manifest("prices", summary)
    return summary
