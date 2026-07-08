from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch_series(series_id: str, user_agent: str) -> pd.Series:
    resp = requests.get(FRED_CSV.format(sid=series_id),
                        headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(pd.to_numeric(df[val_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col]), name=series_id)
    return s.dropna()


def ingest_fred(store, series_lags: dict, user_agent: str, fetch_fn=None) -> dict:
    fetch = fetch_fn or fetch_series
    frames = []
    for sid in series_lags:
        s = fetch(sid, user_agent)
        frames.append(pd.DataFrame({"date": s.index, "series": sid, "value": s.values}))
    df = pd.concat(frames, ignore_index=True)
    store.write("fred", df)
    summary = {"series": sorted(series_lags), "last_date": str(df["date"].max().date())}
    store.set_manifest("fred", summary)
    return summary


def load_fred_lagged(store, series_lags: dict) -> pd.DataFrame:
    raw = store.read("fred")
    wide = raw.pivot(index="date", columns="series", values="value").sort_index()
    daily = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    out = {}
    for sid, lag in series_lags.items():
        if sid not in daily.columns:
            continue
        s = daily[sid].dropna()
        s.index = s.index + pd.Timedelta(days=lag)
        out[sid] = s
    lagged = pd.DataFrame(out)
    # start the index at the RAW minimum date so pre-release days exist (as NaN)
    full_idx = pd.date_range(wide.index.min(), lagged.index.max(), freq="D")
    return lagged.reindex(full_idx).ffill()
