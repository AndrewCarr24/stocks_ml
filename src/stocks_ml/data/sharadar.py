"""Sharadar (Nasdaq Data Link) data source: survivorship-free prices + PIT S&P membership.

Why this source (see AGENTS.md open items): it closes the project's four
measured data holes — delisted price histories (the yfinance ratchet),
unadjusted prices with split factors (the $5-floor traps), licensed S&P 500
membership history since 1957 (replacing the Wikipedia scraper), and a plain
REST API usable from CI.

v1 scope (free-tier plumbing, owner-sequenced before subscribing): fetchers
with cursor pagination, schema mapping to project conventions, and store
writers under NEW keys (sharadar_prices / sharadar_tickers / sharadar_sp500).
Production ingestion does NOT switch automatically — promotion to the main
prices/membership keys happens only after the paid-tier coverage audit passes.

Key handling: SHARADAR_API_KEY env var, else data/.sharadar_key (untracked).
The key must never be committed; CI gets it as an Actions secret.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.sharadar.com/v1.0/data/{table}"
PAGE_LIMIT = 10000
PAGE_PAUSE_S = 0.3


def api_key(data_dir: str | Path = "data") -> str:
    key = os.environ.get("SHARADAR_API_KEY", "").strip()
    if key:
        return key
    path = Path(data_dir) / ".sharadar_key"
    if path.exists():
        return path.read_text().strip()
    raise RuntimeError("no Sharadar key: set SHARADAR_API_KEY or create data/.sharadar_key")


def fetch_table(table_name: str, key: str, fetch_fn=None, max_pages: int = 500,
                **filters) -> pd.DataFrame:
    """All rows of a Sharadar Direct table matching `filters` (limit/offset paging).

    Direct API (https://sharadar.com/llms.txt): GET /data/{table}, key in the
    x-api-key header (never the URL), response {"count": N, "data": [records]}.
    fetch_fn(url, params, headers) -> parsed-JSON dict is injectable for tests
    (house rule: no network in tests). Errors fail loudly: a 403 "Exceeds free
    tier" is a subscription boundary, not an empty result."""
    def default_fetch(url, params, headers):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Sharadar {table_name}: HTTP {r.status_code}: "
                               f"{r.text[:200]}")
        return r.json()

    fetch = fetch_fn or default_fetch
    url = BASE.format(table=table_name)
    headers = {"x-api-key": key}
    frames, offset, pages = [], 0, 0
    while True:
        d = fetch(url, {**filters, "format": "json", "limit": PAGE_LIMIT,
                        "offset": offset}, headers)
        rows = d.get("data", [])
        frames.append(pd.DataFrame(rows))
        pages += 1
        if len(rows) < PAGE_LIMIT:
            break
        if pages >= max_pages:
            raise RuntimeError(f"{table_name}: exceeded {max_pages} pages; use bulk")
        offset += PAGE_LIMIT
        if fetch_fn is None:
            time.sleep(PAGE_PAUSE_S)
    out = pd.concat(frames, ignore_index=True)
    for c in ("date", "lastupdated"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def fetch_prices(tickers: list[str], start, key: str, fetch_fn=None) -> pd.DataFrame:
    """Daily bars for tickers since `start`, in the project's prices schema
    plus Sharadar's adjustment columns.

    SEP semantics (documented, load-bearing): open/high/low/close are
    SPLIT-adjusted only; closeadj is split+dividend adjusted; closeunadj is
    as-traded. We surface all three closes so contemporaneous-price analyses
    (the $5 floor) and total-return analyses stop sharing one ambiguous column."""
    raw = fetch_table("stocks", key, fetch_fn=fetch_fn,
                      ticker=",".join(tickers),
                      **{"from": pd.Timestamp(start).date().isoformat()})
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low",
                                     "close", "volume", "closeadj", "closeunadj"])
    out = raw[["date", "ticker", "open", "high", "low", "close", "volume",
               "closeadj", "closeunadj"]].copy()
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_tickers_meta(key: str, fetch_fn=None) -> pd.DataFrame:
    """Ticker metadata (tickers table): includes
    isdelisted — the coverage-audit column the free source can't provide."""
    raw = fetch_table("tickers", key, fetch_fn=fetch_fn)
    keep = [c for c in ("ticker", "name", "exchange", "isdelisted", "category",
                        "sector", "industry", "firstpricedate", "lastpricedate")
            if c in raw.columns]
    return raw[keep].copy()


def fetch_sp500_membership(key: str, fetch_fn=None) -> pd.DataFrame:
    """S&P 500 membership events (sp500 table): rows of action
    ('added'/'removed'/'current') with dates — the licensed replacement for
    the Wikipedia changes table, mapped to the membership builder's schema."""
    raw = fetch_table("sp500", key, fetch_fn=fetch_fn)
    if raw.empty:
        return pd.DataFrame(columns=["date", "action", "ticker", "name"])
    keep = [c for c in ("date", "action", "ticker", "name") if c in raw.columns]
    out = raw[keep].copy()
    out["action"] = out["action"].str.lower()
    return out.sort_values("date").reset_index(drop=True)


def ingest_sharadar(store, tickers: list[str], start, data_dir="data",
                    fetch_fn=None) -> dict:
    """Fetch and persist all three datasets under sharadar_* store keys.

    Deliberately separate from the production prices/membership keys: the
    swap happens after the paid-tier audit, not silently."""
    key = api_key(data_dir)
    prices = fetch_prices(tickers, start, key, fetch_fn=fetch_fn)
    store.write("sharadar_prices", prices)
    meta = fetch_tickers_meta(key, fetch_fn=fetch_fn)
    store.write("sharadar_tickers", meta)
    sp500 = fetch_sp500_membership(key, fetch_fn=fetch_fn)
    store.write("sharadar_sp500", sp500)
    return {"prices_rows": len(prices),
            "prices_tickers": int(prices["ticker"].nunique()) if len(prices) else 0,
            "tickers_meta": len(meta),
            "delisted_in_meta": int((meta.get("isdelisted") == "Y").sum())
            if "isdelisted" in meta else 0,
            "sp500_events": len(sp500)}
