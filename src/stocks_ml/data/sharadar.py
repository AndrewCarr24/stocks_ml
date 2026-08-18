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

BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/{table}.json"
PAGE_PAUSE_S = 0.6            # free tier is throttle-happy; be polite


def api_key(data_dir: str | Path = "data") -> str:
    key = os.environ.get("SHARADAR_API_KEY", "").strip()
    if key:
        return key
    path = Path(data_dir) / ".sharadar_key"
    if path.exists():
        return path.read_text().strip()
    raise RuntimeError("no Sharadar key: set SHARADAR_API_KEY or create data/.sharadar_key")


def fetch_datatable(table_name: str, key: str, fetch_fn=None, max_pages: int = 200,
                    **filters) -> pd.DataFrame:
    """All rows of a SHARADAR datatable matching `filters`, following cursors.

    fetch_fn(url, params) -> parsed-JSON dict is injectable for tests (house
    rule: no network in tests). Raises on API errors with the vendor message
    surfaced — a disabled/unentitled key should fail loudly, not emptily."""
    def default_fetch(url, params):
        r = requests.get(url, params=params, timeout=60)
        d = r.json()
        if "quandl_error" in d:
            raise RuntimeError(f"Sharadar API error on {table_name}: "
                               f"{d['quandl_error'].get('message')}")
        r.raise_for_status()
        return d

    fetch = fetch_fn or default_fetch
    url = BASE.format(table=table_name)
    params = {**filters, "api_key": key}
    frames, cursor, pages = [], None, 0
    while True:
        page_params = dict(params)
        if cursor:
            page_params["qopts.cursor_id"] = cursor
        d = fetch(url, page_params)
        dt = d["datatable"]
        cols = [c["name"] for c in dt["columns"]]
        frames.append(pd.DataFrame(dt["data"], columns=cols))
        cursor = (d.get("meta") or {}).get("next_cursor_id")
        pages += 1
        if not cursor:
            break
        if pages >= max_pages:
            raise RuntimeError(f"{table_name}: exceeded {max_pages} pages; narrow the query")
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
    raw = fetch_datatable("SEP", key, fetch_fn=fetch_fn,
                          ticker=",".join(tickers),
                          **{"date.gte": pd.Timestamp(start).date().isoformat()})
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low",
                                     "close", "volume", "closeadj", "closeunadj"])
    out = raw[["date", "ticker", "open", "high", "low", "close", "volume",
               "closeadj", "closeunadj"]].copy()
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_tickers_meta(key: str, fetch_fn=None) -> pd.DataFrame:
    """Ticker metadata (SHARADAR/TICKERS, table=SEP universe): includes
    isdelisted — the coverage-audit column the free source can't provide."""
    raw = fetch_datatable("TICKERS", key, fetch_fn=fetch_fn, table="SEP")
    keep = [c for c in ("ticker", "name", "exchange", "isdelisted", "category",
                        "sector", "industry", "firstpricedate", "lastpricedate")
            if c in raw.columns]
    return raw[keep].copy()


def fetch_sp500_membership(key: str, fetch_fn=None) -> pd.DataFrame:
    """S&P 500 membership events (SHARADAR/SP500): rows of action
    ('added'/'removed'/'current') with dates — the licensed replacement for
    the Wikipedia changes table, mapped to the membership builder's schema."""
    raw = fetch_datatable("SP500", key, fetch_fn=fetch_fn)
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
