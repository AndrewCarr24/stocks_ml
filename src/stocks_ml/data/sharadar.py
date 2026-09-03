"""Sharadar Direct (api.sharadar.com): the licensed source behind the world store.

It closes the free-data holes the legacy pipeline measured — delisted price
histories, unadjusted prices, licensed S&P 500 membership since 1957 — and
is a plain REST API usable from Actions. data/world.py does the table
mapping; this module is the transport: key handling and cursor pagination.

Key handling: SHARADAR_API_KEY env var, else data/.sharadar_key (untracked).
The key must never be committed; Actions gets it as a repository secret.
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
