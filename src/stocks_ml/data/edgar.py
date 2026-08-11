from __future__ import annotations

import time

import pandas as pd
import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MONEY_UNIT, SHARE_UNIT = "USD", "shares"
EDGAR_COLS = ["ticker", "concept", "start", "end", "filed", "val", "form"]
REFRESH_DAYS = 90  # a ticker's facts are refetched once its newest filing is older than this


def load_cik_map(user_agent: str) -> dict[str, int]:
    resp = requests.get(TICKERS_URL, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    from stocks_ml.data.membership import normalize_symbol
    return {normalize_symbol(v["ticker"]): int(v["cik_str"]) for v in resp.json().values()}


def _fetch_facts(cik: int, user_agent: str) -> dict:
    resp = requests.get(FACTS_URL.format(cik=cik), headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_facts(cf_json: dict, ticker: str, concept_map: dict) -> pd.DataFrame:
    gaap = cf_json.get("facts", {}).get("us-gaap", {})
    dei = cf_json.get("facts", {}).get("dei", {})
    rows = []
    for concept, tags in concept_map.items():
        unit = SHARE_UNIT if concept == "shares" else MONEY_UNIT
        for tag in tags:
            node = gaap.get(tag) or dei.get(tag)
            if not node or unit not in node.get("units", {}):
                continue
            n_before = len(rows)
            for item in node["units"][unit]:
                if item.get("val") is None or not item.get("filed"):
                    continue
                rows.append({
                    "ticker": ticker, "concept": concept,
                    "start": pd.to_datetime(item.get("start")) if item.get("start") else pd.NaT,
                    "end": pd.to_datetime(item["end"]),
                    "filed": pd.to_datetime(item["filed"]),
                    "val": float(item["val"]), "form": item.get("form", ""),
                })
            if len(rows) > n_before:
                break  # first tag that yields data wins
    return pd.DataFrame(rows, columns=EDGAR_COLS)


def ingest_edgar(store, tickers, concept_map, user_agent,
                 fetch_facts_fn=None, cik_map=None) -> dict:
    fetch_facts = fetch_facts_fn or _fetch_facts
    ciks = cik_map or load_cik_map(user_agent)
    existing = store.read("edgar") if store.exists("edgar") else None
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=REFRESH_DAYS)
    fresh = set()
    if existing is not None and "ticker" in existing.columns:
        max_filed = existing.groupby("ticker")["filed"].max()
        fresh = set(max_filed[max_filed >= cutoff].index)

    frames, failed = [], []
    for t in tickers:
        if t in fresh:
            continue
        cik = ciks.get(t)
        if cik is None:
            failed.append(t)
            continue
        try:
            frames.append(extract_facts(fetch_facts(cik, user_agent), t, concept_map))
        except Exception:
            failed.append(t)
        if fetch_facts_fn is None:
            time.sleep(0.12)  # SEC rate limit: 10 req/s

    nonempty = [frame for frame in frames if not frame.empty]
    new = (pd.concat(nonempty, ignore_index=True) if nonempty
           else pd.DataFrame(columns=EDGAR_COLS))
    parts = [f for f in (existing, new) if f is not None and not f.empty]
    df = pd.concat(parts, ignore_index=True) if parts else new
    if not df.empty:
        df = df.drop_duplicates().sort_values(["ticker", "concept", "filed"]).reset_index(drop=True)
    store.write("edgar", df)
    summary = {"n_ok": int(df["ticker"].nunique()) if not df.empty else 0,
               "failed_tickers": sorted(failed)}
    store.set_manifest("edgar", summary)
    return summary
