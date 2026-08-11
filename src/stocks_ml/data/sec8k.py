from __future__ import annotations

import time
from collections.abc import Callable

import pandas as pd
import requests

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
SEC8K_COLS = [
    "ticker", "accession", "accepted", "filed", "items", "primary_document",
    "is_amendment",
]


def _get_json(url: str, user_agent: str) -> dict:
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    response.raise_for_status()
    return response.json()


def _rows_from_recent(recent: dict, ticker: str) -> list[dict]:
    """Extract immutable 8-K accession metadata from one SEC submissions block."""
    keys = ("accessionNumber", "acceptanceDateTime", "filingDate", "form",
            "items", "primaryDocument")
    n = len(recent.get("accessionNumber", []))
    rows = []
    for i in range(n):
        values = {key: (recent.get(key, [None] * n)[i]
                        if i < len(recent.get(key, [])) else None)
                  for key in keys}
        form = str(values["form"] or "").upper()
        if form not in {"8-K", "8-K/A"}:
            continue
        accession = values["accessionNumber"]
        if not accession:
            continue
        accepted = pd.to_datetime(values["acceptanceDateTime"], utc=True, errors="coerce")
        filed = pd.to_datetime(values["filingDate"], errors="coerce")
        rows.append({
            "ticker": ticker,
            "accession": str(accession),
            "accepted": accepted,
            "filed": filed,
            "items": str(values["items"] or ""),
            "primary_document": str(values["primaryDocument"] or ""),
            "is_amendment": form.endswith("/A"),
        })
    return rows


def extract_8k_submissions(submissions: dict, ticker: str,
                           fetch_file: Callable[[str], dict] | None = None) -> pd.DataFrame:
    """Extract all available 8-K metadata, including SEC historical fragments.

    Accession numbers and acceptance timestamps are preserved as published. The
    economic event/report date is intentionally ignored: features become usable
    only after SEC publication, never when the underlying event allegedly occurred.
    """
    rows = _rows_from_recent(submissions.get("filings", {}).get("recent", {}), ticker)
    if fetch_file is not None:
        for info in submissions.get("filings", {}).get("files", []):
            name = info.get("name")
            if name:
                payload = fetch_file(name)
                recent = payload.get("filings", {}).get("recent", payload)
                rows.extend(_rows_from_recent(recent, ticker))
    frame = pd.DataFrame(rows, columns=SEC8K_COLS)
    if not frame.empty:
        frame = (frame.drop_duplicates("accession", keep="last")
                       .sort_values(["ticker", "filed", "accession"])
                       .reset_index(drop=True))
    return frame


def ingest_sec8k(store, tickers, user_agent: str, fetch_submissions_fn=None,
                  fetch_file_fn=None, cik_map=None) -> dict:
    """Fetch SEC 8-K metadata and merge by immutable accession number."""
    from stocks_ml.data.edgar import load_cik_map

    fetch_submissions = fetch_submissions_fn or (
        lambda cik: _get_json(SUBMISSIONS_URL.format(cik=cik), user_agent)
    )
    fetch_file = fetch_file_fn or (
        lambda name: _get_json(SUBMISSIONS_FILE_URL.format(name=name), user_agent)
    )
    ciks = cik_map or load_cik_map(user_agent)
    existing = store.read("sec8k") if store.exists("sec8k") else None
    existing_tickers = (set(existing["ticker"].unique())
                        if existing is not None and not existing.empty else set())
    frames, failed = [], []
    for ticker in tickers:
        cik = ciks.get(ticker)
        if cik is None:
            failed.append(ticker)
            continue
        try:
            payload = fetch_submissions(cik)
            # Historical submission fragments are immutable and expensive; once
            # stored, refresh only the SEC's recent block for that ticker.
            historical_fetch = None if ticker in existing_tickers else fetch_file
            frames.append(extract_8k_submissions(payload, ticker, historical_fetch))
        except Exception:
            failed.append(ticker)
        if fetch_submissions_fn is None:
            time.sleep(0.12)

    new = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
        not f.empty for f in frames
    ) else pd.DataFrame(columns=SEC8K_COLS)
    parts = [f for f in (existing, new) if f is not None and not f.empty]
    combined = pd.concat(parts, ignore_index=True) if parts else new
    if not combined.empty:
        # Existing rows come first and win: once an accession has entered the
        # point-in-time store, later source revisions may not rewrite history.
        combined = (combined.drop_duplicates("accession", keep="first")
                            .sort_values(["ticker", "filed", "accession"])
                            .reset_index(drop=True))
    store.write("sec8k", combined)
    summary = {
        "n_tickers": int(combined["ticker"].nunique()) if not combined.empty else 0,
        "n_filings": int(len(combined)),
        "failed_tickers": sorted(failed),
    }
    store.set_manifest("sec8k", summary)
    return summary
