from __future__ import annotations

import io

import pandas as pd
import requests

# Verified 2026-07-19: FINRA's public "Query API" (api.finra.org) is reachable
# with a plain unauthenticated POST -- no API key or account registration
# required. Endpoint/dataset confirmed via https://api.finra.org/metadata/group/
# otcMarket/name/consolidatedShortInterest. That metadata endpoint describes the
# data as "available online for one rolling year", which understates actual
# reach: date-range queries empirically returned records back to 2017-12-29 for
# a sampled ticker (continuous coverage from ~2018 onward), well past a rolling
# year from today. No `compareFilters` is required for a full-market pull --
# `dateRangeFilters` alone works and pages (max 5000 rows/page) through all
# symbols for the requested window. There is no separate/deeper "no-key" bulk
# archive beyond this API; FINRA's OTCE portal advertises archives back to 2014
# but that path is a JS SPA behind more surface -- not pursued (see report).
API_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
PAGE_LIMIT = 5000
# FINRA's short-interest files carry a settlement date but no publication date.
# Its dissemination schedule publishes short interest ~9 business days after
# settlement; 14 calendar days is the conservative (always-safe) upper bound
# used here so a feature is never marked visible before it truly was.
PUBLICATION_LAG_DAYS = 14
SHORTINT_COLS = ["ticker", "settlement_date", "publication_date", "short_interest"]
DEFAULT_START = pd.Timestamp("2006-01-01")  # earlier than any real coverage; harmless


def fetch_shortint(user_agent: str, start_date, end_date, session=None) -> pd.DataFrame:
    """Network (thin): FINRA's public Query API, paginated over
    [start_date, end_date] (inclusive) by settlement date. No key/registration."""
    sess = session or requests
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    headers = {"User-Agent": user_agent, "Content-Type": "application/json"}

    frames, offset = [], 0
    while True:
        body = {
            "limit": PAGE_LIMIT,
            "offset": offset,
            "dateRangeFilters": [{"fieldName": "settlementDate",
                                  "startDate": start, "endDate": end}],
        }
        resp = sess.post(API_URL, json=body, headers=headers, timeout=60)
        if resp.status_code == 204:  # no content for this window
            break
        resp.raise_for_status()
        page = pd.read_csv(io.StringIO(resp.text))
        if page.empty:
            break
        frames.append(page)
        if len(page) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def extract_shortint(raw: pd.DataFrame) -> pd.DataFrame:
    """Pure: FINRA raw columns -> ticker, settlement_date, publication_date,
    short_interest. publication_date = settlement_date + 14 calendar days
    (see PUBLICATION_LAG_DAYS)."""
    if raw.empty:
        return pd.DataFrame(columns=SHORTINT_COLS)
    from stocks_ml.data.membership import normalize_symbol

    out = pd.DataFrame({
        "ticker": raw["symbolCode"].map(normalize_symbol),
        "settlement_date": pd.to_datetime(raw["settlementDate"]),
        "short_interest": pd.to_numeric(raw["currentShortPositionQuantity"], errors="coerce"),
    })
    out["publication_date"] = out["settlement_date"] + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    return out[SHORTINT_COLS]


def ingest_shortint(store, user_agent, fetch_fn=None) -> dict:
    """Best-effort FINRA short-interest ingest. Incremental: refetches only the
    window after the latest previously-ingested settlement date. A fetch
    failure is recorded (non-fatal) rather than raised; the existing dataset is
    preserved untouched.
    """
    fetch = fetch_fn or fetch_shortint
    existing = store.read("shortint") if store.exists("shortint") else None

    if existing is not None and not existing.empty:
        start = existing["settlement_date"].max() + pd.Timedelta(days=1)
    else:
        start = DEFAULT_START
    end = pd.Timestamp.today().normalize()

    failed = False
    if start <= end:
        try:
            raw = fetch(user_agent, start, end)
            new = extract_shortint(raw)
        except Exception:
            new = pd.DataFrame(columns=SHORTINT_COLS)
            failed = True
    else:
        new = pd.DataFrame(columns=SHORTINT_COLS)

    parts = [f for f in (existing, new) if f is not None and not f.empty]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SHORTINT_COLS)
    if not df.empty:
        df = (df.drop_duplicates(subset=["ticker", "settlement_date"])
                .sort_values(["ticker", "settlement_date"]).reset_index(drop=True))
    store.write("shortint", df)

    summary = {
        "coverage_start": str(df["settlement_date"].min().date()) if not df.empty else None,
        "coverage_end": str(df["settlement_date"].max().date()) if not df.empty else None,
        "n_rows": int(len(df)),
        "fetch_failed": failed,
    }
    store.set_manifest("shortint", summary)
    return summary
