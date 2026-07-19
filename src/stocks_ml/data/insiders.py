from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

# Verified against a real download (2023q1) on 2026-07-19: the zip contains
# SUBMISSION.tsv, NONDERIV_TRANS.tsv, DERIV_TRANS.tsv, NONDERIV_HOLDING.tsv,
# DERIV_HOLDING.tsv, REPORTINGOWNER.tsv, OWNER_SIGNATURE.tsv, FOOTNOTES.tsv, plus
# a metadata json/readme. Dates are "DD-MON-YYYY" (e.g. "31-MAR-2023"); ISSUERCIK
# is a zero-padded 10-char string. Earliest quarter confirmed reachable: 2006q1.
ZIP_URL = ("https://www.sec.gov/files/structureddata/data/"
           "insider-transactions-data-sets/{year}q{quarter}_form345.zip")
DATE_FMT = "%d-%b-%Y"
OPEN_MARKET_CODES = {"P", "S"}  # open-market purchase/sale; excludes M/A/F/G/... (comp noise)
FIRST_QUARTER = (2006, 1)
FORM4_COLS = ["ticker", "filed", "trans_date", "code", "shares", "value"]


def fetch_quarter(year: int, q: int, user_agent: str) -> dict[str, pd.DataFrame]:
    """Network (thin): download one SEC quarterly Form 3/4/5 zip and return its
    SUBMISSION and NONDERIV_TRANS tables as raw (string-typed) DataFrames."""
    url = ZIP_URL.format(year=year, quarter=q)
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        submission = pd.read_csv(zf.open("SUBMISSION.tsv"), sep="\t", dtype=str)
        trans = pd.read_csv(zf.open("NONDERIV_TRANS.tsv"), sep="\t", dtype=str)
    return {"SUBMISSION": submission, "NONDERIV_TRANS": trans}


def extract_transactions(submission: pd.DataFrame, trans: pd.DataFrame,
                         cik_to_ticker: dict) -> pd.DataFrame:
    """Pure: join NONDERIV_TRANS -> SUBMISSION on ACCESSION_NUMBER, keep only
    open-market P/S transactions, map issuer CIK -> ticker.

    Output columns: ticker, filed (datetime), trans_date (datetime),
    code ("P"|"S"), shares (float), value (float = shares * price, NaN-safe).
    Rows whose issuer CIK has no ticker mapping are dropped.
    """
    if submission.empty or trans.empty:
        return pd.DataFrame(columns=FORM4_COLS)

    sub = submission[["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK"]].copy()
    sub["filed"] = pd.to_datetime(sub["FILING_DATE"], format=DATE_FMT)
    sub["cik"] = sub["ISSUERCIK"].astype(str).str.strip().astype(int)
    sub["ticker"] = sub["cik"].map(cik_to_ticker)
    sub = sub.dropna(subset=["ticker"])  # unmappable CIKs dropped

    t = trans[["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
              "TRANS_SHARES", "TRANS_PRICEPERSHARE"]].copy()
    t = t[t["TRANS_CODE"].isin(OPEN_MARKET_CODES)]
    if t.empty or sub.empty:
        return pd.DataFrame(columns=FORM4_COLS)

    t["trans_date"] = pd.to_datetime(t["TRANS_DATE"], format=DATE_FMT)
    t["shares"] = pd.to_numeric(t["TRANS_SHARES"], errors="coerce")
    price = pd.to_numeric(t["TRANS_PRICEPERSHARE"], errors="coerce")
    t["value"] = t["shares"] * price  # NaN-safe: NaN price/shares -> NaN value
    t["code"] = t["TRANS_CODE"]

    merged = t.merge(sub[["ACCESSION_NUMBER", "filed", "ticker"]],
                     on="ACCESSION_NUMBER", how="inner")
    return merged[FORM4_COLS].reset_index(drop=True)


def _quarter_of(ts: pd.Timestamp) -> tuple[int, int]:
    return (ts.year, (ts.month - 1) // 3 + 1)


def _quarters_through(end: tuple[int, int]) -> list[tuple[int, int]]:
    y, q = FIRST_QUARTER
    ey, eq = end
    out = []
    while (y, q) <= (ey, eq):
        out.append((y, q))
        q += 1
        if q > 4:
            q, y = 1, y + 1
    return out


def ingest_form4(store, user_agent, fetch_fn=None, cik_to_ticker=None) -> dict:
    """Quarterly SEC insider-transactions ingest. Skips quarters already fully
    recorded in the manifest, except the latest quarter, which is always
    refetched (late filings continue to trickle into the most recent quarter's
    dataset for weeks after quarter-end). Non-fatal per-quarter failures
    (e.g. a quarter's zip not yet published) are recorded, not raised.
    """
    fetch = fetch_fn or fetch_quarter
    if cik_to_ticker is None:
        from stocks_ml.data.edgar import load_cik_map
        ticker_to_cik = load_cik_map(user_agent)
        cik_to_ticker = {}
        for ticker, cik in ticker_to_cik.items():
            cik_to_ticker.setdefault(cik, ticker)  # collisions: keep first ticker seen

    existing = store.read("form4") if store.exists("form4") else None
    prior = store.manifest.get("form4", {})
    done = {tuple(q) for q in prior.get("quarters", [])}

    quarters = _quarters_through(_quarter_of(pd.Timestamp.today()))
    latest = quarters[-1] if quarters else None

    frames = [existing] if existing is not None and not existing.empty else []
    newly_done, failed = [], []
    for yq in quarters:
        if yq in done and yq != latest:
            continue
        try:
            raw = fetch(yq[0], yq[1], user_agent)
            frames.append(extract_transactions(raw["SUBMISSION"], raw["NONDERIV_TRANS"],
                                               cik_to_ticker))
            newly_done.append(yq)
        except Exception:
            failed.append(f"{yq[0]}Q{yq[1]}")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FORM4_COLS)
    if not df.empty:
        df = df.drop_duplicates().sort_values(["ticker", "filed", "trans_date"]).reset_index(drop=True)
    store.write("form4", df)

    all_done = sorted(done | set(newly_done))
    summary = {"quarters": [list(x) for x in all_done], "n_rows": int(len(df)),
               "failed_quarters": failed}
    store.set_manifest("form4", summary)
    return summary
