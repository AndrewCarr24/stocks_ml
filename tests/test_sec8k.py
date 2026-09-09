import pandas as pd

from stocks_ml.data.sec8k import extract_8k_submissions, ingest_sec8k
from stocks_ml.data.store import DataStore


def _recent(accessions, accepted, forms, items):
    n = len(accessions)
    return {
        "accessionNumber": accessions,
        "acceptanceDateTime": accepted,
        "filingDate": [str(pd.Timestamp(x).date()) for x in accepted],
        "form": forms,
        "items": items,
        "primaryDocument": [f"doc{i}.htm" for i in range(n)],
    }


def test_extract_8k_uses_acceptance_metadata_and_filters_forms():
    payload = {"filings": {"recent": _recent(
        ["a", "b", "c"],
        ["2024-01-02T21:01:00Z", "2024-02-03T13:00:00Z", "2024-03-04T13:00:00Z"],
        ["8-K", "8-K/A", "10-Q"], ["2.02,9.01", "2.02", ""],
    )}}
    out = extract_8k_submissions(payload, "AAA")
    assert list(out["accession"]) == ["a", "b"]
    assert str(out.iloc[0]["accepted"].tz) == "UTC"
    assert out.iloc[0]["items"] == "2.02,9.01"
    assert bool(out.iloc[1]["is_amendment"])


def test_extract_8k_includes_historical_submission_fragments():
    payload = {
        "filings": {
            "recent": _recent(["new"], ["2024-01-02T12:00:00Z"], ["8-K"], ["8.01"]),
            "files": [{"name": "old.json"}],
        }
    }
    old = _recent(["old"], ["2010-01-02T12:00:00Z"], ["8-K"], ["2.02"])
    out = extract_8k_submissions(payload, "AAA", fetch_file=lambda name: old)
    assert set(out["accession"]) == {"new", "old"}


def test_ingest_sec8k_records_failures_and_deduplicates(tmp_path):
    store = DataStore(tmp_path)
    payload = {"filings": {"recent": _recent(
        ["a"], ["2024-01-02T12:00:00Z"], ["8-K"], ["2.02"]
    )}}

    def fetch(cik):
        if cik == 2:
            raise RuntimeError("unavailable")
        return payload

    summary = ingest_sec8k(
        store, ["AAA", "BBB"], "ua", fetch_submissions_fn=fetch,
        fetch_file_fn=lambda name: {}, cik_map={"AAA": 1, "BBB": 2},
    )
    assert summary["failed_tickers"] == ["BBB"]
    assert summary["n_filings"] == 1
    assert list(store.read("sec8k")["accession"]) == ["a"]


def test_existing_accession_is_immutable_against_source_revision(tmp_path):
    store = DataStore(tmp_path)
    original = {"filings": {"recent": _recent(
        ["a"], ["2024-01-02T12:00:00Z"], ["8-K"], ["2.02"]
    )}}
    revised = {"filings": {"recent": _recent(
        ["a"], ["2024-01-02T12:00:00Z"], ["8-K"], ["8.01"]
    )}}
    kwargs = {"fetch_file_fn": lambda name: {}, "cik_map": {"AAA": 1}}
    ingest_sec8k(store, ["AAA"], "ua", fetch_submissions_fn=lambda cik: original,
                  **kwargs)
    ingest_sec8k(store, ["AAA"], "ua", fetch_submissions_fn=lambda cik: revised,
                  **kwargs)
    assert store.read("sec8k").iloc[0]["items"] == "2.02"


def test_dual_class_listings_keep_their_shared_accessions(tmp_path):
    """GOOGL/GOOG share every filing; deduping on the bare accession deleted
    the second listing's rows (GOOGL had 0 to GOOG's 114)."""
    rows = [{"ticker": t, "accession": "0001-24-000001", "accepted": "2024-01-05T09:00:00",
             "filed": "2024-01-05", "items": "2.02", "primary_document": "d.htm",
             "is_amendment": False} for t in ("GOOG", "GOOGL")]
    df = pd.DataFrame(rows)
    df["filed"] = pd.to_datetime(df["filed"])
    store = DataStore(tmp_path)
    store.write("sec8k", df)
    ingest_sec8k(store, ["GOOG", "GOOGL"], "ua",
                 fetch_submissions_fn=lambda cik: {"cik": cik, "filings": {"recent": {
                     "accessionNumber": [], "acceptanceDateTime": [], "filingDate": [],
                     "form": [], "items": [], "primaryDocument": []}}},
                 cik_map={"GOOG": 1, "GOOGL": 1})
    got = store.read("sec8k")
    assert sorted(got["ticker"]) == ["GOOG", "GOOGL"]
