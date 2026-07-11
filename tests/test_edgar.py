import pandas as pd

from stocks_ml.data.edgar import extract_facts, ingest_edgar
from stocks_ml.data.store import DataStore

CF_JSON = {
    "facts": {
        "us-gaap": {
            "NetIncomeLoss": {"units": {"USD": [
                {"start": "2022-01-01", "end": "2022-12-31", "filed": "2023-02-15",
                 "val": 100.0, "form": "10-K"},
                {"start": "2023-01-01", "end": "2023-03-31", "filed": "2023-05-01",
                 "val": 30.0, "form": "10-Q"},
            ]}},
            "Assets": {"units": {"USD": [
                {"end": "2022-12-31", "filed": "2023-02-15", "val": 1000.0, "form": "10-K"},
            ]}},
            "CommonStockSharesOutstanding": {"units": {"shares": [
                {"end": "2022-12-31", "filed": "2023-02-15", "val": 50.0, "form": "10-K"},
            ]}},
        }
    }
}


def _cf_json_filed(days_ago: int) -> dict:
    """A minimal companyfacts payload whose single fact was filed `days_ago` days ago."""
    filed = (pd.Timestamp.today() - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d")
    return {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        {"start": "2022-01-01", "end": "2022-12-31", "filed": filed, "val": 100.0, "form": "10-K"},
    ]}}}}}


def test_extract_facts_shapes_and_fallbacks():
    concept_map = {"net_income": ["NetIncomeLoss"], "assets": ["Assets"],
                   "shares": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
                   "gross_profit": ["GrossProfit"]}  # missing tag -> simply absent
    df = extract_facts(CF_JSON, "AAA", concept_map)
    assert set(df.concept.unique()) == {"net_income", "assets", "shares"}
    ni = df[df.concept == "net_income"]
    assert len(ni) == 2
    assert pd.notna(ni.iloc[0]["start"])            # duration concept keeps start
    assert df[df.concept == "assets"].iloc[0]["val"] == 1000.0
    assert pd.isna(df[df.concept == "assets"].iloc[0]["start"])  # instant concept


def test_ingest_edgar_records_failures(tmp_path):
    store = DataStore(tmp_path)

    def fake_facts(cik, user_agent):
        if cik == 2:
            raise RuntimeError("no facts")
        return CF_JSON

    summary = ingest_edgar(store, ["AAA", "BBB"], {"net_income": ["NetIncomeLoss"]},
                           "ua", fetch_facts_fn=fake_facts, cik_map={"AAA": 1, "BBB": 2})
    assert summary["failed_tickers"] == ["BBB"]
    df = store.read("edgar")
    assert set(df.ticker.unique()) == {"AAA"}


def test_total_failure_run_does_not_corrupt_store(tmp_path):
    store = DataStore(tmp_path)

    def failing_facts(cik, user_agent):
        raise RuntimeError("403")

    s1 = ingest_edgar(store, ["AAA"], {"net_income": ["NetIncomeLoss"]}, "ua",
                      fetch_facts_fn=failing_facts, cik_map={"AAA": 1})
    assert s1["failed_tickers"] == ["AAA"]
    # second run must not crash and must succeed for AAA
    s2 = ingest_edgar(store, ["AAA"], {"net_income": ["NetIncomeLoss"]}, "ua",
                      fetch_facts_fn=lambda cik, ua: CF_JSON, cik_map={"AAA": 1})
    assert s2["failed_tickers"] == []
    assert "AAA" in set(store.read("edgar").ticker.unique())


def test_stale_ticker_is_refetched(tmp_path):
    store = DataStore(tmp_path)
    concept_map = {"net_income": ["NetIncomeLoss"]}
    # first ingest: filing is ~200 days old -> stale on the next run
    stale_json = _cf_json_filed(200)
    s1 = ingest_edgar(store, ["AAA"], concept_map, "ua",
                      fetch_facts_fn=lambda cik, ua: stale_json, cik_map={"AAA": 1})
    assert s1["failed_tickers"] == []

    calls = []

    def recording_fetch(cik, ua):
        calls.append(cik)
        return stale_json

    ingest_edgar(store, ["AAA"], concept_map, "ua",
                fetch_facts_fn=recording_fetch, cik_map={"AAA": 1})
    assert calls == [1]  # stale ticker was refetched


def test_fresh_ticker_is_not_refetched(tmp_path):
    store = DataStore(tmp_path)
    concept_map = {"net_income": ["NetIncomeLoss"]}
    # first ingest: filing is ~10 days old -> still fresh on the next run
    fresh_json = _cf_json_filed(10)
    s1 = ingest_edgar(store, ["AAA"], concept_map, "ua",
                      fetch_facts_fn=lambda cik, ua: fresh_json, cik_map={"AAA": 1})
    assert s1["failed_tickers"] == []

    calls = []

    def recording_fetch(cik, ua):
        calls.append(cik)
        return fresh_json

    ingest_edgar(store, ["AAA"], concept_map, "ua",
                fetch_facts_fn=recording_fetch, cik_map={"AAA": 1})
    assert calls == []  # fresh ticker was skipped


def test_fallback_tag_used_when_first_tag_has_no_usable_rows():
    cf = {"facts": {"us-gaap": {
        "TagA": {"units": {"USD": [{"end": "2022-12-31", "val": 1.0, "form": "10-K"}]}},  # no 'filed' -> unusable
        "TagB": {"units": {"USD": [{"end": "2022-12-31", "filed": "2023-02-15", "val": 2.0, "form": "10-K"}]}},
    }}}
    df = extract_facts(cf, "AAA", {"net_income": ["TagA", "TagB"]})
    assert len(df) == 1
    assert df.iloc[0]["val"] == 2.0
