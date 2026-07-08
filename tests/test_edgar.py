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
