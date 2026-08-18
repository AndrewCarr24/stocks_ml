import pandas as pd
import pytest

from stocks_ml.data.sharadar import (PAGE_LIMIT, fetch_prices,
                                     fetch_sp500_membership, fetch_table,
                                     ingest_sharadar)


def _rec(ticker="AAPL", date="2026-01-02", closeunadj=210.0):
    return {"ticker": ticker, "date": date, "open": 10, "high": 11, "low": 9,
            "close": 10.5, "volume": 100, "closeadj": 10.4,
            "closeunadj": closeunadj, "lastupdated": "2026-01-03"}


def test_fetch_table_follows_offset_pagination():
    pages = [{"count": PAGE_LIMIT, "data": [_rec(date="2026-01-02")] * PAGE_LIMIT},
             {"count": 1, "data": [_rec(date="2026-01-03")]}]
    calls = []

    def fake(url, params, headers):
        calls.append(params["offset"])
        assert headers == {"x-api-key": "k"}          # key in header, never URL
        return pages[len(calls) - 1]

    out = fetch_table("stocks", "k", fetch_fn=fake)
    assert len(out) == PAGE_LIMIT + 1
    assert calls == [0, PAGE_LIMIT]
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_fetch_table_surfaces_vendor_errors():
    def fake(url, params, headers):
        raise RuntimeError("Sharadar stocks: HTTP 403: Exceeds free tier")

    with pytest.raises(RuntimeError, match="Exceeds free tier"):
        fetch_table("stocks", "k", fetch_fn=fake)


def test_fetch_prices_keeps_all_three_close_columns():
    def fake(url, params, headers):
        assert "data/stocks" in url
        assert params["from"] == "2026-01-01"
        return {"count": 1, "data": [_rec()]}

    out = fetch_prices(["AAPL"], "2026-01-01", "k", fetch_fn=fake)
    # split-adjusted close, fully-adjusted closeadj, as-traded closeunadj:
    # the disambiguation the yfinance pipeline cannot provide
    assert {"close", "closeadj", "closeunadj"} <= set(out.columns)
    assert out.loc[0, "closeunadj"] == 210.0


def test_sp500_membership_maps_actions():
    def fake(url, params, headers):
        return {"count": 2, "data": [
            {"date": "2026-08-05", "action": "added", "ticker": "FERG",
             "name": "Ferguson"},
            {"date": "2026-08-05", "action": "REMOVED", "ticker": "EA",
             "name": "Electronic Arts"}]}

    out = fetch_sp500_membership("k", fetch_fn=fake)
    assert list(out["action"]) == ["added", "removed"]      # normalized


def test_ingest_writes_separate_store_keys(tmp_path, synthetic_store):
    def fake(url, params, headers):
        if "data/stocks" in url:
            return {"count": 1, "data": [_rec()]}
        if "data/tickers" in url:
            return {"count": 2, "data": [
                {"ticker": "AAPL", "name": "Apple", "exchange": "NASDAQ",
                 "isdelisted": "N", "category": "Domestic", "sector": "Tech",
                 "industry": "Hw", "firstpricedate": "1980-12-12",
                 "lastpricedate": "2026-08-15"},
                {"ticker": "LEH", "name": "Lehman", "exchange": "NYSE",
                 "isdelisted": "Y", "category": "Domestic", "sector": "Fin",
                 "industry": "Bank", "firstpricedate": "1994-05-02",
                 "lastpricedate": "2008-09-17"}]}
        return {"count": 1, "data": [{"date": "2026-08-05", "action": "added",
                                      "ticker": "FERG", "name": "Ferguson"}]}

    (tmp_path / ".sharadar_key").write_text("k")
    res = ingest_sharadar(synthetic_store, ["AAPL"], "2026-01-01",
                          data_dir=tmp_path, fetch_fn=fake)
    assert res["prices_rows"] == 1 and res["delisted_in_meta"] == 1
    assert res["sp500_events"] == 1
    assert synthetic_store.exists("sharadar_prices")
    assert synthetic_store.exists("sharadar_sp500")
