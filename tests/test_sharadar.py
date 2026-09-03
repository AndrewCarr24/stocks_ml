import pandas as pd
import pytest

from stocks_ml.data.sharadar import PAGE_LIMIT, fetch_table


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

