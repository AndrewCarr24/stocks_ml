import pandas as pd
import pytest

from stocks_ml.data.sharadar import (fetch_datatable, fetch_prices,
                                     fetch_sp500_membership, ingest_sharadar)


def _page(rows, cols, cursor=None):
    return {"datatable": {"data": rows,
                          "columns": [{"name": c, "type": "t"} for c in cols]},
            "meta": {"next_cursor_id": cursor}}


SEP_COLS = ["ticker", "date", "open", "high", "low", "close", "volume",
            "closeadj", "closeunadj", "lastupdated"]


def test_fetch_datatable_follows_cursor_pagination():
    pages = [_page([["AAPL", "2026-01-02", 1, 2, 0.5, 1.5, 100, 1.4, 150, "2026-01-03"]],
                   SEP_COLS, cursor="abc"),
             _page([["AAPL", "2026-01-03", 1, 2, 0.5, 1.6, 110, 1.5, 160, "2026-01-04"]],
                   SEP_COLS, cursor=None)]
    calls = []

    def fake(url, params):
        calls.append(params.get("qopts.cursor_id"))
        return pages[len(calls) - 1]

    out = fetch_datatable("SEP", "k", fetch_fn=fake)
    assert len(out) == 2
    assert calls == [None, "abc"]                 # cursor threaded through
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_fetch_datatable_surfaces_vendor_errors():
    def fake(url, params):
        raise RuntimeError("Sharadar API error on SEP: account disabled")

    with pytest.raises(RuntimeError, match="account disabled"):
        fetch_datatable("SEP", "k", fetch_fn=fake)


def test_fetch_prices_keeps_all_three_close_columns():
    def fake(url, params):
        return _page([["AAPL", "2026-01-02", 10, 11, 9, 10.5, 100, 10.4, 210.0, "x"]],
                     SEP_COLS)

    out = fetch_prices(["AAPL"], "2026-01-01", "k", fetch_fn=fake)
    # split-adjusted close, fully-adjusted closeadj, as-traded closeunadj:
    # the disambiguation the yfinance pipeline cannot provide
    assert {"close", "closeadj", "closeunadj"} <= set(out.columns)
    assert out.loc[0, "closeunadj"] == 210.0


def test_sp500_membership_maps_actions():
    cols = ["date", "action", "ticker", "name"]
    def fake(url, params):
        return _page([["2026-08-05", "added", "FERG", "Ferguson"],
                      ["2026-08-05", "REMOVED", "EA", "Electronic Arts"]], cols)

    out = fetch_sp500_membership("k", fetch_fn=fake)
    assert list(out["action"]) == ["added", "removed"]      # normalized


def test_ingest_writes_separate_store_keys(tmp_path, synthetic_store):
    def fake(url, params):
        if "SEP.json" in url and "ticker" in params and params.get("table") is None:
            return _page([["AAPL", "2026-01-02", 10, 11, 9, 10.5, 100, 10.4, 210.0, "x"]],
                         SEP_COLS)
        if "TICKERS" in url:
            return _page([["AAPL", "Apple", "NASDAQ", "N", "Domestic", "Tech",
                           "Hw", "1980-12-12", "2026-08-15"],
                          ["LEH", "Lehman", "NYSE", "Y", "Domestic", "Fin",
                           "Bank", "1994-05-02", "2008-09-17"]],
                         ["ticker", "name", "exchange", "isdelisted", "category",
                          "sector", "industry", "firstpricedate", "lastpricedate"])
        return _page([["2026-08-05", "added", "FERG", "Ferguson"]],
                     ["date", "action", "ticker", "name"])

    (tmp_path / ".sharadar_key").write_text("k")
    res = ingest_sharadar(synthetic_store, ["AAPL"], "2026-01-01",
                          data_dir=tmp_path, fetch_fn=fake)
    assert res["prices_rows"] == 1 and res["delisted_in_meta"] == 1
    assert res["sp500_events"] == 1
    assert synthetic_store.exists("sharadar_prices")
    assert synthetic_store.exists("sharadar_sp500")
