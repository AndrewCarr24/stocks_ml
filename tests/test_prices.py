import pandas as pd

from stocks_ml.data.prices import all_tickers_ever, ingest_prices
from stocks_ml.data.store import DataStore


def fake_fetch(tickers, start, end):
    dates = pd.bdate_range(start, end or "2024-01-10")
    rows = []
    for t in tickers:
        if t == "DEADCO":
            continue  # simulates an unfetchable delisted ticker
        for d in dates:
            rows.append({"date": d, "ticker": t, "open": 10.0, "high": 11.0,
                         "low": 9.0, "close": 10.5, "volume": 1000.0})
    return pd.DataFrame(rows)


def test_ingest_writes_prices_and_reports_failures(tmp_path):
    store = DataStore(tmp_path)
    summary = ingest_prices(store, ["AAA", "BBB", "DEADCO"], "2024-01-02", "2024-01-10",
                            fetch_fn=fake_fetch)
    assert summary["failed_tickers"] == ["DEADCO"]
    assert summary["n_ok"] == 2
    df = store.read("prices")
    assert set(df.ticker.unique()) == {"AAA", "BBB"}
    assert store.manifest["prices"]["failed_tickers"] == ["DEADCO"]


def test_ingest_is_incremental_and_dedupes(tmp_path):
    store = DataStore(tmp_path)
    ingest_prices(store, ["AAA"], "2024-01-02", "2024-01-10", fetch_fn=fake_fetch)
    n1 = len(store.read("prices"))
    ingest_prices(store, ["AAA"], "2024-01-02", "2024-01-15", fetch_fn=fake_fetch)
    df = store.read("prices")
    assert len(df) > n1
    assert not df.duplicated(subset=["date", "ticker"]).any()


def test_all_tickers_ever_includes_spy():
    mem = pd.DataFrame({"ticker": ["AAA", "OLD"], "start_date": pd.to_datetime(["2000-01-01"] * 2),
                        "end_date": [pd.NaT, pd.Timestamp("2010-01-01")], "sector": ["T", None]})
    assert all_tickers_ever(mem) == ["AAA", "OLD", "SPY"]
