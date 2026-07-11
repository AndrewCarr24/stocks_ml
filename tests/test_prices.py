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


def test_new_ticker_on_second_run_gets_full_history(tmp_path):
    calls = []

    def recording_fetch(tickers, start, end):
        calls.append((tuple(sorted(tickers)), str(pd.Timestamp(start).date())))
        return fake_fetch(tickers, start, end)

    store = DataStore(tmp_path)
    ingest_prices(store, ["AAA"], "2024-01-02", "2024-01-10", fetch_fn=recording_fetch)
    ingest_prices(store, ["AAA", "NEWCO"], "2024-01-02", "2024-01-10", fetch_fn=recording_fetch)
    # NEWCO must be fetched from the original start, not from AAA's max date
    newco_calls = [c for c in calls if "NEWCO" in c[0]]
    assert newco_calls and newco_calls[0][1] == "2024-01-02"
    df = store.read("prices")
    assert df[df.ticker == "NEWCO"].date.min() == pd.Timestamp("2024-01-02")


def test_rerun_with_no_new_rows_reports_no_false_failures(tmp_path):
    store = DataStore(tmp_path)
    ingest_prices(store, ["AAA"], "2024-01-02", "2024-01-10", fetch_fn=fake_fetch)

    def empty_fetch(tickers, start, end):
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    summary = ingest_prices(store, ["AAA"], "2024-01-02", "2024-01-10", fetch_fn=empty_fetch)
    assert summary["failed_tickers"] == []
    assert summary["n_ok"] == 1


def test_all_tickers_ever_includes_spy():
    mem = pd.DataFrame({"ticker": ["AAA", "OLD"], "start_date": pd.to_datetime(["2000-01-01"] * 2),
                        "end_date": [pd.NaT, pd.Timestamp("2010-01-01")], "sector": ["T", None]})
    assert all_tickers_ever(mem) == ["AAA", "OLD", "SPY"]


def test_drop_corrupt_series_kills_repeat_jumpers_spares_single_bounce():
    from stocks_ml.data.prices import drop_corrupt_series

    dates = pd.bdate_range("2020-01-01", periods=10)
    def series(t, closes):
        return pd.DataFrame({"date": dates[:len(closes)], "ticker": t,
                             "open": closes, "high": closes, "low": closes,
                             "close": closes, "volume": 1e6})
    corrupt = series("BAD", [10, 20, 20, 20, 40, 40, 40, 40, 40, 40])   # two doublings
    bounce = series("AIG", [10, 4, 4, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5])  # one crash -60%, one +62% -> neither trips 1.9x twice
    clean = series("OK", [10 + 0.1 * i for i in range(10)])
    prices = pd.concat([corrupt, bounce, clean], ignore_index=True)
    out, dropped = drop_corrupt_series(prices)
    assert dropped == ["BAD"]
    assert set(out.ticker.unique()) == {"AIG", "OK"}
