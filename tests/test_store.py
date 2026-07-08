import pandas as pd
import pytest

from stocks_ml.data.store import DataStore


def test_write_read_roundtrip(tmp_path):
    store = DataStore(tmp_path)
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "ticker": ["AAPL"], "close": [190.0]})
    store.write("prices", df)
    assert store.exists("prices")
    out = store.read("prices")
    pd.testing.assert_frame_equal(out, df)


def test_read_missing_raises(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.read("prices")


def test_manifest_roundtrip(tmp_path):
    store = DataStore(tmp_path)
    assert store.manifest == {}
    store.set_manifest("prices", {"last_date": "2024-01-02", "failed_tickers": ["XYZ"]})
    fresh = DataStore(tmp_path)
    assert fresh.manifest["prices"]["failed_tickers"] == ["XYZ"]


def test_unknown_name_rejected(tmp_path):
    store = DataStore(tmp_path)
    with pytest.raises(ValueError):
        store.write("nonsense", pd.DataFrame())
