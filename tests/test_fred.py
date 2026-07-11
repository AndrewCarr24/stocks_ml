import pandas as pd

from stocks_ml.data.fred import ingest_fred, load_fred_lagged
from stocks_ml.data.store import DataStore


def fake_fetch(series_id, user_agent):
    return pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-01-08"]),
                     name=series_id)


def test_ingest_and_lagged_load(tmp_path):
    store = DataStore(tmp_path)
    ingest_fred(store, {"TESTX": 3}, "ua", fetch_fn=fake_fetch)
    lagged = load_fred_lagged(store, {"TESTX": 3})
    # value 1.0 dated Jan 1 becomes visible Jan 4; before that it is NaN
    assert pd.isna(lagged.loc[pd.Timestamp("2024-01-03"), "TESTX"])
    assert lagged.loc[pd.Timestamp("2024-01-04"), "TESTX"] == 1.0
    # value 2.0 dated Jan 8 visible Jan 11; Jan 10 still shows 1.0 (ffill)
    assert lagged.loc[pd.Timestamp("2024-01-10"), "TESTX"] == 1.0
    assert lagged.loc[pd.Timestamp("2024-01-11"), "TESTX"] == 2.0
