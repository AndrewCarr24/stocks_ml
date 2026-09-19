"""Sector-relative feature ranks (2026-09-19): the input in the label's frame."""
import numpy as np
import pandas as pd

from stocks_ml.features.panel import SR_PREFIX, sector_relative_ranks


def test_sector_relative_ranks_rerank_within_week_and_sector():
    d = pd.Timestamp("2010-01-08")
    panel = pd.DataFrame({"date": [d] * 6, "ticker": list("ABCDEF"),
                          "f_x": [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]})       # the week's ranks
    sector = pd.Series(["U", "U", "U", "T", "T", None])                   # F: sector unknown
    out = sector_relative_ranks(panel, sector, ["f_x"])
    c = SR_PREFIX + "f_x"
    assert list(out.columns) == [c]
    assert np.allclose(out[c].tolist()[:3], [-1 / 3, 1 / 3, 1.0])            # the utilities re-ranked among themselves
    assert out[c].tolist()[3:5] == [0.0, 1.0]                              # the two techs
    assert out[c].iloc[5] == 1.0                                           # unknown sector: the week's rank kept
    two = pd.concat([panel, panel.assign(date=d + pd.Timedelta(days=7))], ignore_index=True)
    out2 = sector_relative_ranks(two, pd.concat([sector, sector], ignore_index=True), ["f_x"])
    assert out2[c].tolist()[:6] == out2[c].tolist()[6:]                    # per week


def test_append_sector_relative_writes_derived_columns_once(tmp_path):
    from stocks_ml.data.store import DataStore
    from stocks_ml.data.world import append_sector_relative
    d = pd.Timestamp("2010-01-08")
    pd.DataFrame({"date": [d] * 4, "ticker": list("ABCD"), "f_mom_4w": [-1.0, -0.3, 0.3, 1.0], "f_sec_x": [0.0] * 4,
                  "label_4w": [0.0] * 4}).to_parquet(tmp_path / "panel_sf.parquet", index=False)
    DataStore(tmp_path).write("membership", pd.DataFrame({"ticker": list("ABCD"), "start_date": [d] * 4, "end_date": [pd.NaT] * 4,
                                                          "sector": ["U", "U", "T", "T"]}))
    added = append_sector_relative(tmp_path, log=lambda m: None)
    assert added == [SR_PREFIX + "f_mom_4w"]                              # admitted features only
    pan = pd.read_parquet(tmp_path / "panel_sf.parquet")
    assert pan[SR_PREFIX + "f_mom_4w"].tolist() == [0.0, 1.0, 0.0, 1.0]     # pct ranks of a pair: 1/2 and 1 -> 0 and 1
    assert append_sector_relative(tmp_path, log=lambda m: None) == []      # idempotent
