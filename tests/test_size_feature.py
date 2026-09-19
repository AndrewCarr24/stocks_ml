"""The Sharadar market cap as an opt-in feature (2026-09-18)."""
import numpy as np
import pandas as pd

from stocks_ml.features.panel import PENDING_ABLATION_FEATURES, feature_cols
from stocks_ml.features.sharadar_fundamentals import SF_RAW_COLS, sharadar_fundamental_features


def test_sharadar_size_is_computed_but_only_opt_in():
    base = pd.DataFrame({"date": pd.to_datetime(["2010-06-04"] * 2), "ticker": ["A", "B"]})
    fund = pd.DataFrame([
        {"ticker": "A", "dimension": "ARQ", "date": pd.Timestamp("2010-05-01"), "reportperiod": pd.Timestamp("2010-03-31"),
         "bvps": 5.0, "currentratio": 1.0, "de": 0.5, "debt": 1.0, "equity": 10.0, "assets": 20.0, "sharesbas": 1000.0,
         "netinc": 1.0, "revenue": 10.0},
        {"ticker": "A", "dimension": "ART", "date": pd.Timestamp("2010-05-01"), "reportperiod": pd.Timestamp("2010-03-31"),
         "epsusd": 1.0, "revenue": 10.0, "netinc": 1.0, "ebitda": 2.0, "fcf": 1.0, "gp": 5.0, "ebitdamargin": 0.2},
    ])
    close = pd.Series([10.0, 20.0], index=base.index)
    out = sharadar_fundamental_features(fund, base, close)
    assert "f_sf_log_mktcap" in SF_RAW_COLS
    assert out["f_sf_log_mktcap"].iloc[0] == np.log(10.0 * 1000.0) and np.isnan(out["f_sf_log_mktcap"].iloc[1])
    assert "f_sf_log_mktcap" in PENDING_ABLATION_FEATURES
    pan = pd.DataFrame({"date": [pd.Timestamp("2010-06-04")], "ticker": ["A"], "f_mom_4w": [0.1], "f_sf_log_mktcap": [0.2]})
    assert feature_cols(pan) == ["f_mom_4w"]             # the champion's matrix is unchanged
