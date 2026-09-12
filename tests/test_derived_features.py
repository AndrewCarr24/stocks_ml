"""features/derived.py: stage D's sector-relative and rank-momentum
features are point-in-time and read only the panel."""
import numpy as np
import pandas as pd
import pytest

from stocks_ml.features import derived


def toy_panel():
    dates = pd.date_range("2010-01-08", periods=6, freq="7D")
    rows = []
    for i, d in enumerate(dates):
        for tk in ("A", "B", "C", "D"):
            if tk == "D" and i < 2:
                continue                      # D joins the panel at week 2
            rows.append({"date": d, "ticker": tk, "f_x": {"A": 1.0, "B": 0.5, "C": -0.5, "D": -1.0}[tk] * (i + 1) / 6,
                         "f_mkt_mom_4w": float(i), "f_evt_8k_7d": 0.0})
    return pd.DataFrame(rows)


SMAP = {"A": "s1", "B": "s1", "C": "s2"}     # D has no sector


def test_ranked_features_excludes_within_week_constants_and_flags():
    pan = toy_panel()
    assert derived.ranked_features(pan) == ["f_x"]


def test_sector_relative_is_the_same_week_sector_median():
    pan = toy_panel()
    out = derived.sector_relative(pan, ["f_x"], SMAP)
    d0 = pan["date"].iloc[0]
    wk = pan[pan.date == d0].set_index("ticker")["f_x"]
    got = out.loc[pan.date == d0].set_index(pan.loc[pan.date == d0, "ticker"])["d_sec_f_x"]
    assert got["A"] == pytest.approx(wk["A"] - np.median([wk["A"], wk["B"]]))
    assert got["C"] == pytest.approx(0.0)                     # alone in its sector
    d2 = pan["date"].iloc[-1]
    wk2 = pan[pan.date == d2].set_index("ticker")["f_x"]
    got2 = out.loc[pan.date == d2].set_index(pan.loc[pan.date == d2, "ticker"])["d_sec_f_x"]
    assert got2["D"] == pytest.approx(wk2["D"] - wk2.median())   # no sector: the week's median


def test_rank_momentum_reads_only_the_same_ticker_lag_weeks_back():
    pan = toy_panel()
    out = derived.rank_momentum(pan, ["f_x"], lags=(1, 2))
    both = pd.concat([pan, out], axis=1)
    a = both[both.ticker == "A"].reset_index(drop=True)
    assert np.isnan(a.loc[0, "d_mom1_f_x"]) and np.isnan(a.loc[1, "d_mom2_f_x"])
    assert a.loc[3, "d_mom1_f_x"] == pytest.approx(a.loc[3, "f_x"] - a.loc[2, "f_x"])
    assert a.loc[3, "d_mom2_f_x"] == pytest.approx(a.loc[3, "f_x"] - a.loc[1, "f_x"])
    d = both[both.ticker == "D"].reset_index(drop=True)
    assert np.isnan(d.loc[0, "d_mom1_f_x"])                     # joined this week: no history
    assert d.loc[1, "d_mom1_f_x"] == pytest.approx(d.loc[1, "f_x"] - d.loc[0, "f_x"])


def test_rank_momentum_is_causal():
    """Perturbing a future week's value changes nothing before it."""
    pan = toy_panel()
    base = derived.rank_momentum(pan, ["f_x"], lags=(1,))
    pan2 = pan.copy()
    last = pan2.date == pan2.date.max()
    pan2.loc[last, "f_x"] = 99.0
    alt = derived.rank_momentum(pan2, ["f_x"], lags=(1,))
    earlier = (pan.date < pan.date.max()).to_numpy()
    pd.testing.assert_frame_equal(base[earlier], alt[earlier])


def test_add_derived_names_and_shape():
    pan = toy_panel()
    out, new = derived.add_derived(pan, SMAP, lags=(1, 2))
    assert new == ["d_sec_f_x", "d_mom1_f_x", "d_mom2_f_x"]
    assert len(out) == len(pan) and all(c in out.columns for c in new)
    # feature_cols still sees only the f_ columns: derived ones enter by name
    from stocks_ml.features.panel import feature_cols
    assert feature_cols(out) == feature_cols(pan)
