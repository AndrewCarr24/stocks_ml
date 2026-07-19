import pandas as pd

from stocks_ml.features.panel import build_panel, feature_cols
from stocks_ml.features.ranking import RANK_EXEMPT_PREFIXES, rank_normalize


def test_rank_normalize_maps_to_unit_interval():
    df = pd.DataFrame({"date": [pd.Timestamp("2022-01-07")] * 4,
                       "f_x": [10.0, 20.0, 30.0, float("nan")]})
    out = rank_normalize(df, ["f_x"])
    vals = out["f_x"].dropna().tolist()
    assert min(vals) > -1 and max(vals) <= 1
    assert vals == sorted(vals)          # order preserved
    assert out["f_x"].isna().sum() == 1  # NaN stays NaN


def test_build_panel_shape_and_membership(synthetic_store, tiny_cfg):
    panel = build_panel(synthetic_store, tiny_cfg)
    assert {"date", "ticker", "fwd_ret", "label", "aux_vol"} <= set(panel.columns)
    fcols = feature_cols(panel)
    assert any(c.startswith("f_mom_") for c in fcols)
    assert any(c.startswith("f_sec_") for c in fcols)
    assert any(c.startswith("f_macro_") for c in fcols)
    assert any(c == "f_roe" for c in fcols)
    # membership: GGG left 2021-06-01, HHH joined then
    assert "GGG" not in panel[panel.date > "2021-06-15"].ticker.values
    assert "HHH" not in panel[panel.date < "2021-05-15"].ticker.values
    # ranked features bounded
    ranked = [c for c in fcols if not c.startswith(RANK_EXEMPT_PREFIXES)]
    sub = panel[ranked].stack()
    assert sub.min() > -1.0001 and sub.max() <= 1.0001
    # aux_vol is raw (annualized-vol scale, not the (-1,1] rank scale)
    assert panel["aux_vol"].dropna().max() > 0.05
    # panel persisted
    assert synthetic_store.exists("panel")


def test_build_panel_v2_features_present_and_bounded(synthetic_store, tiny_cfg):
    panel = build_panel(synthetic_store, tiny_cfg)
    fcols = feature_cols(panel)
    expected_new = {"f_evt_filed_5d", "f_days_since_filing", "f_pead",
                    "f_overnight_4w", "f_intraday_4w", "f_beta_60d", "f_idio_vol_60d",
                    "f_mom_4w_sect", "f_mom_12w_sect", "f_mkt_dispersion"}
    assert expected_new <= set(fcols)

    # binary filing flag is rank-exempt: never rank-mangled, only {0.0, 1.0}
    assert "f_evt_" in RANK_EXEMPT_PREFIXES
    vals = set(panel["f_evt_filed_5d"].dropna().unique().tolist())
    assert vals <= {0.0, 1.0}

    # sector-relative momentum is rank-normalized like other stock features
    sect = panel["f_mom_4w_sect"].dropna()
    assert sect.min() > -1.0001 and sect.max() <= 1.0001
    sect12 = panel["f_mom_12w_sect"].dropna()
    assert sect12.min() > -1.0001 and sect12.max() <= 1.0001


def test_build_panel_insider_and_short_features_present_and_exercised(synthetic_store, tiny_cfg):
    panel = build_panel(synthetic_store, tiny_cfg)
    fcols = feature_cols(panel)
    expected_new = {"f_insider_net_13w", "f_insider_buyers_13w", "f_evt_insider_buy_2w",
                    "f_short_ratio", "f_short_dtc"}
    assert expected_new <= set(fcols)

    # rank-exempt evt flag stays a clean {0.0, 1.0} indicator
    vals = set(panel["f_evt_insider_buy_2w"].dropna().unique().tolist())
    assert vals <= {0.0, 1.0}

    # conftest's synthetic form4/shortint fixtures (dated 2022-08/09, after
    # tiny_cfg.backtest_start) must actually be exercised end-to-end: some row
    # shows a non-default, non-NaN value for each new feature.
    assert (panel["f_insider_net_13w"] != 0.0).any()
    assert (panel["f_insider_buyers_13w"] != 0.0).any()
    assert (panel["f_evt_insider_buy_2w"] == 1.0).any()
    assert panel["f_short_ratio"].notna().any()
    assert panel["f_short_dtc"].notna().any()

    # ranked (non-evt) insider/short features are bounded like other ranked features
    for c in ("f_insider_net_13w", "f_insider_buyers_13w", "f_short_ratio", "f_short_dtc"):
        sub = panel[c].dropna()
        assert sub.min() > -1.0001 and sub.max() <= 1.0001


def test_build_panel_missing_form4_and_shortint_datasets_is_harmless(synthetic_store, tiny_cfg):
    """A store that never ingested the (optional) insider/short-interest
    datasets must not break build_panel -- features fall back to defaults."""
    (synthetic_store.root / "form4.parquet").unlink()
    (synthetic_store.root / "shortint.parquet").unlink()
    panel = build_panel(synthetic_store, tiny_cfg)
    # no insider activity anywhere -> every ticker ties at the same raw 0.0, so
    # after cross-sectional rank-normalization they all collapse to one constant
    # (not necessarily 0.0 itself -- that depends on the tie-break formula).
    assert panel["f_insider_net_13w"].nunique() == 1
    assert panel["f_insider_buyers_13w"].nunique() == 1
    assert (panel["f_evt_insider_buy_2w"] == 0.0).all()  # rank-exempt: stays literal 0.0
    assert panel["f_short_ratio"].isna().all()
    assert panel["f_short_dtc"].isna().all()


def test_build_panel_drops_corrupt_tickers(synthetic_store, tiny_cfg):
    import numpy as np
    prices = synthetic_store.read("prices")
    bad = prices[prices.ticker == "AAA"].copy()
    bad["ticker"] = "BADCO"
    doubles = bad.index[::100]
    for col in ("open", "high", "low", "close"):
        bad.loc[doubles, col] = bad.loc[doubles, col] * 4  # repeated 4x spikes
    synthetic_store.write("prices", pd.concat([prices, bad], ignore_index=True))
    mem = synthetic_store.read("membership")
    mem = pd.concat([mem, pd.DataFrame({"ticker": ["BADCO"], "start_date": [pd.Timestamp("2015-01-01")],
                                        "end_date": [pd.NaT], "sector": ["Tech"]})], ignore_index=True)
    synthetic_store.write("membership", mem)
    panel = build_panel(synthetic_store, tiny_cfg)
    assert "BADCO" not in panel.ticker.values
    assert "BADCO" in synthetic_store.manifest["corrupt_tickers"]
