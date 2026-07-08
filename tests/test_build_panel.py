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
