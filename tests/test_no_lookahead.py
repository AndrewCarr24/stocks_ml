import numpy as np
import pandas as pd

from stocks_ml.backtest.simulator import run_backtest
from stocks_ml.backtest.strategies import EqualWeightTopK
from stocks_ml.data.store import DataStore
from stocks_ml.features.panel import build_panel, feature_cols
from stocks_ml.models.candidates import MomentumRank

CUTOFF = pd.Timestamp("2022-06-30")


def _corrupt_store(store, tmp_path, factor=3.0):
    """Copy of the store with all prices strictly after CUTOFF multiplied."""
    corrupted = DataStore(tmp_path / "corrupted")
    prices = store.read("prices").copy()
    after = prices["date"] > CUTOFF
    for col in ("open", "high", "low", "close"):
        prices.loc[after, col] *= factor
    corrupted.write("prices", prices)
    for name in ("membership", "fred", "edgar"):
        corrupted.write(name, store.read(name))
    return corrupted


def test_features_before_cutoff_unaffected_by_future(synthetic_store, tiny_cfg, tmp_path):
    panel_a = build_panel(synthetic_store, tiny_cfg)
    panel_b = build_panel(_corrupt_store(synthetic_store, tmp_path), tiny_cfg)
    fcols = feature_cols(panel_a)
    a = panel_a[panel_a.date <= CUTOFF].set_index(["date", "ticker"])[fcols].sort_index()
    b = panel_b[panel_b.date <= CUTOFF].set_index(["date", "ticker"])[fcols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10)


def test_weights_before_cutoff_unaffected_by_future(synthetic_store, tiny_cfg, tmp_path):
    def weights_through_cutoff(store):
        panel = build_panel(store, tiny_cfg)
        prices = store.read("prices")
        res = run_backtest(panel, prices, EqualWeightTopK(k=2), MomentumRank(), tiny_cfg)
        # exec happens the day after the rebalance; a margin of 7 days keeps us clear
        return res.weights[res.weights.index <= CUTOFF - pd.Timedelta(days=7)]

    w_a = weights_through_cutoff(synthetic_store)
    w_b = weights_through_cutoff(_corrupt_store(synthetic_store, tmp_path))
    pd.testing.assert_frame_equal(w_a, w_b, check_exact=False, rtol=1e-10)


def test_labels_use_strictly_future_prices(synthetic_store, tiny_cfg, tmp_path):
    """Corrupting future prices MUST change labels at dates just before the cutoff
    (labels look forward) while leaving features untouched (proven above)."""
    panel_a = build_panel(synthetic_store, tiny_cfg)
    panel_b = build_panel(_corrupt_store(synthetic_store, tmp_path), tiny_cfg)
    window = (panel_a.date <= CUTOFF) & (panel_a.date >= CUTOFF - pd.Timedelta(days=6))
    a = panel_a[window]["fwd_ret"].to_numpy()
    b = panel_b[panel_b.date.isin(panel_a[window].date.unique())]["fwd_ret"].to_numpy()
    assert not np.allclose(a, b, equal_nan=True)


def test_panel_respects_membership_windows(synthetic_store, tiny_cfg):
    panel = build_panel(synthetic_store, tiny_cfg)
    # from conftest: GGG leaves 2021-06-01, HHH joins 2021-06-01
    assert panel[(panel.date > "2021-06-15") & (panel.ticker == "GGG")].empty
    assert panel[(panel.date < "2021-05-15") & (panel.ticker == "HHH")].empty
