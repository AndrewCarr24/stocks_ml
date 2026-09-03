import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from stocks_ml.data.store import DataStore
from stocks_ml.features.panel import all_feature_cols, build_panel
from stocks_ml.models.walk import walk_forward_predictions

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
    # Include research-only columns: rejection from the production model matrix
    # must not exempt a generated feature from the no-lookahead contract.
    fcols = all_feature_cols(panel_a)
    a = panel_a[panel_a.date <= CUTOFF].set_index(["date", "ticker"])[fcols].sort_index()
    b = panel_b[panel_b.date <= CUTOFF].set_index(["date", "ticker"])[fcols].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10)


class TrainedMean(BaseEstimator, RegressorMixin):
    """Predicts each row's momentum plus the mean training label: sensitive to
    both the features it scores and every label it was trained on, so a
    leak through either channel shows up in the predictions."""

    def fit(self, X, y):
        self.offset_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return X["f_mom_26w"].to_numpy() + self.offset_


def test_predictions_before_cutoff_unaffected_by_future(synthetic_store, tiny_cfg, tmp_path):
    def preds_through_cutoff(store):
        wf = walk_forward_predictions(build_panel(store, tiny_cfg), TrainedMean(), tiny_cfg)
        # the label looks a week ahead plus the purge; a margin of 7 days on top
        # of the purge keeps every training label clear of the corruption
        last = CUTOFF - pd.Timedelta(days=tiny_cfg.purge_days + 7)
        return {t: p for t, p in wf.preds.items() if t <= last}

    p_a = preds_through_cutoff(synthetic_store)
    p_b = preds_through_cutoff(_corrupt_store(synthetic_store, tmp_path))
    assert len(p_a) >= 10 and sorted(p_a) == sorted(p_b)
    for t in p_a:
        pd.testing.assert_series_equal(p_a[t], p_b[t], check_exact=False, rtol=1e-10)


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
