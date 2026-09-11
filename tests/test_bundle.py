import numpy as np
import pandas as pd

from stocks_ml.features import generated as gen
from stocks_ml.features.bundle import FEATURES, FORMULAS


def test_bundle_is_retired_and_empty():
    # 2026-09-11: the split-leak bundle retired; the champion is the clean line
    assert FORMULAS == {} and FEATURES == []


def test_add_generated_with_no_formulas_is_a_no_op():
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.to_datetime(["2024-01-05", "2024-01-12"]), 30)
    inputs = pd.DataFrame(rng.normal(size=(60, 2)), columns=["f_a", "r_close"])
    inputs.insert(0, "ticker", [f"T{i:02d}" for i in range(30)] * 2)
    inputs.insert(0, "date", dates)
    panel = inputs[["date", "ticker"]].copy()
    out = gen.add_generated(panel, inputs, FORMULAS)
    assert [c for c in out.columns if c.startswith(gen.PREFIX)] == []
    pd.testing.assert_frame_equal(out, panel)


def test_a_formula_bundle_still_evaluates_and_ranks_within_the_week():
    # the plumbing stays for a bundle that passes the gate later
    formulas = {"g_00": "(r_close*f_a)", "g_01": "abs(f_a)"}
    rng = np.random.default_rng(0)
    dates = np.repeat(pd.to_datetime(["2024-01-05", "2024-01-12"]), 30)
    inputs = pd.DataFrame(rng.normal(size=(60, 2)), columns=["f_a", "r_close"])
    inputs.insert(0, "ticker", [f"T{i:02d}" for i in range(30)] * 2)
    inputs.insert(0, "date", dates)
    panel = inputs[["date", "ticker"]].copy()
    out = gen.add_generated(panel, inputs, formulas)
    assert [c for c in out.columns if c.startswith("g_")] == list(formulas)
    g = out[list(formulas)]
    assert np.isfinite(g.to_numpy()).all() and g.abs().le(1.0).all().all()
    for _, wk in out.groupby("date"):
        assert wk["g_00"].nunique() == 30
