import numpy as np
import pandas as pd

from stocks_ml.features import generated as gen
from stocks_ml.features.bundle import FEATURES, FORMULAS


def test_bundle_is_forty_named_g_columns_over_known_inputs():
    assert FEATURES == list(FORMULAS) == [f"g_{i:02d}" for i in range(40)]
    assert len(set(FORMULAS.values())) == 40
    for f in FORMULAS.values():
        names = gen.formula_inputs(f)
        assert 1 <= len(names) <= 2 and all(n.startswith(("f_", "r_")) for n in names), f


def test_every_formula_evaluates_and_ranks_within_the_week():
    rng = np.random.default_rng(0)
    names = sorted({n for f in FORMULAS.values() for n in gen.formula_inputs(f)})
    dates = np.repeat(pd.to_datetime(["2024-01-05", "2024-01-12"]), 30)
    inputs = pd.DataFrame(rng.normal(size=(60, len(names))), columns=names)
    inputs.insert(0, "ticker", [f"T{i:02d}" for i in range(30)] * 2)
    inputs.insert(0, "date", dates)
    panel = inputs[["date", "ticker"]].copy()
    out = gen.add_generated(panel, inputs, FORMULAS)
    assert [c for c in out.columns if c.startswith("g_")] == FEATURES
    g = out[FEATURES]
    assert np.isfinite(g.to_numpy()).all() and g.abs().le(1.0).all().all()
    # ranked per week: each week's column is a permutation of the same grid (ties aside)
    for _, wk in out.groupby("date"):
        assert wk["g_00"].nunique() == 30
