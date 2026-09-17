"""stocks_ml.leak_audit: the identity gate on a toy world, the verdict line."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stocks_ml.leak_audit as la
from stocks_ml.features.ranking import rank_normalize


def _toy(shift_stored=0.0):
    """Three members at one week; bvps filed a month earlier; C is a future
    splitter (closeunadj 4x close_split). The stored panel column is the
    rank of bvps / close_split — the builder's rule — unless shifted."""
    wk = pd.Timestamp("2010-06-04")
    days = pd.date_range("2010-05-24", "2010-06-04", freq="B")
    px = []
    for tk, cs, unadj in (("A", 10.0, 10.0), ("B", 20.0, 20.0), ("C", 5.0, 20.0)):
        for d in days:
            px.append({"date": d, "ticker": tk, "close": cs, "close_split": cs, "closeunadj": unadj})
    prices = pd.DataFrame(px)
    fund = pd.DataFrame([{"ticker": tk, "dimension": "ARQ", "date": pd.Timestamp("2010-05-01"),
                          "reportperiod": pd.Timestamp("2010-03-31"), "bvps": b}
                         for tk, b in (("A", 5.0), ("B", 5.0), ("C", 10.0))])
    ratio = pd.DataFrame({"date": [wk] * 3, "ticker": list("ABC"), "f_sf_book_to_market": [0.5, 0.25, 2.0]})
    pan = rank_normalize(ratio, ["f_sf_book_to_market"])
    pan["f_sf_book_to_market"] += shift_stored
    return SimpleNamespace(pan=pan), prices, fund, wk


def test_identity_check_reproduces_the_stored_ranks_and_names_the_splitters():
    ctx, prices, fund, wk = _toy()
    got = la.identity_check(ctx, prices, fund, wk)
    assert got["ok"] and got["members"] == 3 and got["max_abs_rank_diff"] == 0.0
    assert list(got["top_splitters"])[0] == "C"
    c = got["top_splitters"]["C"]
    assert c["future_split_factor"] == 4.0 and c["bvps_asof"] == 10.0 and c["close_split"] == 5.0
    assert c["stored_rank"] == c["recomputed_rank"] == 1.0


def test_identity_check_fails_when_the_stored_column_drifts():
    ctx, prices, fund, wk = _toy(shift_stored=1e-6)
    got = la.identity_check(ctx, prices, fund, wk)
    assert not got["ok"] and got["max_abs_rank_diff"] == pytest.approx(1e-6)
    ctx.pan = ctx.pan.drop(columns=["f_sf_book_to_market"])
    assert la.identity_check(ctx, prices, fund, wk)["ok"] is False


def test_nw_t_and_the_leak_line():
    x = pd.Series(np.full(100, 0.5))
    assert np.isnan(la.nw_t(pd.Series([1.0] * 5)))
    assert la.nw_t(x + np.r_[np.ones(50), -np.ones(50)] * 1e-9) > 1e6
    seg = {"IDENTITY": "PASS", "spearman_score_vs_factor": -0.017, "spearman_t": -0.9, "ic": 0.0351,
           "ic_t": 4.0, "ic_retention": 0.8}
    line = la.leak_line({"VERDICT": "PASS", "segments": {"select": seg, "extend": seg}})
    assert line.startswith("PASS — select: identity PASS, score-vs-split-factor -0.017 (t -0.9), IC +0.0351 (t +4.0), retention 0.8; extend:")
    assert la.leak_line({"VERDICT": "FAIL"}) == "FAIL."


def test_audit_refuses_the_holdout(tmp_path):
    from stocks_ml.selection import HOLDOUT_START
    pd.DataFrame({"week": [HOLDOUT_START], "ticker": ["A"], "c1": [1.0]}).to_parquet(tmp_path / "p.parquet")
    with pytest.raises(RuntimeError, match="pre-holdout"):
        la.audit("unused", tmp_path / "p.parquet")


def test_feature_factor_check_flags_a_column_that_tracks_future_splits():
    from types import SimpleNamespace

    from stocks_ml.leak_audit import FEATURE_FACTOR_LIMIT, feature_factor_check
    rng = np.random.default_rng(0)
    dates = pd.date_range("2006-01-06", periods=30, freq="W-FRI")
    rows = []
    for d in dates:
        for i in range(40):
            factor = 0.0 if i % 2 else float(rng.uniform(0.1, 0.7))        # half the names split later
            rows.append({"date": d, "ticker": f"T{i}", "factor": factor,
                         "leaky": -factor + rng.normal(0, 0.01), "clean": rng.normal()})
    df = pd.DataFrame(rows)
    ctx = SimpleNamespace(pan=df[["date", "ticker", "leaky", "clean"]],
                          prices=pd.DataFrame({"date": df["date"], "ticker": df["ticker"],
                                               "close_split": np.exp(df["factor"]), "closeunadj": 1.0}))
    res = feature_factor_check(ctx, ["leaky", "clean"], "2006-01-01", "2006-12-31")
    assert res["leaky"]["verdict"] == "FAIL" and abs(res["leaky"]["corr_with_future_split_factor"]) > FEATURE_FACTOR_LIMIT
    assert res["clean"]["verdict"] == "PASS" and res["VERDICT"] == "FAIL"
