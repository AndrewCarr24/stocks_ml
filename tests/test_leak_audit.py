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


def test_identity_check_uses_the_builders_arq_frame_for_a_refiled_quarter():
    """A quarter filed twice becomes available at its LAST filing (the
    builder's rule); the check must agree with the stored panel, not with
    the first filing."""
    ctx, prices, fund, wk = _toy()
    again = fund[fund["ticker"] == "A"].assign(date=pd.Timestamp("2010-06-10"))   # re-filed after the week
    fund2 = pd.concat([fund, again], ignore_index=True)
    got = la.identity_check(ctx, prices, fund2, wk)
    assert not got["ok"]                        # A's bvps is not yet available: the stored rank must change
    ratio = pd.DataFrame({"date": [wk] * 3, "ticker": list("ABC"), "f_sf_book_to_market": [np.nan, 0.25, 2.0]})
    ctx.pan = rank_normalize(ratio, ["f_sf_book_to_market"])
    assert la.identity_check(ctx, prices, fund2, wk)["ok"]


def test_fundamentals_file_prefers_the_frozen_vintage(tmp_path):
    (tmp_path / "fundamentals.parquet").write_bytes(b"")
    assert la.fundamentals_file(tmp_path) == tmp_path / "fundamentals.parquet"
    (tmp_path / "fundamentals.frozen_2026-09-10.parquet").write_bytes(b"")
    (tmp_path / "fundamentals.frozen_2026-09-01.parquet").write_bytes(b"")
    assert la.fundamentals_file(tmp_path) == tmp_path / "fundamentals.frozen_2026-09-10.parquet"


def test_feature_scan_lists_the_models_own_features_beyond_the_limit(monkeypatch):
    monkeypatch.setattr(la, "feature_factor_check", lambda ctx, feats, lo, hi: {
        **{f: {"corr_with_future_split_factor": v, "weeks": 500, "verdict": "PASS" if abs(v) <= 0.15 else "FAIL"}
           for f, v in zip(feats, [0.02, -0.42, 0.20])}, "VERDICT": "FAIL"})
    scan = la.feature_scan(None, ["f_a", "f_dollar_vol", "f_b"])
    assert list(scan["beyond_limit"]) == ["f_dollar_vol", "f_b"] and scan["n_features"] == 3
    line = la.leak_line({"VERDICT": "PASS", "segments": {}, "feature_scan": scan})
    assert "feature scan: 2 of 3 beyond 0.15 (f_dollar_vol -0.42, f_b +0.20)" in line


def test_model_features_follow_the_walks_record(tmp_path):
    import json
    from types import SimpleNamespace
    pan = pd.DataFrame({"date": [pd.Timestamp("2010-01-08")], "ticker": ["A"], "f_mom_4w": [0.1], "f_dollar_vol": [0.2], "x_dollar_vol": [0.3]})
    (tmp_path / "spec.json").write_text(json.dumps({"recipe": {"label": "l", "train_years": 8, "features": ["x_dollar_vol"], "drop": ["f_dollar_vol"]}}))
    assert la.model_features(tmp_path / "preds.parquet", SimpleNamespace(pan=pan)) == ["f_mom_4w", "x_dollar_vol"]


def test_live_archive_and_comparison(tmp_path):
    from types import SimpleNamespace
    t = pd.Timestamp("2026-09-18")
    pan = pd.DataFrame({"date": [t] * 60, "ticker": [f"T{i}" for i in range(60)],
                        "f_a": np.linspace(-1, 1, 60), "f_b": np.linspace(-1, 1, 60)})
    path = la.archive_live_rows(tmp_path, SimpleNamespace(pan=pan), t)
    assert path.name == "features_2026-09-18.parquet" and len(pd.read_parquet(path)) == 60
    later = pan.copy()
    later["f_b"] = later["f_b"].iloc[::-1].to_numpy()          # f_b's history rewritten by the later build
    out = la.live_vs_rebuilt(tmp_path, later)
    got = out.set_index("feature")
    assert got.loc["f_a", "rank_agreement"] > 0.999 and got.loc["f_b", "rank_agreement"] < -0.99
    assert got.loc["f_a", "share_moved_0.1"] == 0.0 and got.loc["f_b", "share_moved_0.1"] > 0.9
    assert la.live_vs_rebuilt(tmp_path / "empty", later).empty


def test_missingness_scan_flags_a_feature_whose_blanks_mark_the_names_that_later_leave():
    """The 2026-09-21 EDGAR leak in miniature: `survivor` is blank for names that are
    gone by the panel's last week, and those names earn less; `random` is blank at
    random; `ind` is a 0/1 indicator and is skipped."""
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2016-01-08", periods=120, freq="7D")
    names = [f"T{i:02d}" for i in range(40)]
    leavers = set(names[:20])                                   # gone by the last week
    rows = []
    for w in weeks:
        for t in names:
            ret = rng.normal(-0.01 if t in leavers else 0.01, 0.02)
            rows.append((w, t, 0.0 if t in leavers else rng.uniform(-1, 1), 0.0 if rng.random() < 0.2 else rng.uniform(-1, 1),
                         float(rng.random() < 0.1), ret))
    pan = pd.DataFrame(rows, columns=["date", "ticker", "survivor", "random", "ind", "fwd_ret_4w"])
    members = {w: names for w in weeks[:-1]}
    members[weeks[-1]] = [t for t in names if t not in leavers]
    ctx = SimpleNamespace(pan=pan, weeks=list(weeks), members=members)
    out = la.missingness_scan(ctx, ["survivor", "random", "ind"], lo=weeks[0], hi=weeks[-1])
    assert set(out["beyond_limit"]) == {"survivor"}
    assert out["survival_keyed"] == ["survivor"] and out["VERDICT"] == "FAIL"
    s = out["all"]["survivor"]
    assert s["blank_share_left"] == 1.0 and s["blank_share_stayed"] == 0.0 and s["t"] < -la.MISSING_T_LIMIT and s["return_gap_pp"] < 0
    assert out["all"]["ind"]["skipped"] and not out["all"]["random"].get("skipped")
    line = la.leak_line({"VERDICT": "PASS", "segments": {}, "missingness_scan": out})
    assert "missingness scan: 1 of 3" in line and "survivor blank 50%" in line and "left 100% vs stayed 0%" in line


def test_coverage_by_survival_flags_a_table_pulled_for_todays_members(tmp_path):
    D = pd.Timestamp
    mem = pd.DataFrame([("STAY", D("2000-01-01"), pd.NaT, "s"), ("GONE", D("2000-01-01"), D("2020-01-01"), "s")],
                       columns=["ticker", "start_date", "end_date", "sector"])
    mem.to_parquet(tmp_path / "membership.parquet", index=False)
    days = pd.date_range("2015-01-01", "2019-12-31", freq="30D")
    pd.DataFrame({"ticker": "STAY", "filed": days}).to_parquet(tmp_path / "edgar.parquet", index=False)      # survivors only
    pd.DataFrame({"ticker": ["STAY", "GONE"] * len(days), "date": list(days) * 2}).to_parquet(tmp_path / "fundamentals.parquet", index=False)
    out = la.coverage_by_survival(tmp_path, years=(2016, 2019))
    assert out["tables"]["edgar"][2016] == {"left": 0.0, "stayed": 1.0} and out["tables"]["fundamentals"][2019] == {"left": 1.0, "stayed": 1.0}
    assert list(out["beyond_limit"]) == ["edgar"]
