"""The feature screen's statistics, probe and exam on synthetic frames."""
import numpy as np
import pandas as pd
import pytest

import stocks_ml.feature_screen as fs


def _panel(n_weeks=60, n_names=150, seed=0):
    """Weekly panel with a monotone candidate, a noise candidate and a sparse flag."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2010-01-08", periods=n_weeks, freq="7D")
    rows = []
    for w in weeks:
        label = rng.normal(size=n_names)
        good = label + rng.normal(scale=0.5, size=n_names)     # positively related
        noise = rng.normal(size=n_names)
        flag = np.where(rng.random(n_names) < 0.03, 1.0, 0.0)   # ~3% active
        rows.append(pd.DataFrame({"date": w, "ticker": [f"T{i}" for i in range(n_names)],
                                  "label_4w": label, "x_good": good, "x_noise": noise,
                                  "x_flag": flag}))
    return pd.concat(rows, ignore_index=True)


def test_nw_t_matches_plain_t_at_lag_zero():
    x = pd.Series(np.random.default_rng(1).normal(0.3, 1, 200))
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert fs.nw_t(x, 0) == pytest.approx(plain)
    assert np.isnan(fs.nw_t(x.iloc[:5], 4))


def test_hac_t_equals_plain_t_on_spaced_weeks():
    rng = np.random.default_rng(2)
    d = pd.Series(rng.normal(0.5, 1, 80))
    weeks = pd.Series(pd.date_range("2010-01-01", periods=80, freq="28D"))
    plain = d.mean() / (d.std(ddof=0) / np.sqrt(len(d)))
    assert fs.hac_t(d, weeks, 28) == pytest.approx(plain)
    # a smooth (autocorrelated) series on adjacent weeks: overlap widens the variance
    weekly = pd.Series(pd.date_range("2010-01-01", periods=80, freq="7D"))
    smooth = pd.Series(np.sin(np.arange(80) / 8) + 0.5)
    assert abs(fs.hac_t(smooth, weekly, 28)) < abs(fs.hac_t(smooth, weeks, 28))


def test_probe_keeps_signal_drops_noise_and_handles_sparse():
    pan = _panel()
    lo, hi = pan["date"].min(), pan["date"].max()
    res = fs.probe(pan, lo, hi, kweeks=4, label="label_4w").set_index("feature")
    assert res.loc["x_good", "keep"] and res.loc["x_good", "t"] > fs.T_BAR
    assert not res.loc["x_noise", "keep"]
    assert res.loc["x_flag", "sparse"] and not res.loc["x_good", "sparse"]
    # labels ending after the window are excluded: the last 5 weeks drop
    assert res.loc["x_good", "weeks"] == 60 - 5
    assert fs.keepers(res.reset_index()) == ["x_good"]


def test_screen_candidates_lists_x_and_pending_only():
    pan = pd.DataFrame({"f_mom": [1.0], "x_b": [1.0], "x_a": [1.0], "label": [0.0]})
    assert fs.screen_candidates(pan) == ["x_a", "x_b"]


def _holdings(weeks, top6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"week": weeks, "spy": rng.normal(0.01, 0.02, len(weeks)),
                         "rand_mean": rng.normal(0.01, 0.02, len(weeks)),
                         "top3": top6 + 0.001, "top6": top6, "top10": top6 - 0.001})


def test_exam_pass_and_fail():
    weeks = pd.Series(pd.date_range("2010-01-01", periods=120, freq="7D"))
    base = np.random.default_rng(3).normal(0.01, 0.03, 120)
    without = _holdings(weeks, base)
    lo, hi = weeks.iloc[0], weeks.iloc[-1] + pd.Timedelta(days=60)
    lift = _holdings(weeks, base + 0.02, seed=1)
    df = fs.paired_rows(without, lift, lo, hi, kweeks=4)
    assert len(df) == 120 and {"top6_without", "top6_with", "week"} <= set(df.columns)
    ex = fs.exam(df, kweeks=4)
    assert ex["passed"] and ex["hac_bandwidth_days"] == 28
    assert ex["stats"]["top6"]["diff"] == pytest.approx(0.02)
    same = fs.exam(fs.paired_rows(without, _holdings(weeks, base, seed=2), lo, hi, 4), 4)
    assert not same["passed"] and same["stats"]["top6"]["diff"] == pytest.approx(0.0)
    # weeks whose forward return ends after hi are dropped
    short = fs.paired_rows(without, lift, lo, weeks.iloc[-1], kweeks=4)
    assert len(short) == 120 - 5


def test_exam_admits_by_compounding_not_by_a_t_bar():
    """v3.1: the rule is the book layer's — the top-6 book compounds faster
    with the bundle — so a small lift admits without clearing any t bar,
    and a bundle that only adds variance (same mean, slower compounding)
    does not."""
    weeks = pd.Series(pd.date_range("2010-01-01", periods=200, freq="7D"))
    rng = np.random.default_rng(7)
    base = rng.normal(0.01, 0.03, 200)
    lo, hi = weeks.iloc[0], weeks.iloc[-1] + pd.Timedelta(days=60)
    without = _holdings(weeks, base)
    small = fs.exam(fs.paired_rows(without, _holdings(weeks, base + 0.002 + rng.normal(0, 0.02, 200), 1), lo, hi, 4), 4)
    assert small["rule"] == "argmax" and small["passed"]
    assert abs(small["stats"]["top6"]["t"]) < 2 and small["compounded_pct"]["with"] > small["compounded_pct"]["without"]
    swing = base + rng.permutation(np.repeat([0.25, -0.25], 100))         # same mean, huge variance
    noisy = fs.exam(fs.paired_rows(without, _holdings(weeks, swing, 1), lo, hi, 4), 4)
    assert noisy["stats"]["top6"]["diff"] == pytest.approx(0.0) and not noisy["passed"]


def test_write_and_load_screen(tmp_path):
    pan = _panel(n_weeks=40)
    lo, hi = pan["date"].min(), pan["date"].max()
    res = fs.probe(pan, lo, hi, 4, "label_4w")
    keep = fs.keepers(res)
    weeks = pd.Series(pd.date_range("2010-01-01", periods=50, freq="7D"))
    base = np.random.default_rng(5).normal(0.01, 0.03, 50)
    df = fs.paired_rows(_holdings(weeks, base), _holdings(weeks, base + 0.02, 1),
                        weeks.iloc[0], weeks.iloc[-1] + pd.Timedelta(days=60), 4)
    ex = fs.exam(df, 4)
    summary = fs.write_screen(tmp_path, "t", (lo, hi), 4, "label_4w", res, keep, ex, df,
                              report_path=tmp_path / "copy.md")
    assert summary["admitted"] == ["x_good"]
    loaded = fs.load_screen(tmp_path)
    assert loaded["admitted"] == ["x_good"] and loaded["exam"]["passed"]
    assert (tmp_path / "copy.md").read_text() == (tmp_path / "screen.md").read_text()
    assert fs.load_screen(tmp_path / "nothing") is None
    # no keepers: nothing admitted, no exam
    none = fs.write_screen(tmp_path / "n", "t", (lo, hi), 4, "label_4w",
                           res.assign(keep=False), [], None, None)
    assert none["admitted"] == [] and none["exam"] is None
