"""features/generated.py: the raw inputs on a synthetic store, OpenFE's formula
syntax and operator semantics, the candidate space, the ranked g_ columns,
and the scoring helpers (LightGBM parts skipped when it is not installed)."""
import numpy as np
import pandas as pd
import pytest

from stocks_ml.features import generated as gen
from stocks_ml.features.ranking import rank_normalize

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def _base(weeks="2020-01-03", periods=60):
    dates = pd.date_range(weeks, periods=periods, freq="7D")
    return pd.DataFrame({"date": np.repeat(dates, len(TICKERS)), "ticker": TICKERS * len(dates)})


# ----------------------------------------------------------------------------- inputs
def test_sf1_inputs_level_and_yoy_are_point_in_time():
    rows = []
    for q, rp in enumerate(pd.date_range("2018-12-31", periods=8, freq="QE")):
        for dim in ("ARQ", "ART"):
            rows.append({"ticker": "AAA", "dimension": dim, "calendardate": rp, "reportperiod": rp,
                         "date": rp + pd.Timedelta(days=40), "revenue": 100.0 + 10 * q,
                         "netinc": -5.0 if dim == "ART" else 1.0, "note": "x"})
    fund = pd.DataFrame(rows)
    assert gen.fundamental_fields(fund) == ["netinc", "revenue"]
    base = pd.DataFrame({"date": pd.to_datetime(["2020-02-07", "2020-02-14", "2020-06-05"]),
                         "ticker": "AAA"})
    out = gen.sf1_inputs(fund, base)
    assert list(out.columns) == ["r_sf_netinc", "r_sf_revenue", "r_sf_netinc_yoy", "r_sf_revenue_yoy"]
    # Q4-2019 (reportperiod 2019-12-31) is filed 2020-02-09 -> usable from 02-10
    assert out["r_sf_revenue"].tolist()[:2] == [130.0, 140.0]
    assert out["r_sf_revenue_yoy"].iloc[1] == pytest.approx(140 / 100 - 1)
    # ART rows only: netinc is the ART value; yoy needs prev > 0
    assert out["r_sf_netinc"].iloc[0] == -5.0 and np.isnan(out["r_sf_netinc_yoy"].iloc[0])
    assert out["r_sf_revenue"].iloc[2] == 150.0


def test_sec8k_inputs_count_the_last_26_weeks_per_code():
    sec8k = pd.DataFrame([
        {"ticker": "AAA", "filed": pd.Timestamp("2020-03-02"), "items": "2.02,9.01",
         "accepted": pd.Timestamp("2020-03-02 21:00", tz="UTC")},
        {"ticker": "AAA", "filed": pd.Timestamp("2020-05-04"), "items": "2.02, 9.01",
         "accepted": pd.Timestamp("2020-05-04 21:00", tz="UTC")},
        {"ticker": "AAA", "filed": pd.Timestamp("2020-05-06"), "items": "5.02", "accepted": pd.NaT},
        {"ticker": "BBB", "filed": pd.Timestamp("2020-05-06"), "items": "12", "accepted": pd.NaT},
    ])
    base = pd.DataFrame({"date": pd.to_datetime(["2020-03-02", "2020-03-06", "2020-05-08", "2020-09-04"]),
                         "ticker": "AAA"})
    out = gen.sec8k_inputs(sec8k, base, min_filings=1)
    assert list(out.columns) == ["r_8k_2_02", "r_8k_5_02", "r_8k_9_01"]     # the old numbering is out
    assert out["r_8k_2_02"].tolist() == [0.0, 1.0, 2.0, 1.0]                 # acceptance + 1 day; 182 days
    assert out["r_8k_5_02"].tolist() == [0.0, 0.0, 1.0, 1.0]                 # filing date + 1 when no acceptance
    assert gen.item_codes(sec8k, min_filings=2) == ["2.02", "9.01"]


def test_close_input_is_the_close_as_of_the_rank_date():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    prices = pd.concat([pd.DataFrame({"date": days, "ticker": t, "close": np.arange(len(days)) + 1.0 + i})
                        for i, t in enumerate(TICKERS)])
    base = pd.DataFrame({"date": pd.to_datetime(["2020-01-03", "2020-01-05"]), "ticker": ["AAA", "BBB"]})
    out = gen.close_input(prices, base)
    assert out.name == "r_close"
    assert out.tolist() == [3.0, 4.0]             # 01-05 is a Sunday: the Friday close carries


# ----------------------------------------------------------------------------- formulas
def test_evaluate_matches_openfe_semantics():
    df = pd.DataFrame({"a": [1.0, -2.0, 0.0, np.nan], "b": [2.0, 0.0, 3.0, 1.0]})
    assert gen.evaluate("(a+b)", df).tolist()[:3] == [3.0, -2.0, 3.0]
    assert gen.evaluate("(a-b)", df).tolist()[:3] == [-1.0, -2.0, -3.0]
    assert gen.evaluate("(a*b)", df).tolist()[:3] == [2.0, -0.0, 0.0]
    div = gen.evaluate("(a/b)", df)
    assert div[0] == 0.5 and np.isnan(div[1]) and div[2] == 0.0 and np.isnan(div[3])
    assert gen.evaluate("min(a,b)", df).tolist()[:3] == [1.0, -2.0, 0.0]
    assert gen.evaluate("max(a,b)", df).tolist()[:3] == [2.0, 0.0, 3.0]
    assert np.isnan(gen.evaluate("max(a,b)", df)[3])
    assert gen.evaluate("abs(a)", df).tolist()[:3] == [1.0, 2.0, 0.0]
    assert gen.evaluate("b", df).tolist() == [2.0, 0.0, 3.0, 1.0]
    big = pd.DataFrame({"a": [1e300], "b": [1e300]})
    assert np.isnan(gen.evaluate("(a*b)", big)[0])
    for bad in ("(a+b", "log(a)", "(a**b)", "c"):
        with pytest.raises(ValueError):
            gen.evaluate(bad, df)
    assert gen.formula_inputs("(a/b)") == ["a", "b"]
    assert gen.formula_inputs("min(a,b)") == ["a", "b"]
    assert gen.formula_inputs("abs(a)") == ["a"] and gen.formula_inputs("a") == ["a"]


def test_enumerate_candidates_is_openfe_order_one_without_the_trivial_ones():
    out = gen.enumerate_candidates(["b", "a", "c"])
    assert len(out) == 3 * 6 + 3
    assert out[:2] == ["abs(a)", "(a+b)"] and "min(a,b)" in out and "(b/c)" in out
    assert len(set(out)) == len(out)
    # a week-constant m: no m-with-m formulas, no abs(m), no x+-m; products and ratios stay
    out = gen.enumerate_candidates(["a", "m", "n"], constant={"m", "n"}, identity=["a", "m"])
    assert "abs(m)" not in out and "(m+n)" not in out and "min(m,n)" not in out
    assert "(a+m)" not in out and "(a-m)" not in out
    assert "(a*m)" in out and "(a/m)" in out and "max(a,m)" in out
    assert out[0] == "a" and "m" not in out


def test_rank_week_matches_rank_normalize_and_add_generated_ranks_g_columns():
    base = _base(periods=5)
    rng = np.random.default_rng(0)
    inputs = base.assign(a=rng.normal(size=len(base)), b=rng.normal(size=len(base)))
    inputs.loc[3, "a"] = np.nan
    r = gen.rank_week(inputs["a"].to_numpy(), inputs["date"].to_numpy())
    expect = rank_normalize(inputs, ["a"])["a"].to_numpy()
    assert np.allclose(r, expect)
    panel = base.iloc[::-1].reset_index(drop=True).assign(f_x=1.0, g_old=9.0)   # reversed rows, a stale g_
    out = gen.add_generated(panel, inputs, {"g_00": "(a*b)", "g_01": "abs(a)"})
    assert list(out.columns) == ["date", "ticker", "f_x", "g_00", "g_01"]
    assert out[["date", "ticker", "f_x"]].equals(panel[["date", "ticker", "f_x"]])
    got = out.set_index(["date", "ticker"])["g_01"]
    want = rank_normalize(inputs.assign(g=np.abs(inputs["a"])), ["g"]).set_index(["date", "ticker"])["g"]
    assert np.allclose(got.reindex(want.index), want)
    assert out["g_00"].between(-1, 1).all()


def test_week_constant_finds_market_columns():
    base = _base(periods=3)
    inputs = base.assign(f_mom=np.arange(len(base), dtype=float),
                         f_mkt=np.repeat([1.0, 2.0, 3.0], len(TICKERS)),
                         f_month=np.repeat([np.nan, 2.0, 2.0], len(TICKERS)))
    assert gen.week_constant(inputs) == {"f_mkt", "f_month"}


# ----------------------------------------------------------------------------- scoring
def test_week_blocks_and_purge():
    base = _base(periods=60)                       # 60 weeks: blocks of 26 -> folds 0, 1, 2
    fold = gen.week_blocks(base["date"], block_weeks=26, folds=5)
    dates = np.array(sorted(base["date"].unique()))
    assert set(fold) == {0, 1, 2}
    assert (fold[base["date"] < dates[26]] == 0).all() and (fold[base["date"] >= dates[52]] == 2).all()
    train = gen.purged_train(base["date"], fold, 1, purge_days=35)
    d = base["date"]
    assert not train[fold == 1].any()
    assert train[d < dates[26] - pd.Timedelta(days=35)].all()
    assert not train[(d >= dates[26] - pd.Timedelta(days=35)) & (d <= dates[51] + pd.Timedelta(days=35))].any()
    assert train[d > dates[51] + pd.Timedelta(days=35)].all()


def test_predictive_score_rewards_the_residual_and_gain_ranking_puts_it_first():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(1)
    base = _base(periods=120)
    n = len(base)
    X = rng.normal(size=(n, 2))
    signal = rng.normal(size=n)
    y = X[:, 0] + signal + 0.1 * rng.normal(size=n)
    fold = gen.week_blocks(base["date"], block_weeks=26, folds=5)
    init = gen.init_scores(X, y, base["date"], fold)
    assert init.shape == (n,) and np.corrcoef(init, y)[0, 1] > 0.5
    train, val = gen.purged_train(base["date"], fold, 0), fold == 0
    rmse = gen.init_metric(init, y, val)
    good = gen.predictive_score(signal, y, init, train, val, rmse)
    noise = gen.predictive_score(rng.normal(size=n), y, init, train, val, rmse)
    assert good > 0.2 and noise < 0.05
    M = np.column_stack([X, rng.normal(size=n), signal])
    gains = gen.gain_ranking(M, ["x0", "x1", "noise", "signal"], y, init, train, val)
    assert gains.index[0] == "signal"


# ----------------------------------------------------------------------------- ranking yardstick (v2)
def _wide(periods=120, names=20, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-03", periods=periods, freq="7D")
    tickers = [f"T{i:02d}" for i in range(names)]
    base = pd.DataFrame({"date": np.repeat(dates, names), "ticker": tickers * periods})
    return base, rng


def test_weekly_ic_t_scores_the_ranking_signal_and_ignores_noise():
    base, rng = _wide()
    n = len(base)
    signal = rng.normal(size=n)
    y = 0.02 * signal + 0.05 * rng.normal(size=n)
    ic = gen.WeeklyIC(y, base["date"], min_weeks=60)
    m, t, w = ic(gen.rank_week(signal, base["date"]))
    assert w == 120 and m > 0.25 and t > 8
    m2, t2, _ = ic(gen.rank_week(-signal, base["date"]))
    assert m2 == pytest.approx(-m) and t2 == pytest.approx(-t)
    _, tn, _ = ic(gen.rank_week(rng.normal(size=n), base["date"]))
    assert abs(tn) < 4
    # a column constant in half the weeks: those weeks drop out; below min_weeks -> NaN
    half = signal.copy()
    half[base["date"] < base["date"].iloc[len(base) // 2]] = 0.0
    _, th, wh = ic(gen.rank_week(half, base["date"]))
    assert wh == 60 and th > 5
    ic2 = gen.WeeklyIC(y, base["date"], min_weeks=61)
    assert np.isnan(ic2(gen.rank_week(half, base["date"]))[1])
    # matches the pandas Spearman week by week
    r = gen.rank_week(signal, base["date"])
    want = base.assign(r=r, y=y).groupby("date")[["r", "y"]].corr(method="spearman").xs("y", level=1)["r"]
    assert m == pytest.approx(float(want.mean()), abs=1e-9)


def test_dedup_skips_near_copies_and_stops_at_n():
    base, rng = _wide(periods=40)
    n = len(base)
    cols = {"a": rng.normal(size=n)}
    cols["a_copy"] = cols["a"] + 0.01 * rng.normal(size=n)
    cols["b"] = rng.normal(size=n)
    cols["b_copy"] = -cols["b"] + 0.01 * rng.normal(size=n)
    cols["d"] = rng.normal(size=n)
    ranked = {k: gen.rank_week(v, base["date"]) for k, v in cols.items()}
    held = np.column_stack([ranked["a"]])
    kept = gen.dedup(["a_copy", "b", "b_copy", "d"], held, ranked.__getitem__, n=5, max_abs=0.9)
    assert [k for k, _ in kept] == ["b", "d"]
    assert all(r < 0.3 for _, r in kept)
    assert [k for k, _ in gen.dedup(["a_copy", "b", "b_copy", "d"], held, ranked.__getitem__, n=1)] == ["b"]
    # nothing held: the first candidate is always kept
    assert gen.dedup(["a_copy"], np.empty((n, 0)), ranked.__getitem__, n=1) == [("a_copy", 0.0)]


# ----------------------------------------------------------------------------- nominal basis (2026-09)
def _leak_prices():
    """AAA splits 4:1 between 2020 and the download: closeadj/close carry the
    restated history (nominal 40 stored as 10), closeunadj the tape."""
    dates = pd.date_range("2019-12-30", "2020-07-03", freq="B")
    rows = []
    for d in dates:
        rows.append({"date": d, "ticker": "AAA", "open": 10.0, "close": 10.0,
                     "volume": 100.0, "closeunadj": 40.0, "close_split": 10.0})
        rows.append({"date": d, "ticker": "BBB", "open": 20.0, "close": 20.0,
                     "volume": 100.0, "closeunadj": 20.0, "close_split": 20.0})
    return pd.DataFrame(rows)


def test_close_input_nominal_reads_the_tape():
    base = pd.DataFrame({"date": pd.to_datetime(["2020-02-07"] * 2), "ticker": ["AAA", "BBB"]})
    adj = gen.close_input(_leak_prices(), base)
    nom = gen.close_input(_leak_prices(), base, field="closeunadj")
    assert adj.tolist() == [10.0, 20.0]        # the leak: AAA reads a quarter of its price
    assert nom.tolist() == [40.0, 20.0]


def test_sf1_per_share_levels_are_restored_by_the_split_factor():
    rows = []
    for q, rp in enumerate(pd.date_range("2018-12-31", periods=5, freq="QE")):
        rows.append({"ticker": "AAA", "dimension": "ART", "reportperiod": rp,
                     "date": rp + pd.Timedelta(days=40),
                     "bvps": 1.9, "revenue": 100.0 + q})    # bvps stored split-restated (true 7.6)
    fund = pd.DataFrame(rows)
    base = pd.DataFrame({"date": pd.to_datetime(["2020-02-07"]), "ticker": ["AAA"]})
    factor = gen.split_factor_input(_leak_prices(), base)
    assert factor.tolist() == [4.0]
    out = gen.sf1_inputs(fund, base, split_factor=factor)
    assert out["r_sf_bvps"].iloc[0] == pytest.approx(7.6)   # the as-of-2020 book value
    assert out["r_sf_revenue"].iloc[0] == 103.0   # dollar totals untouched (Q4 files 02-09: not yet visible)
    # yoy is a same-basis ratio: untouched by the factor
    plain = gen.sf1_inputs(fund, base)
    assert plain["r_sf_bvps"].iloc[0] == pytest.approx(1.9)


def test_store_inputs_nominal_requires_the_level_columns():
    class FakeStore:
        def read(self, name):
            if name == "prices":
                return _leak_prices().drop(columns=["closeunadj", "close_split"])
            raise AssertionError(name)
    base = pd.DataFrame({"date": pd.to_datetime(["2020-02-07"]), "ticker": ["AAA"]})
    with pytest.raises(RuntimeError, match="closeunadj"):
        gen.store_inputs(FakeStore(), base, price_basis="nominal")
