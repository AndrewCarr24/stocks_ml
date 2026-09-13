from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stocks_ml.selection import (COST, COST_BPS, compounded_pct, decide_book, floor_split, label_end,
                                 next_open, pick_capped, price_frames, simulate, slice_row, week_slot)


def _frame(weeks, top6, spy=0.0, rand=0.0, top3=None, top10=None):
    return pd.DataFrame({
        "week": weeks, "top3": top3 if top3 is not None else top6,
        "top6": top6, "top10": top10 if top10 is not None else top6,
        "spy": spy, "rand_mean": rand, "top15": "A,B,C"})


def test_decide_book_uses_named_columns():
    weeks = pd.date_range("2010-01-01", periods=104, freq="W-FRI")
    df = _frame(weeks, top6=0.01, top3=0.02, top10=0.005)
    b, res = decide_book(df, "4w", weeks[0], weeks[-1])
    assert b == 3 and res[3] > res[10]


def test_compounded_pct_reads_every_week_through_the_phases():
    """A 4-week book compounds along four non-overlapping chains; the
    statistic is their average, so a phase-0-only reading (which sees only
    the +4% weeks here) is not the answer."""
    weeks = pd.date_range("2010-01-01", periods=104, freq="W-FRI")
    r = np.where(np.arange(104) % 4 == 0, 0.04, 0.0)          # phase 0 alone earns
    df = _frame(weeks, r)
    phase0 = ((1 + 0.04 - COST) ** 26) ** (1 / 2) - 1           # 26 periods over 2 years
    other = ((1 - COST) ** 26) ** (1 / 2) - 1
    assert compounded_pct(df, "top6", 4, weeks[0], weeks[-1]) == pytest.approx(
        (phase0 + 3 * other) / 4 * 100)
    # one-week books have a single phase: the plain chain
    assert compounded_pct(df, "top6", 1, weeks[0], weeks[-1]) == pytest.approx(
        (float(np.prod(1 + r - COST)) ** (52 / 104) - 1) * 100)
    b, res = decide_book(df, "4w", weeks[0], weeks[-1])
    assert res[6] == pytest.approx((phase0 + 3 * other) / 4 * 100)


def test_pick_capped_spills_to_next_sector():
    smap = {"A": "tech", "B": "tech", "C": "tech", "D": "oil", "E": "oil", "F": "bank"}
    got = pick_capped(["A", "B", "C", "D", "E", "F"], cap=2, k=4, smap=smap)
    assert got == ["A", "B", "D", "E"]  # C blocked by tech cap


def test_pick_capped_fills_when_short():
    smap = {"A": "tech", "B": "tech", "C": "tech"}
    got = pick_capped(["A", "B", "C"], cap=2, k=3, smap=smap)
    assert len(got) == 3  # falls back rather than returning a short book


# --- rank-date alignment ---------------------------------------------------
# Rank dates are the last trading day of the week: a Friday, or a Thursday
# when the Friday is a holiday. A pick dated t is paid for the week that
# starts at t's close; snapping t back to the previous Friday label would
# pay it for the week that had already happened.

def _daily(closes, start="2015-12-04", holidays=(), opens=None):
    """A daily world from weekly closes: every session of a week carries the
    week's close and each open is the previous session's close, so a fill at
    Monday's open is Friday's close unless `opens` ({ticker: {date: open}})
    says otherwise."""
    n = len(next(iter(closes.values())))
    labels = pd.date_range(start, periods=n, freq="W-FRI")
    days = pd.bdate_range(labels[0], labels[-1]).difference(pd.to_datetime(list(holidays)))
    wk = np.searchsorted(labels.values, days.values)
    cl = pd.DataFrame({tk: np.asarray(v, float)[wk] for tk, v in closes.items()}, index=days)
    op = cl.shift(1).fillna(cl)
    for tk, fix in (opens or {}).items():
        for d, px in fix.items():
            op.loc[pd.Timestamp(d), tk] = px
    return SimpleNamespace(smap={}, members={}, **price_frames(cl, op))


FEE = COST_BPS / 1e4                                   # one side


def test_week_slot_is_the_friday_label_of_the_pick_week():
    idx = pd.date_range("2015-12-04", periods=5, freq="W-FRI")
    assert week_slot(idx, pd.Timestamp("2015-12-24")) == pd.Timestamp("2015-12-25")
    assert week_slot(idx, pd.Timestamp("2015-12-25")) == pd.Timestamp("2015-12-25")
    assert week_slot(idx, pd.Timestamp("2016-01-08")) is None


def test_next_open_is_the_first_open_after_the_label():
    days = pd.bdate_range("2015-12-04", "2015-12-18")
    op = pd.DataFrame({"A": np.arange(len(days), dtype=float),   # open = session number
                       "B": np.arange(len(days), dtype=float)}, index=days)
    op.loc["2015-12-07", "A"] = np.nan                            # no Monday print
    op.loc["2015-12-07":"2015-12-11", "B"] = np.nan               # halted all week
    labels = pd.DatetimeIndex(["2015-12-04", "2015-12-11", "2015-12-18"])
    got = next_open(op, labels)
    assert got.loc["2015-12-04", "A"] == 2                        # Tuesday's open
    assert np.isnan(got.loc["2015-12-04", "B"])                   # nothing within five sessions
    assert got.loc["2015-12-11", "B"] == 6                        # Monday 12-14
    assert np.isnan(got.loc["2015-12-18", "A"])                   # nothing after the last session


def test_simulate_pays_a_thursday_pick_for_the_following_week():
    # labels: 12-04, 12-11, 12-18, 12-25, 01-01; Fri 12-25 is a holiday, so
    # the pick is dated Thu 12-24. A: +10% into that week, -5% the week
    # after. The book buys at Monday 12-28's open and pays a side.
    ctx = _daily({"A": [100, 100, 100, 110, 104.5], "SPY": [100] * 5},
                 holidays=["2015-12-25", "2016-01-01"])
    holdings = pd.DataFrame({"week": [pd.Timestamp("2015-12-24")], "top15": ["A"]})
    rets = simulate(ctx, holdings, "1w", book=1, cap=None, stop=None, floor="none")
    assert list(rets.index) == [pd.Timestamp("2016-01-01")]
    assert rets.iloc[0] == pytest.approx(0.95 / (1 + FEE) - 1)


def test_simulate_holds_the_book_through_a_week_without_a_pick():
    # picks on 12-04 (A) and 12-18 (B); no pick dated 12-11.
    ctx = _daily({"A": [100, 102, 105.06, 105.06, 105.06],
                  "B": [100, 100, 100, 103, 103], "SPY": [100] * 5})
    holdings = pd.DataFrame({"week": pd.to_datetime(["2015-12-04", "2015-12-18"]),
                             "top15": ["A", "B"]})
    rets = simulate(ctx, holdings, "1w", book=1, cap=None, stop=None, floor="none")
    assert list(rets.index) == list(pd.to_datetime(["2015-12-11", "2015-12-18", "2015-12-25"]))
    assert rets.iloc[0] == pytest.approx(1.02 / (1 + FEE) - 1)           # A bought
    assert rets.iloc[1] == pytest.approx(0.03)                           # A held, no trade
    assert rets.iloc[2] == pytest.approx(1.03 * (1 - FEE) / (1 + FEE) - 1)  # A sold, B bought


def test_simulate_fills_at_the_next_open_not_the_rank_close():
    # A gaps up 2% at Monday's open and is flat after: a fill at Friday's
    # close would have earned the gap; the deployed book buys after it.
    ctx = _daily({"A": [100, 102, 102], "SPY": [100] * 3},
                 opens={"A": {"2015-12-07": 102}})
    holdings = pd.DataFrame({"week": [pd.Timestamp("2015-12-04")], "top15": ["A"]})
    rets = simulate(ctx, holdings, "1w", 1, None, None, "none")
    assert rets.loc[pd.Timestamp("2015-12-11")] == pytest.approx(1 / (1 + FEE) - 1)


def test_simulate_skips_dust_rebalances():
    # two names drift 1% apart: the 0.5% rebalance each way is under the
    # ledger's threshold, so the week after costs nothing
    ctx = _daily({"A": [100, 101, 101], "B": [100, 100, 100], "SPY": [100] * 3})
    holdings = pd.DataFrame({"week": pd.to_datetime(["2015-12-04", "2015-12-11"]),
                             "top15": ["A,B", "A,B"]})
    trace = []
    rets = simulate(ctx, holdings, "1w", 2, None, None, "none", trace=trace)
    assert rets.iloc[1] == pytest.approx(0.0)
    assert trace[1]["fills"] == []


def test_simulate_stop_parks_a_sleeve_in_spy_until_it_rotates():
    # four sleeves (4w) all hold A; it falls 30%. The stop moves every
    # sleeve into SPY except the one rotating that week, which re-enters A
    # at the new price; the next rotation brings another sleeve back.
    ctx = _daily({"A": [100, 100, 70, 70, 70], "SPY": [100] * 5})
    weeks = pd.date_range("2015-12-04", periods=5, freq="W-FRI")
    holdings = pd.DataFrame({"week": weeks, "top15": ["A"] * 5})
    trace = []
    simulate(ctx, holdings, "4w", 1, None, -0.25, "none", trace=trace)
    assert [x["weights"] for x in trace] == [
        {"A": 1.0}, {"A": 1.0}, {"SPY": 0.75, "A": 0.25}, {"A": 0.5, "SPY": 0.5}]
    trace = []
    simulate(ctx, holdings, "4w", 1, None, None, "none", trace=trace)
    assert all(x["weights"] == {"A": 1.0} for x in trace)


def test_floor_split_menu():
    gates = {"30": "IEF", "40": "SPY", "52": "SPY"}
    assert floor_split("none", gates) == (1.0, {})
    assert floor_split("halfgate", gates) == (pytest.approx(1 - 0.5 / 3), {"30": "IEF"})
    assert floor_split("halfgate", {"30": "SPY", "40": "SPY", "52": "SPY"}) == (1.0, {})
    assert floor_split("70/30", gates) == (0.7, gates)


def test_slice_row_reads_the_forward_return_from_the_pick_week():
    names = [f"N{i}" for i in range(101)]
    closes = {n: [100] * 6 for n in names}
    closes["N0"] = [100, 100, 100, 110, 104.5, 104.5]      # +10% then -5%
    closes["SPY"] = [100, 100, 100, 101, 102.01, 102.01]     # +1% then +1%
    ctx = _daily(closes, holidays=["2015-12-25"])
    t = pd.Timestamp("2015-12-24")
    ctx.members[t] = names + ["SPY"]
    preds = pd.Series(np.linspace(1, 0, len(names)), index=names)  # N0 ranked first
    row = slice_row(ctx, t, "1w", preds)
    assert row["spy"] == pytest.approx(0.01)
    assert row["top3"] == pytest.approx(-0.05 / 3)


def test_slice_row_grades_on_the_fill_basis():
    # N0 gaps up 2% at Monday's open and is flat: the pick earns nothing,
    # because the book buys at the open the gap has already happened at
    names = [f"N{i}" for i in range(101)]
    closes = {n: [100, 100, 100] for n in names}
    closes["N0"] = [100, 102, 102]
    closes["SPY"] = [100, 100, 100]
    ctx = _daily(closes, opens={"N0": {"2015-12-07": 102}})
    t = pd.Timestamp("2015-12-04")
    ctx.members[t] = names + ["SPY"]
    preds = pd.Series(np.linspace(1, 0, len(names)), index=names)
    assert slice_row(ctx, t, "1w", preds)["top3"] == pytest.approx(0.0)


def test_simulate_trace_reassembles_each_credited_week():
    ctx = _daily({"A": [100, 102, 105.06, 105.06, 105.06],
                  "B": [100, 100, 100, 103, 103], "SPY": [100, 101, 102, 103, 104]})
    holdings = pd.DataFrame({"week": pd.to_datetime(["2015-12-04", "2015-12-18"]),
                             "top15": ["A", "B"]})
    trace = []
    rets = simulate(ctx, holdings, "1w", 1, None, None, "60/40", trace=trace)
    assert [x["nxt"] for x in trace] == list(rets.index)
    assert [x["rotated"] for x in trace] == [[0], [], [0]]
    assert [x["sleeves"] for x in trace] == [[["A"]], [["A"]], [["B"]]]
    assert [x["t"] for x in trace] == list(pd.to_datetime(["2015-12-04", "2015-12-04", "2015-12-18"]))
    assert trace[0]["weights"] == pytest.approx({"A": 0.6, "SPY": 0.4})
    nav = 100.0
    for x in trace:
        assert sum(x["pnl"].values()) == pytest.approx(x["nav"] - nav)  # fees included
        assert x["r"] == pytest.approx(x["nav"] / nav - 1)
        nav = x["nav"]
    assert trace[0]["vals"]["A"] == pytest.approx(0.02)   # bought Monday at 100, closed 102
    assert trace[1]["vals"]["A"] == pytest.approx(0.03)   # held
    assert trace[2]["vals"]["A"] == pytest.approx(0.0)    # sold Monday at Friday's close
    assert trace[2]["vals"]["B"] == pytest.approx(0.03)   # bought
    fee_a = next(f for f in trace[2]["fills"] if f[1] == "A")[4]
    assert trace[2]["pnl"]["A"] == pytest.approx(-fee_a)    # sold flat: the fee is the loss


def test_decisions_stop_where_labels_would_end_after_the_window():
    """No layer reads a rank week whose forward label ends after the window's
    end: at a window closing at the holdout's edge those weeks would be
    graded on holdout prices (label_span_days: 35 days for 4w, 14 for 1w)."""
    assert label_end("2024-07-18") == pd.Timestamp("2024-06-13")
    assert label_end("2024-07-18", 1) == pd.Timestamp("2024-07-04")
    weeks = pd.date_range("2010-01-01", periods=104, freq="W-FRI")
    lo, hi = weeks[0], weeks[-1]
    tail = weeks > label_end(hi)                                  # the last five rank weeks
    assert tail.sum() == 5
    # a windfall in the tail would flip the book layer; the cut leaves it unread
    df = _frame(weeks, top6=0.01, top3=np.where(tail, 0.9, 0.0), top10=0.005)
    b, res = decide_book(df, "4w", lo, hi)
    assert b == 6
    b, res = decide_book(_frame(weeks, top6=0.01, top3=np.where(weeks > label_end(hi, 1), 0.9, 0.0)), "1w", lo, hi)
    assert b == 6                                                  # 1w's own (shorter) span


def test_holdout_start_is_the_exclusive_grade_bound():
    """One convention (2026-09-09): every pre-holdout window ends at
    HOLDOUT_START exclusive, so the label credited at the first holdout close
    is never counted. Two conventions coexisted before (967- vs 966-week
    pre-holdout grades)."""
    from stocks_ml.selection import HOLDOUT_START, metrics
    assert str(HOLDOUT_START.date()) == "2024-07-19"
    idx = pd.DatetimeIndex(["2024-07-05", "2024-07-12", "2024-07-19"])
    r = pd.Series([0.01, 0.02, 99.0], index=idx)       # a holdout label to be excluded
    m = metrics(r, pd.Timestamp("2024-01-01"), HOLDOUT_START)
    assert m["n_weeks"] == 2 and m["terminal_100"] == pytest.approx(103.0, rel=1e-3)


def test_compounded_pct_chains_do_not_rephase_at_a_gap():
    """Chains stride by calendar week (week_index), so a missing rank week
    leaves the other chains intact; positional striding shifted every later
    week into the wrong chain."""
    from stocks_ml.selection import COST, compounded_pct
    weeks = pd.date_range("2024-01-05", periods=12, freq="W-FRI")
    df = pd.DataFrame({"week": weeks, "top6": np.where(np.arange(12) % 4 == 0, 0.04, 0.0)})
    # every 4th week earns 4%: those weeks share one calendar chain, the other
    # three chains are flat. Annualised chain rates don't depend on length, so
    # dropping a flat week must leave the statistic exactly unchanged.
    x, z = (1 + 0.04 - COST) ** 13 - 1, (1 - COST) ** 13 - 1
    expect = (x + 3 * z) / 4 * 100
    assert compounded_pct(df, "top6", 4, weeks[0], weeks[-1]) == pytest.approx(expect)
    gapped = compounded_pct(df.drop(index=1), "top6", 4, weeks[0], weeks[-1])
    assert gapped == pytest.approx(expect)     # positional striding re-phased here


def test_price_frames_grades_a_delisting_to_its_last_print():
    """delist='last_print': a name acquired (flat at deal price, then gone)
    grades to the deal; a collapse grades to its final near-zero print; weeks
    entirely after the end stay NaN; the data edge is right-censored."""
    from stocks_ml.selection import price_frames
    days = pd.bdate_range("2024-01-01", "2024-06-28")
    cl = pd.DataFrame({"DEAL": 50.0, "BUST": 40.0, "LIVE": 10.0, "SPY": 500.0}, index=days)
    op = cl - 0.1
    cl.loc["2024-02-20":, "DEAL"] = 65.0            # deal announced: jumps to offer
    cl.loc["2024-03-08":, "DEAL"] = np.nan          # cashed out
    op.loc["2024-02-20":, "DEAL"] = 64.9
    op.loc["2024-03-08":, "DEAL"] = np.nan
    bust_days = days[(days >= "2024-02-26") & (days <= "2024-03-05")]
    cl.loc["2024-02-26":, "BUST"] = np.nan
    op.loc["2024-02-26":, "BUST"] = np.nan
    cl.loc[bust_days, "BUST"] = np.linspace(8.0, 0.4, len(bust_days))   # collapse to pennies
    op.loc[bust_days, "BUST"] = np.linspace(8.1, 0.5, len(bust_days))
    drop = price_frames(cl.copy(), op.copy())
    hon = price_frames(cl.copy(), op.copy(), delist="last_print")
    w = pd.Timestamp("2024-02-16")                  # 4w window crosses both endings
    assert np.isnan(drop["fwd"]["4w"].at[w, "DEAL"])
    assert np.isnan(drop["fwd"]["4w"].at[w, "BUST"])
    entry_deal = hon["opens"].at[pd.Timestamp("2024-02-19"), "DEAL"]   # first fill after the label
    assert entry_deal == pytest.approx(49.9)                            # bought before the deal pop
    assert hon["fwd"]["4w"].at[w, "DEAL"] == pytest.approx(65.0 / entry_deal - 1)
    entry_bust = hon["opens"].at[pd.Timestamp("2024-02-19"), "BUST"]
    assert hon["fwd"]["4w"].at[w, "BUST"] == pytest.approx(0.4 / entry_bust - 1)
    assert hon["fwd"]["4w"].at[w, "BUST"] < -0.9
    # five weeks after both are gone: no label
    late = pd.Timestamp("2024-04-19")
    assert np.isnan(hon["fwd"]["4w"].at[late, "DEAL"])
    # LIVE runs to the data edge: right-censored, never treated as delisted
    assert np.isnan(hon["fwd"]["4w"].at[pd.Timestamp("2024-06-21"), "LIVE"])
    assert hon["last_print"]["BUST"] == pd.Timestamp("2024-03-05")


def test_make_labels_last_print_reaches_training():
    from stocks_ml.features.panel import make_labels
    days = pd.bdate_range("2024-01-01", "2024-06-28")
    rows = []
    for d in days:
        rows.append({"date": d, "ticker": "AAA", "open": 10.0, "close": 10.0})
        if d <= pd.Timestamp("2024-02-23"):
            rows.append({"date": d, "ticker": "DEAD", "open": 20.0, "close": 20.0})
    prices = pd.DataFrame(rows)
    dates = pd.DatetimeIndex(["2024-02-02"])
    drop = make_labels(prices, dates, 20)
    hon = make_labels(prices, dates, 20, delist="last_print")
    d_row = lambda df, tk: df[(df.ticker == tk)].iloc[0]
    assert np.isnan(d_row(drop, "DEAD")["fwd_ret"])
    assert d_row(hon, "DEAD")["fwd_ret"] == pytest.approx(0.0)      # exits flat at last print
    assert d_row(hon, "AAA")["fwd_ret"] == pytest.approx(0.0)
    # the terminal name now carries a LABEL (recentred), so training sees it
    assert np.isfinite(d_row(hon, "DEAD")["label"])


def test_slice_row_universe_uses_lives_traded_rule_under_last_print():
    from stocks_ml.selection import price_frames, slice_row
    days = pd.bdate_range("2023-06-01", "2024-06-28")
    tickers = {f"T{i:02d}": 10.0 + i for i in range(120)}
    cl = pd.DataFrame({**tickers, "STALE": 5.0, "SPY": 500.0}, index=days)
    cl.loc["2024-01-20":, "STALE"] = np.nan                  # last print weeks before t
    op = cl - 0.1
    frames = price_frames(cl.copy(), op.copy(), delist="last_print")
    ctx = SimpleNamespace(**frames, members={}, smap={}, delist_labels="last_print")
    t = pd.Timestamp("2024-03-01")
    ctx.members = {t: list(tickers) + ["STALE"]}
    preds = pd.Series(1.0, index=list(tickers) + ["STALE"])
    row = slice_row(ctx, t, "4w", preds)
    assert row is not None and "STALE" not in row["top15"]
    # drop mode keeps the old behavior byte for byte
    frames0 = price_frames(cl.copy(), op.copy())
    ctx0 = SimpleNamespace(**frames0, members={t: list(tickers)}, smap={}, delist_labels="drop")
    assert slice_row(ctx0, t, "4w", preds.drop("STALE")) is not None
