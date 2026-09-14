"""The rolling procedure: windows end at label_end(t), a trailing rule waits
for its whole window, the followed rule uses the decision in force, and the
lookback is chosen by argmax on the common span."""
import json

import numpy as np
import pandas as pd
import pytest

import stocks_ml.rolling as roll
import stocks_ml.selection as sel


def test_windows_end_at_label_end_and_a_trailing_rule_waits_for_its_window():
    weeks = pd.date_range("2002-01-04", "2010-12-31", freq="W-FRI")
    w = roll.windows(weeks, 3)
    t, lo, hi = w[0]
    assert t == pd.Timestamp("2005-01-07")                       # first t with t - 3y >= first week
    assert lo == t - pd.DateOffset(years=3)
    assert hi == t - pd.Timedelta(days=35)                       # label_end: the 4w return is realized
    assert all(hi_ == t_ - pd.Timedelta(days=35) for t_, _, hi_ in w)
    assert len(roll.windows(weeks, 3, cadence=4)) == (len(w) + 3) // 4
    e = roll.windows(weeks, None, min_years=3)                   # expanding: lo pinned to the start
    assert e[0][0] == pd.Timestamp("2005-01-07") and all(lo_ == weeks[0] for _, lo_, _ in e)
    assert roll.windows(weeks, 20) == []


def test_parse_lookback_and_variant_names():
    assert roll.parse_lookback("expanding") is None and roll.parse_lookback("8") == 8
    assert roll.variant_name(None, 1) == "expanding_c1" and roll.variant_name(5, 4) == "trailing_5_c4"
    with pytest.raises(ValueError):
        roll.parse_lookback("0")


def test_decisions_call_decide_strategy_on_each_window_in_process(monkeypatch):
    weeks = pd.date_range("2006-01-06", periods=52 * 5, freq="W-FRI")
    hold = pd.DataFrame({"week": weeks, "top3": 0.0, "top6": 0.0, "top10": 0.0,
                         "spy": 0.0, "rand_mean": 0.0, "top15": "A"})
    seen = []

    def fake(ctx, holdings, horizon, lo, hi):
        seen.append((lo, hi))
        book = 10 if hi.year >= 2010 else 6
        return {"book": book, "floor": "halfgate", "stop": None, "cap": None,
                "evidence": {"book": {"6": 1.0}}}
    monkeypatch.setattr(sel, "decide_strategy", fake)
    dec = roll.decisions(sel, None, "store", hold, 3, cadence=13, workers=1, log=lambda m: None)
    assert list(dec.columns) == ["lo", "hi", "book", "floor", "stop", "cap", "evidence"]
    assert dec.index[0] == weeks[weeks >= weeks[0] + pd.DateOffset(years=3)][0]
    assert seen[0] == (dec.index[0] - pd.DateOffset(years=3), dec.index[0] - pd.Timedelta(days=35))
    assert (dec.book == 6).sum() > 0 and (dec.book == 10).sum() > 0


def test_settings_at_uses_the_decision_in_force_and_reads_nan_as_none():
    dec = pd.DataFrame({"book": [6, 10], "floor": ["60/40", "halfgate"],
                        "stop": [-0.25, np.nan], "cap": [2, None]},
                       index=pd.to_datetime(["2010-01-08", "2011-01-07"]))
    assert sel.settings_at(dec, "2010-06-04") == (6, "60/40", -0.25, 2)
    assert sel.settings_at(dec, "2011-01-07") == (10, "halfgate", None, None)
    with pytest.raises(ValueError):
        sel.settings_at(dec, "2009-12-31")


def test_summarize_counts_changes_and_the_share_equal_to_the_spec():
    dec = pd.DataFrame({"book": [6, 6, 10, 10], "floor": ["halfgate"] * 4,
                        "stop": [None] * 4, "cap": [None, 2, None, None]},
                       index=pd.date_range("2010-01-08", periods=4, freq="W-FRI"))
    s = roll.summarize(dec, {"book": 10, "floor": "halfgate", "stop": None, "cap": None})
    assert (s["decisions"], s["changes"], s["share_equal_to_spec"]) == (4, 2, 0.5)
    assert s["mode"]["book"] in ("6", "10") and s["share_by_layer"]["cap"]["2"] == 0.25


def _rolling_json(path, name, first, weekly_rule, weekly_fixed, weekly_spy, spec):
    idx = pd.date_range("2008-01-04", periods=len(weekly_rule), freq="W-FRI")
    rec = {"variant": name, "spec_settings": spec,
           "summary": {"first": first, "decisions": 10, "changes": 2, "share_equal_to_spec": 0.5},
           "decisions": [{"t": "2016-01-08", "lo": "2008-01-04", "hi": "2015-12-04",
                          "book": 10, "floor": "halfgate", "stop": None, "cap": None}],
           "weekly": {"rule": {str(d.date()): float(v) for d, v in zip(idx, weekly_rule)},
                      "fixed": {str(d.date()): float(v) for d, v in zip(idx, weekly_fixed)},
                      "spy": {str(d.date()): float(v) for d, v in zip(idx, weekly_spy)}}}
    path.write_text(json.dumps(rec))
    return path


def test_choose_is_the_argmax_on_the_common_span_and_refuses_a_late_rule(tmp_path, monkeypatch):
    n = 52 * 17
    wig = 0.001 * (-1.0) ** np.arange(n)                          # so no series is constant
    spec = {"book": 10, "floor": "halfgate", "stop": None, "cap": None}
    a = _rolling_json(tmp_path / "a.json", "trailing_3_c1", "2008-01-04",
                      0.003 + wig, 0.002 + wig, 0.001 + wig, spec)
    b = _rolling_json(tmp_path / "b.json", "trailing_5_c1", "2008-01-04",
                      0.001 + wig, 0.002 + wig, 0.001 + wig, spec)
    led = []
    import stocks_ml.models.trials as trials
    monkeypatch.setattr(trials, "record_trials", lambda rows, **k: led.extend(rows))
    out = roll.choose([a, b], "2010-01-01", "2015-12-31", log=lambda m: None)
    assert out["choice"] == "trailing_3_c1" and out["graded_on"] == ["2010-01-01", "2015-12-31"]
    assert set(out["candidates"]) == {"trailing_3_c1", "trailing_5_c1"}
    assert set(out["one_look"]["rows"]) == {"trailing_3_c1 (rolling)", "spec settings, fixed", "sp500"}
    assert led and led[0]["kind"] == "rolling_lookback"
    assert (tmp_path / "lookback_choice_2010-2015.json").exists()
    late = _rolling_json(tmp_path / "c.json", "trailing_8_c1", "2011-01-07",
                         0.003 + wig, 0.002 + wig, 0.001 + wig, spec)
    with pytest.raises(SystemExit):
        roll.choose([a, late], "2010-01-01", "2015-12-31", log=lambda m: None)
