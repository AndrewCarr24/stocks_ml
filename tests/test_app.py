"""stocks_ml.app.build: the champion's explorer page (`stocks-ml app`)."""
import json
from pathlib import Path

import pytest

import stocks_ml.app.build as app


def _rec(t, wk, nxt, sleeves, pnl, vals, rotated, nav_prev):
    nav = nav_prev + sum(pnl.values())
    return {"t": t, "wk": wk, "nxt": nxt, "sleeves": sleeves, "pnl": pnl, "vals": vals,
            "rotated": rotated, "g": 0.0, "nav": nav, "r": nav / nav_prev - 1}


def test_positions_follow_names_across_sleeves_and_reassemble_nav():
    # the engine's NAV is 200 when the window opens; the app shows it from $100
    trace = [
        _rec("d0", "w0", "w1", [["A", "B"], ["C", "D"]],
             {"A": 5.0, "B": -0.1, "C": -0.1, "D": -0.1, "SPY": 1.0},
             {"A": 0.10, "B": 0.0, "C": 0.0, "D": 0.0}, rotated=[0, 1], nav_prev=200.0),
        _rec("d1", "w1", "w2", [["A", "B"], ["A", "E"]],
             {"A": 6.0, "B": 0.0, "C": -0.05, "D": -0.05, "E": -0.1, "SPY": 0.0},
             {"A": 0.05, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}, rotated=[1], nav_prev=205.7),
        _rec("d2", "w2", "w3", [["F", "G"], ["A", "E"]],
             {"A": -1.0, "B": -0.05, "E": 0.0, "F": -0.1, "G": -0.1, "SPY": 0.5},
             {"A": -0.02, "B": 0.01, "E": 0.0, "F": 0.0, "G": 0.0}, rotated=[0], nav_prev=211.5),
    ]
    nav, pos = app.build_positions(trace)
    assert nav[0] == 100.0
    assert nav[-1] == pytest.approx(100 * (1 + trace[0]["r"]) * (1 + trace[1]["r"]) * (1 + trace[2]["r"]))
    by = {p["k"]: p for p in pos}
    assert by["A"]["n"] == [1, 2, 1] and by["A"]["s"] is None and by["A"]["b"] == "d0"
    assert by["A"]["p"] == pytest.approx([2.5, 3.0, -0.5])            # halved: $200 -> $100
    assert by["C"]["s"] == "w1" and by["C"]["n"] == [1, 0]             # sold the week after
    assert by["C"]["p"] == pytest.approx([-0.05, -0.025]) and by["C"]["v"] == [0.0, 0.0]
    assert by["B"]["s"] == "w2" and by["B"]["v"][-1] == 0.01           # the gap on the way out
    assert by["F"]["b"] == "d2" and by["F"]["j0"] == 2
    # every week: the names' $ plus the ballast's add back to the NAV change
    for j, rec in enumerate(trace):
        names = sum(p["p"][j - p["j0"]] for p in pos if p["j0"] <= j < p["j0"] + len(p["p"]))
        assert names + rec["pnl"]["SPY"] / 2 == pytest.approx(nav[j + 1] - nav[j])


def test_build_positions_rejects_a_trace_that_does_not_add_up():
    bad = [_rec("d0", "w0", "w1", [["A"]], {"A": 1.0}, {"A": 0.01}, [0], 100.0)]
    bad[0]["nav"] = 102.0
    with pytest.raises(AssertionError):
        app.build_positions(bad)


def test_stopped_names_are_read_off_the_weights():
    rec = {"sleeves": [["A", "B"], ["C"]], "weights": {"A": .2, "C": .2, "SPY": .6}}
    assert app.stopped(rec) == ["B"]                                    # B's money is in SPY
    assert app.stopped({"sleeves": [["A"]], "weights": {"A": .6, "SPY": .4}}) == []


def test_render_inlines_json_without_closing_the_script_and_sets_the_title(tmp_path):
    tpl = tmp_path / "app.html"
    tpl.write_text('<title>__TITLE__</title><script id="data" type="application/json">__DATA__</script>')
    out = app.render({"x": "</script><b>", "meta": {"title": "Champion Explorer"}},
                     template=tpl, out=tmp_path / "out.html")
    html = out.read_text()
    assert html.count("</script>") == 1 and "<title>Champion Explorer</title>" in html
    start = html.index('json">') + len('json">')
    assert json.loads(html[start:html.index("</script>")]) == {"x": "</script><b>", "meta": {"title": "Champion Explorer"}}


def test_the_template_reads_only_meta_fields_the_builder_writes():
    import re
    used = set(re.findall(r"D\.meta\.(\w+)", app.TEMPLATE.read_text()))
    written = {"title", "crumb", "scale", "store", "intro", "verdict", "built", "config", "lo", "hi",
               "first_pick", "last_pick", "book_word", "accounting"}
    assert used <= written, used - written


def test_words_describe_the_spec_settings_not_typed_ones():
    st = app.spec_settings()
    config = dict(horizon="4w", **st)
    text = app.describe(config, 8)
    assert text.startswith("4-week horizon, 8-year training window, top-")
    assert f"top-{st['book']} in four staggered sleeves" in text
    assert (f"sector cap {st['cap']}" if st["cap"] else "no sector cap") in text
    assert app.FLOOR_WORDS.get(st["floor"], f"{st['floor']} trend ballast") in text
    frozen = dict(horizon="4w", book=3, cap=None, stop=-0.25, floor="60/40")
    assert app.describe(frozen, 5) == ("4-week horizon, 5-year training window, top-3 in four "
                                       "staggered sleeves, no sector cap, 60/40 trend ballast, "
                                       "-25% stop-loss")
    a = app.accounting(frozen)
    assert "replaced by the top three;" in a and "40% of NAV" in a and "fallen 25%" in a
    b = app.accounting(dict(horizon="4w", book=10, cap=2, stop=None, floor="halfgate"))
    assert "top ten (at most two per sector)" in b and "stopped" not in b
    assert "100% with none breached, 50% with all three" in b and "rest sits in IEF" in b
    assert "no ballast: the whole of NAV is in the book" in app.accounting({**frozen, "floor": "none"})


def test_verdict_quotes_the_eval_record_not_prose():
    spec = {"ensemble": {"k_copies": 16}, "horizon": {"label": "label_4w_sector"}, "training_window_years": 8}
    row = lambda end: {"terminal_100": end, "cagr_pct": 1.0, "sharpe": 1.0, "max_dd": 0.1}  # noqa: E731
    ev = {"label": "champion", "evaluated_at": "2026-09-13 10:00:00",
          "table": {"champion": {"2006-2015": row(302), "2016-2024": row(660), "2006-2024": row(1994),
                                 "paired_t_vs_sp500": {"2006-2024": 1.9, "2016-2024": 1.52}},
                    "sp500": {"2006-2015": row(197), "2016-2024": row(316), "2006-2024": row(621)}},
          "incumbent": {"walk": "data/experiments/inc"},
          "falsification": {"2016-2024": {"t": 1.32, "weeks": 446}, "verdict": "not rejected"},
          "confidence": {"2016-2024": {"excess_cagr_vs_sp500_pct": {"point": 9.0, "nested_95": [-3.9, 24.2]},
                                       "p_excess_positive_nested": 0.91},
                         "2006-2024": {"excess_cagr_vs_sp500_pct": {"point": 6.5, "nested_95": [-2.5, 16.2]}}},
          "leak_audit": {"VERDICT": "PASS"}}
    v = app.verdict(spec, ev)
    assert "target label_4w_sector (the stock's 4-week return minus its sector's median that week)" in v
    assert "8-year trailing window" in v and "sixteen copies" in v
    assert "2016–2024 $660 vs $316" in v and "2006–2024 $1,994 vs $621" in v
    assert "paired weekly t vs the S&P 500 +1.90 (2006–2024) / +1.52 (2016–2024)" in v
    assert "incumbent (inc): paired weekly t +1.32 on 446 weeks of 2016–2024, not rejected" in v
    assert "-3.9..+24.2%/yr around +9.0 (2016–2024)" in v and "91% of resampled" in v
    assert "Leak audit PASS" in v and "holdout (2024-07-19 →) is untouched" in v
    # without an incumbent or intervals the paragraph still reads
    v2 = app.verdict(spec, {**ev, "falsification": None, "incumbent": None, "confidence": None})
    assert "Falsification" not in v2 and "nested interval" not in v2 and "Leak audit PASS" in v2


def test_build_refuses_a_walk_without_its_eval(tmp_path, monkeypatch):
    walk = tmp_path / "walk"
    (walk / "select").mkdir(parents=True)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"procedure": {"preds": {"path": str(walk / "select/preds.parquet")},
                                              "decision": {"book_size": 10, "floor": "halfgate",
                                                           "stop_loss": None, "sector_cap": None}},
                                "training_window_years": 8, "ensemble": {"k_copies": 16},
                                "horizon": {"label": "label_4w_sector"}}))
    with pytest.raises(SystemExit, match="run stocks-ml eval first"):
        app.build(store="unused", out=tmp_path / "out.html", spec_path=spec)
