import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "oos_build", Path(__file__).resolve().parents[1] / "app/oos/build.py")
oos_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oos_build)


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
    nav, pos = oos_build.build_positions(trace)
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
        oos_build.build_positions(bad)


def test_render_inlines_json_without_closing_the_script(tmp_path):
    tpl = tmp_path / "app.html"
    tpl.write_text('<script id="data" type="application/json">__DATA__</script>')
    out = oos_build.render({"x": "</script><b>"}, template=tpl, out=tmp_path / "out.html")
    html = out.read_text()
    assert html.count("</script>") == 1
    start = html.index(">") + 1
    assert json.loads(html[start:html.index("</script>")]) == {"x": "</script><b>"}


def test_variants_describe_their_configuration_and_write_separate_pages(tmp_path):
    v = oos_build.VARIANTS
    assert len({x["out"] for x in v.values()}) == len(v)
    assert v["oos"]["rankings"].parent.name == "nested3_v2"              # procedure v3's population holdings
    assert v["oos"]["config"].parent == v["oos"]["screen"].parent == v["oos"]["rankings"].parent
    assert (v["oos"]["lo"], v["oos"]["hi"]) == ("2016-01-01", "2024-07-19")  # the grade's span, hi exclusive
    assert v["oos_x"]["rankings"].name == "holdings_4w_5y_xeeaf48_s0.parquet"  # the walk with the bundle
    assert v["oos_x"]["config"].name == "frozen_config_x.json"
    assert (v["oos_x"]["lo"], v["oos_x"]["hi"]) == (v["oos"]["lo"], v["oos"]["hi"])
    assert v["select"]["rankings"].name == "holdings_4w_5y_s0.parquet"   # the champion's cached 5y ranks
    assert (v["select"]["lo"], v["select"]["hi"]) == ("2006-01-01", "2024-07-18")  # holdout untouched
    # the champion page reads the spec, on the recorded basis: ranks before the holdout
    assert v["champion"]["config"]["features"] == oos_build.spec_config()["features"]
    assert v["champion"]["rankings"].parent.name == "openfe_v3_2006_2015"   # the generated bundle's walk
    assert v["champion"]["rankings"].name == "holdings_4w_5y_x724d05_s0.parquet"
    assert v["champion"]["screen"] is None                                  # no asterisk line: nothing screened
    assert oos_build.describe(oos_build.load_config(v["champion"]), 5).endswith(
        "sector cap 2, 70/30 trend ballast, no stop-loss, the generated bundle of 40 features")
    assert v["champion"]["ranks_before"] == "2024-07-18" and v["champion"]["lo"] == "2006-01-01"
    assert oos_build.load_config(v["champion"]) == dict(
        horizon="4w", book=6, cap=2, stop=None, floor="70/30", features=v["champion"]["config"]["features"])
    frozen = dict(horizon="4w", book=3, cap=None, stop=-0.25, floor="60/40")
    assert oos_build.describe(frozen, 5) == ("4-week horizon, 5-year training window, top-3 in four "
                                             "staggered sleeves, no sector cap, 60/40 trend ballast, "
                                             "-25% stop-loss")
    a = oos_build.accounting(frozen)
    assert "replaced by the top three;" in a and "40% of NAV" in a and "fallen 25%" in a
    # a cascade's frozen_config.json: the engine keys are read, evidence ignored
    cfg = tmp_path / "frozen_config.json"
    cfg.write_text(json.dumps(dict(horizon="4w", train_years=5, book=10, floor="60/40", stop=None,
                                   cap=2, features=[], evidence={"book": {"10": 1.9}})))
    clean = Path("holdings_4w_5y_s0.parquet")
    c = oos_build.load_config(dict(config=cfg, train_years=5, rankings=clean))
    assert oos_build.describe(c, 5).endswith(
        "top-10 in four staggered sleeves, sector cap 2, 60/40 trend ballast, no stop-loss")
    b = oos_build.accounting(c)
    assert "top ten (at most two per sector)" in b and "stopped" not in b
    with pytest.raises(AssertionError):                                  # the page describes a 5y walk
        oos_build.load_config(dict(config=cfg, train_years=3, rankings=clean))
    # a bundle's page must replay the walk that saw the bundle (the hash in the stem)
    cfg.write_text(json.dumps(dict(horizon="4w", train_years=5, book=3, floor="halfgate", stop=None,
                                   cap=2, features=["x_price_level"])))
    with pytest.raises(AssertionError):
        oos_build.load_config(dict(config=cfg, train_years=5, rankings=clean))
    stem = oos_build.holdings_name("4w", 5, ["x_price_level"])
    c = oos_build.load_config(dict(config=cfg, train_years=5, rankings=Path(f"{stem}_s0.parquet")))
    assert oos_build.describe(c, 5).endswith(
        "sector cap 2, half-gate trend ballast, no stop-loss, the screened bundle of 1 engineered features*")
    b = oos_build.accounting(c)
    assert "100% with none breached, 50% with all three" in b and "rest sits in IEF" in b
    assert "no ballast: the whole of NAV is in the book" in oos_build.accounting({**c, "floor": "none"})


def test_features_line_reads_the_screen_record(tmp_path):
    assert oos_build.features_line(None) is None
    screen = tmp_path / "screen.json"
    keep = ["x_price_level", "f_vol_chg_12w"]
    rec = {"window": ["2006-01-01", "2015-12-31"], "kweeks": 4, "candidates": list("abcdefgh"),
           "keepers": keep, "admitted": [],
           "rule": {"t_bar": 2.0, "admit": "argmax"},
           "exam": {"primary": "top6", "rule": "argmax",
                    "compounded_pct": {"without": 4.0328, "with": 10.1611},
                    "stats": {"top6": {"diff": 0.00448, "t": 1.607, "n": 508}}}}
    screen.write_text(json.dumps(rec))
    line = oos_build.features_line(screen)
    assert line.startswith("+ engineered features*: the screen on 2006–2015 kept 2 of 8 candidates "
                           "(x_price_level, f_vol_chg_12w)")
    assert "top-6 picks earned +0.45% more per 4 weeks" in line and "(HAC t 1.61, 508 weeks)" in line
    assert "compounded at 10.16 %/yr with it against 4.03 without" in line
    assert "(v3.1): not admitted, so this line equals the clean one on this page" in line
    assert line.endswith("*The candidate ideas were written after reading the whole 2006–2024 record, "
                         "grading years included.")
    screen.write_text(json.dumps({**rec, "admitted": keep}))
    assert "admitted; the page built with it is oos_x_explorer.html" in oos_build.features_line(screen)
    assert "admitted, and this page's walk carries it" in oos_build.features_line(screen, keep)
    with pytest.raises(AssertionError):                                  # a page on some other bundle
        oos_build.features_line(screen, ["x_price_level"])


def test_stopped_names_are_read_off_the_weights_and_carried_over_hold_weeks():
    rec = {"sleeves": [["A", "B"], ["C"]], "weights": {"A": .2, "C": .2, "SPY": .6}}
    assert oos_build.stopped(rec) == ["B"]                              # B's money is in SPY
    assert oos_build.stopped({"sleeves": [["A"]], "weights": {"A": .6, "SPY": .4}}) == []


def test_render_sets_the_title_from_meta(tmp_path):
    tpl = tmp_path / "app.html"
    tpl.write_text('<title>__TITLE__</title><script id="data" type="application/json">__DATA__</script>')
    out = oos_build.render({"meta": {"title": "Selection-Window Explorer"}}, template=tpl,
                           out=tmp_path / "out.html")
    assert "<title>Selection-Window Explorer</title>" in out.read_text()
