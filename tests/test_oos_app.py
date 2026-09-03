import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "oos_build", Path(__file__).resolve().parents[1] / "app/oos/build.py")
oos_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oos_build)


def _rec(t, wk, nxt, sleeves, vals, rotated, fr=0.0, cost=0.0):
    rs = [sum(v) / len(v) for v in vals]
    book_r = sum(rs) / len(rs) - cost
    return {"t": t, "wk": wk, "nxt": nxt, "sleeves": sleeves, "vals": vals,
            "rotated": rotated, "fr": fr, "cost": cost, "g": 0.0,
            "r": 0.6 * book_r + 0.4 * fr}


def test_positions_follow_names_across_sleeves_and_reassemble_nav():
    trace = [
        _rec("d0", "w0", "w1", [["A", "B"], ["C", "D"]], [[0.10, 0.0], [0.0, 0.0]],
             rotated=[0, 1], cost=0.001, fr=0.01),
        _rec("d1", "w1", "w2", [["A", "B"], ["A", "E"]], [[0.05, 0.0], [0.05, 0.0]],
             rotated=[1], cost=0.0005),
        _rec("d2", "w2", "w3", [["F", "G"], ["A", "E"]], [[0.0, 0.0], [-0.02, 0.0]],
             rotated=[0], cost=0.0005),
    ]
    nav, pos = oos_build.build_positions(trace, ballast=0.4, cost=0.001)
    assert nav[-1] == pytest.approx(100 * (1 + trace[0]["r"]) * (1 + trace[1]["r"]) * (1 + trace[2]["r"]))
    by = {p["k"]: p for p in pos}
    assert by["A"]["n"] == [1, 2, 1] and by["A"]["s"] is None and by["A"]["b"] == "d0"
    assert by["C"]["s"] == "w1" and by["C"]["n"] == [1]
    assert by["F"]["b"] == "d2" and by["F"]["j0"] == 2
    # A's first week: 60% x 1/(2 sleeves x 2 names) x 10% on $100, less its cost share
    assert by["A"]["p"][0] == pytest.approx(100 * (0.6 * 0.25 * 0.10 - 0.6 * (0.001 / 2) / 2))
    # every week's $ splits add back to the engine's return
    for j, rec in enumerate(trace):
        names_pnl = sum(p["p"][j - p["j0"]] for p in pos if p["j0"] <= j < p["j0"] + len(p["p"]))
        assert names_pnl + nav[j] * 0.4 * rec["fr"] == pytest.approx(nav[j] * rec["r"])


def test_render_inlines_json_without_closing_the_script(tmp_path):
    tpl = tmp_path / "app.html"
    tpl.write_text('<script id="data" type="application/json">__DATA__</script>')
    out = oos_build.render({"x": "</script><b>"}, template=tpl, out=tmp_path / "out.html")
    html = out.read_text()
    assert html.count("</script>") == 1
    start = html.index(">") + 1
    assert json.loads(html[start:html.index("</script>")]) == {"x": "</script><b>"}
