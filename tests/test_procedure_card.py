import json
from pathlib import Path

from stocks_ml.procedure_card import SPEC_PATH, render


def test_render_from_champion_spec():
    spec = json.loads(Path(SPEC_PATH).read_text())
    card = render(spec, today="2026-09-01")
    assert "trailing 5 years" in card
    assert "top-6" in card
    assert "K=16" in card
    assert "70% book / 30% ballast" in card
    assert "no stop" in card
    assert "max 2/sector" in card
    assert "never on a calendar" in card
    assert "| floor |" in card and "Sharpe" in card
    assert "2024-07-19+ is holdout" in card


def test_render_tracks_spec_changes():
    spec = json.loads(Path(SPEC_PATH).read_text())
    spec["training_window_years"] = 2
    assert "trailing 2 years" in render(spec)


def test_render_lists_the_bundle_with_its_formulas():
    spec = json.loads(Path(SPEC_PATH).read_text())
    card = render(spec)
    # the champion carries no bundle since 2026-09-11 (the split-leak bundle retired)
    assert spec["features"] == [] and spec["formulas"] == {}
    assert "| Features |" in card and "no screened bundle" in card
    assert "level features on the nominal basis" in card and "final print (last_print)" in card
    spec["features"] = ["g_00", "g_01"]                          # a generated bundle with formulas
    spec["formulas"] = {"g_00": "(r_close*f_a)", "g_01": "log(f_b)"}
    card = render(spec)
    assert "generated bundle of 2" in card and "no asterisk" in card
    for f in spec["features"]:
        assert f"{f} = `{spec['formulas'][f]}`" in card
    spec["features"] = ["x_cash_runway", "f_vol_chg_12w"]      # a screened bundle of ideas
    assert "screened bundle of 2: x_cash_runway, f_vol_chg_12w" in render(spec) and "asterisk" in render(spec)
