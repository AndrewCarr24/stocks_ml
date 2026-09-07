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
    assert "| Features |" in card and "generated bundle of 40" in card and "no asterisk" in card
    for f in spec["features"]:
        assert f"{f} = `{spec['formulas'][f]}`" in card
    spec["features"] = ["x_cash_runway", "f_vol_chg_12w"]      # a screened bundle of ideas
    assert "screened bundle of 2: x_cash_runway, f_vol_chg_12w" in render(spec) and "asterisk" in render(spec)
    spec["features"] = []
    assert "no screened bundle" in render(spec)
