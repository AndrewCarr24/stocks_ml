import json
from pathlib import Path

from stocks_ml.procedure_card import SPEC_PATH, render


def test_render_from_champion_spec():
    spec = json.loads(Path(SPEC_PATH).read_text())
    card = render(spec, today="2026-09-01")
    assert f"trailing {spec['training_window_years']} years" in card
    assert f"top-{spec['strategy']['book_size']}" in card
    assert f"{spec['horizon']['label']}: " in card
    assert f"{spec['procedure']['model']['label']} / {spec['procedure']['model']['train_years']}-year window" in card
    assert "K=16" in card
    assert spec["ballast"]["mix"] in card                      # whatever the procedure wrote
    assert "| Decided by | `stocks-ml procedure`" in card
    assert f"floor {spec['procedure']['decision']['floor']}" in card
    assert "no stop" in card
    cap = spec["strategy"]["sector_cap"]
    assert (f"max {cap}/sector" if cap else "no sector cap") in card
    assert "never on a calendar" in card
    assert "| floor |" in card and "Sharpe" in card
    assert "2024-07-19+ is holdout" in card


def test_render_tracks_spec_changes():
    spec = json.loads(Path(SPEC_PATH).read_text())
    spec["training_window_years"] = 2
    assert "trailing 2 years" in render(spec)


def test_render_names_the_features():
    spec = json.loads(Path(SPEC_PATH).read_text())
    card = render(spec)
    # since 2026-09-19 the champion's one extra column is the split-consistent dollar volume (a correction, not a bundle)
    assert spec["features"] == ["x_dollar_vol"] and spec["drop_features"] == ["f_dollar_vol"]
    assert "| Features |" in card and "no screened bundle" in card and "split-consistent" in card
    assert "level features on the nominal basis" in card and "final print (last_print)" in card
    spec["features"] = ["x_cash_runway", "f_vol_chg_12w"]      # extra columns carry the asterisk
    assert "extra columns x_cash_runway, f_vol_chg_12w" in render(spec) and "asterisk" in render(spec)


def test_readme_champion_block_is_rendered_from_spec_and_eval(tmp_path):
    import pytest

    from stocks_ml import procedure_card as pc
    from stocks_ml.selection import LABELS_4W
    spec = json.loads(Path(SPEC_PATH).read_text())
    spec["character"] = "It buys calmer names."
    walk = tmp_path / "walk"
    (walk / "select").mkdir(parents=True)
    spec["procedure"]["preds"]["path"] = str(walk / "select" / "preds.parquet")
    m = {"terminal_100": 844.2, "cagr_pct": 28.24, "sharpe": 0.922, "max_dd": 0.383, "n_weeks": 446}
    row = {w: m for w in pc.SPANS}
    ev = {"label": "champion", "evaluated_at": "2026-09-14 19:00:00",
          "table": {"champion": {**row, "paired_t_vs_sp500": {w: 1.7 for w in pc.SPANS}},
                    "sp500": {w: {**m, "terminal_100": 316.0, "cagr_pct": 14.3, "max_dd": 0.32} for w in pc.SPANS}},
          "confidence": {"2016-2024": {"excess_cagr_vs_sp500_pct": {"point": 12.2, "nested_95": [-4.8, 33.1]},
                                       "p_excess_positive_nested": 0.92}},
          "falsification": {"2016-2024": {"t": 0.76}, "verdict": "not rejected"},
          "leak_audit": {"VERDICT": "PASS"}, "charts": ["reports/champion_vs_sp500_2016_2024.png"]}
    block = pc.champion_block(spec, ev)
    assert "| champion | $844, +28.2% | 0.92, 38% |" in block and "| sp500 | $316, +14.3%" in block
    assert "+12.2%/yr" in block and "-4.8 to +33.1" in block and "92% of the resampled" in block
    assert "t +0.76 (not rejected" in block and "Leak audit PASS" in block and "It buys calmer names." in block
    assert LABELS_4W[spec["horizon"]["label"]] in block and f"top-{spec['strategy']['book_size']} per sleeve" in block
    assert "![Growth of $100, 2016-2024, out of sample](reports/champion_vs_sp500_2016_2024.png)" in block
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n\n{pc.BLOCK_BEGIN}\nPLACEHOLDER\n{pc.BLOCK_END}\n\ntail\n")
    assert pc.write_readme_block(spec, readme) is False                 # no eval.json yet: untouched
    assert "PLACEHOLDER" in readme.read_text()
    (walk / "eval.json").write_text(json.dumps(ev))
    assert pc.write_readme_block(spec, readme) is True
    text = readme.read_text()
    assert "PLACEHOLDER" not in text and "| champion | $844" in text and text.endswith("tail\n")
    assert pc.write_readme_block(spec, readme) is True and readme.read_text() == text   # idempotent
    (tmp_path / "nomarkers.md").write_text("x")
    with pytest.raises(RuntimeError):
        pc.write_readme_block(spec, tmp_path / "nomarkers.md")
