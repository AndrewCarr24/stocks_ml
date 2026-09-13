"""Beta test of the built explorer (reports/champion_explorer.html) in a real
browser.

Needs playwright + chromium (not project dependencies):
    /opt/homebrew/Caskroom/miniconda/base/bin/python -m pytest tests/e2e -q
Screenshots land in $APP_SHOTS if set.
"""
import json
import os
import re
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "reports/champion_explorer.html"
SHOTS = os.environ.get("APP_SHOTS")


@pytest.fixture(scope="module")
def expected():
    """The page must show the eval's grade for the spec's walk — the numbers
    the page carries come from the same engine on the same weeks."""
    spec = json.loads((ROOT / "models/champion_spec.json").read_text())
    walk = Path(spec["procedure"]["preds"]["path"]).parent.parent
    ev = json.loads((ROOT / walk / "eval.json").read_text())
    return dict(picks=spec["procedure"]["decision"]["book_size"],
                end=f"${ev['table']['champion']['2006-2024']['terminal_100']:,.0f}",
                spy=f"${ev['table']['sp500']['2006-2024']['terminal_100']:,.0f}")


@pytest.fixture(scope="module")
def page():
    if not HTML.exists():
        pytest.skip("build the explorer first: stocks-ml app")
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1200, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(HTML.as_uri())
        pg.wait_for_selector("#chart path")
        yield pg
        assert errors == [], errors
        browser.close()


def _shot(page, name):
    if SHOTS:
        page.screenshot(path=str(Path(SHOTS) / f"{name}.png"), full_page=True)


def test_chart_draws_both_series_and_the_summary(page, expected):
    assert page.title() == "Champion Explorer"
    assert page.locator("h1").inner_text() == "Champion Explorer"
    assert page.locator("#chart path").count() == 2
    strip = page.locator(".strip").inner_text()
    assert "Procedure" in strip and "S&P 500" in strip and expected["end"] in strip and expected["spy"] in strip
    # the relative-strength panel: procedure ÷ S&P 500, dotted 1.0, ends at the summary's ratio
    assert page.locator("#ratio path").count() == 1
    rlabels = page.locator("#ratio text").all_text_contents()
    assert "1.0×" in rlabels
    end = float(expected["end"].strip("$").replace(",", "")) / float(expected["spy"].strip("$").replace(",", ""))
    shown = [float(m.group(1)) for t in rlabels for m in [re.search(r"ends (\d+\.\d\d)×", t)] if m]
    assert shown and abs(shown[0] - end) <= 0.02, (shown, end)
    page.locator("#rhit").hover()
    assert "ratio" in page.locator("#tip").inner_text()
    page.mouse.move(0, 0)
    labels = page.locator("#chart text").all_text_contents()
    assert "$1600" in labels                                  # the log grid doubles
    assert "2006" in labels
    _shot(page, "01_chart")


def test_hover_shows_that_weeks_picks_and_click_pins_them(page, expected):
    box = page.locator("#hit").bounding_box()
    x, y = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5
    page.mouse.move(x, y)
    tip = page.locator("#tip")
    assert tip.is_visible()
    assert tip.locator(".picks li a.tk").count() == expected["picks"]
    assert "Click to pin" in tip.inner_text()
    page.mouse.click(x, y)
    assert "pinned" in tip.get_attribute("class")
    book = page.locator("#book .sleeve")
    assert book.count() == 4
    assert page.locator("#book .sleeve.rot").count() >= 1     # two when a stale sleeve catches up
    _shot(page, "02_pinned")


def test_clicking_a_company_opens_its_history(page):
    first = page.locator("#tip .picks li a.tk").first
    ticker = first.inner_text().strip()
    first.click()
    page.wait_for_url(f"**#/co/{ticker}")
    assert page.locator(".co-head .tk").inner_text().strip() == ticker
    rows = page.locator("table.list tr").count() - 1
    assert rows >= 1
    assert page.locator("#tlsvg rect.bar").count() == rows
    stats = page.locator(".stats").inner_text().lower()
    assert "won / lost" in stats and "holds" in stats
    assert "−" in stats or "+" in stats  # signed dollars survive the charset
    _shot(page, "03_company")
    page.locator("#tlsvg rect.bar").first.hover()
    assert page.locator("#tltip").is_visible()
    page.locator(".crumbs a").click()
    page.wait_for_selector("#chart path")
    assert page.locator("#tip.pinned").is_visible()  # the pinned week survives the round trip


def test_company_search_navigates(page):
    page.goto(HTML.as_uri() + "#/")
    page.wait_for_selector("#goto")
    ticker = page.evaluate("document.querySelector('#tks option').value")
    page.locator("#goto").fill(ticker)
    page.locator("#goto").dispatch_event("change")
    page.wait_for_url("**#/co/*")
    assert page.locator(".co-head .tk").inner_text().strip() == ticker
    _shot(page, "04_search")


def test_the_notes_quote_the_eval_and_the_store(page):
    page.goto(HTML.as_uri() + "#/")
    page.wait_for_selector("#note p")
    note = page.locator("#note").inner_text()
    assert "Leak audit PASS" in note and "stocks-ml app" in note and "sharadar_world2000_nominal_dl" in note


def test_unknown_company_falls_back_to_the_chart(page):
    page.goto(HTML.as_uri() + "#/co/NOPE")
    page.wait_for_selector("#chart path")
    assert page.locator("#chart path").count() == 2


def test_dark_theme_paints_its_own_ground(page):
    page.emulate_media(color_scheme="dark")
    page.goto(HTML.as_uri())
    page.wait_for_selector("#chart path")
    bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert bg == "rgb(26, 26, 25)"
    _shot(page, "05_dark")
    page.emulate_media(color_scheme="light")
