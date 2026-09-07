"""Beta test of the built explorers (reports/{oos,oos_x,select,champion}_explorer.html)
in a real browser.

Needs playwright + chromium (not project dependencies):
    /opt/homebrew/Caskroom/miniconda/base/bin/python -m pytest tests/e2e -q
Screenshots land in $OOS_SHOTS if set.
"""
import os
import re
from collections import namedtuple
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

REPORTS = Path(__file__).resolve().parents[2] / "reports"
SHOTS = os.environ.get("OOS_SHOTS")
Page = namedtuple("Page", "file title picks end spy scale")
PAGES = [Page("oos_explorer.html", "Out-of-Sample Explorer", 6, "$521", "$316", "linear"),
         Page("oos_x_explorer.html", "Out-of-Sample Explorer, with the bundle", 3, "$1,379", "$316", "log"),
         Page("select_explorer.html", "Selection-Window Explorer", 3, "$1,424", "$623", "log"),
         Page("champion_explorer.html", "Champion Explorer", 6, "$3,498", "$608", "log")]


@pytest.fixture(scope="module", params=PAGES, ids=[p.file.rsplit("_", 1)[0] for p in PAGES])
def spec(request):
    return request.param


@pytest.fixture(scope="module")
def HTML(spec):
    return REPORTS / spec.file


@pytest.fixture(scope="module")
def page(HTML):
    if not HTML.exists():
        pytest.skip("build the explorers first: .venv/bin/python app/oos/build.py")
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


def test_chart_draws_both_series_and_the_summary(page, spec):
    assert page.title() == spec.title
    assert page.locator("h1").inner_text() == spec.title
    assert page.locator("#chart path").count() == 2
    strip = page.locator(".strip").inner_text()
    assert "Procedure" in strip and "S&P 500" in strip and spec.end in strip and spec.spy in strip
    # the relative-strength panel: procedure ÷ S&P 500, dotted 1.0, ends at the summary's ratio
    assert page.locator("#ratio path").count() == 1
    rlabels = page.locator("#ratio text").all_text_contents()
    assert "1.0×" in rlabels
    # the label is the unrounded ratio; the strip's whole dollars can move it by a hundredth
    end = float(spec.end.strip("$").replace(",", "")) / float(spec.spy.strip("$").replace(",", ""))
    shown = [float(m.group(1)) for t in rlabels for m in [re.search(r"ends (\d+\.\d\d)×", t)] if m]
    assert shown and abs(shown[0] - end) <= 0.02, (shown, end)
    page.locator("#rhit").hover()
    assert "ratio" in page.locator("#tip").inner_text()
    page.mouse.move(0, 0)
    labels = page.locator("#chart text").all_text_contents()
    assert ("$1600" in labels) == (spec.scale == "log")      # log grid doubles; linear steps
    assert str(int(page.evaluate("JSON.parse(document.getElementById('data').textContent).meta.lo").split("-")[0])) in labels
    _shot(page, f"01_chart_{spec.file.split('_')[0]}")


def test_hover_shows_that_weeks_picks_and_click_pins_them(page, spec):
    box = page.locator("#hit").bounding_box()
    x, y = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5
    page.mouse.move(x, y)
    tip = page.locator("#tip")
    assert tip.is_visible()
    assert tip.locator(".picks li a.tk").count() == spec.picks
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
    assert "\u2212" in stats or "+" in stats  # signed dollars survive the charset
    _shot(page, "03_company")
    page.locator("#tlsvg rect.bar").first.hover()
    assert page.locator("#tltip").is_visible()
    page.locator(".crumbs a").click()
    page.wait_for_selector("#chart path")
    assert page.locator("#tip.pinned").is_visible()  # the pinned week survives the round trip


def test_a_stopped_name_is_marked_in_the_book(page, spec, HTML):
    page.goto(HTML.as_uri() + "#/")
    page.wait_for_selector("#chart path")
    if not page.evaluate("JSON.parse(document.getElementById('data').textContent).meta.config.stop"):
        pytest.skip("this configuration has no stop-loss")
    # the middle of a run of weeks with a parked name, so a 1-px miss still lands on one
    j = page.evaluate("""(() => { const D = JSON.parse(document.getElementById('data').textContent);
        return D.weeks.findIndex((w, j) => j > 0 && j + 1 < D.weeks.length && w.stop.length
            && D.weeks[j - 1].stop.length && D.weeks[j + 1].stop.length); })()""")
    assert j > 0
    x = page.evaluate(f"""(() => {{ const s = document.getElementById('chart'), r = s.getBoundingClientRect();
        return r.left + s._geom.xs[{j}] / 1000 * r.width; }})()""")
    box = page.locator("#hit").bounding_box()
    page.mouse.click(x, box["y"] + box["height"] * 0.5)
    assert "stopped · in SPY" in page.locator("#book").inner_text()
    _shot(page, "05_stopped")


def test_company_search_navigates(page, HTML):
    page.goto(HTML.as_uri() + "#/")
    page.wait_for_selector("#goto")
    ticker = page.evaluate("document.querySelector('#tks option').value")
    page.locator("#goto").fill(ticker)
    page.locator("#goto").dispatch_event("change")
    page.wait_for_url("**#/co/*")
    assert page.locator(".co-head h1").inner_text().strip()


def test_unknown_company_falls_back_to_the_chart(page, HTML):
    page.goto(HTML.as_uri() + "#/co/NOPE")
    page.wait_for_selector("#chart path")
    assert page.locator("#chart path").count() == 2


def test_dark_theme_paints_its_own_ground(page, HTML):
    page.emulate_media(color_scheme="dark")
    page.goto(HTML.as_uri())
    page.wait_for_selector("#chart path")
    bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
    assert bg == "rgb(26, 26, 25)"
    _shot(page, "04_dark")
    page.emulate_media(color_scheme="light")
