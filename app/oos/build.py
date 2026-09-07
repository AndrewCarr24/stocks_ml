"""Build the explorers: a selection procedure's chosen configuration, replayed
week by week as a self-contained page.

  oos       the nested OOS test: procedure v3 (every week of 2006-2015, no
            sampled layer) selected on 2006-2015 alone, its choice graded
            2016 -> 2024-07-19 exclusive             -> reports/oos_explorer.html
            The notes carry the "+ engineered features" line from the screen
            stage's record; the bundle the screen admitted has its own page:
  oos_x     the same nested test on the walk WITH the screened bundle (rule
            v3.1): the cascade re-run with it, its choice graded on the same
            weeks                                    -> reports/oos_x_explorer.html
  select    what the mechanical cascade froze when run on the whole selection
            window (2006-01 -> 2024-07-18) under procedure v2, graded on that
            same window: in-sample for the selection, holdout untouched
                                                     -> reports/select_explorer.html
  champion  the champion as deployed (models/champion_spec.json, the generated
            bundle included) on its recorded basis: the OpenFE arm v3's walk
            (2006-01-06 ->), ranks before the holdout, graded 2006-01 -> 2024-07
                                                     -> reports/champion_explorer.html

The replay goes through selection.simulate itself (with its trace hook), so
the app's curve is the engine's curve — the live ledger's rules, fills at
Monday's open; this file only strings the engine's per-name attribution into
positions and inlines the result into app.html.

    .venv/bin/python app/oos/build.py [oos|oos_x|select|champion]    # default: all
"""
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from stocks_ml.selection import HORIZONS, holdings_name, metrics, price_frames, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT / "data/sharadar_world2000"
TICKERS = ROOT / "data/r5_live/sharadar_tickers.parquet"
TEMPLATE = Path(__file__).with_name("app.html")
NESTED = ROOT / "data/experiments/nested3_v2"        # v1's caches plus the v3.1 screen and the bundle walk
CHAMPION = ROOT / "data/experiments/champion_2006_2024"
GENERATED = ROOT / "data/experiments/openfe_v3_2006_2015"   # the generated bundle's walk (the champion's since 2026-09-06)
FROZEN = CHAMPION / "frozen_config.json"
SPEC = ROOT / "models/champion_spec.json"
BUNDLE_PAGE = "oos_x_explorer.html"

NESTED_VERDICT = ("Ledger rows nested3_frozen_config and nested3_verdict (2026-09-04). Procedure "
                  "v2 (nested2_verdict_amended) let two layers read samples of the same years; "
                  "its book layer chose top-10 on 29 weeks and graded $463 vs SPY $307 on 443 "
                  "weeks — $469 vs $316 on this page's weeks — and a rerun of its sample chose "
                  "top-3 instead. v3 reads every week: top-6 won, top-3 came last. Under rule "
                  "v3.1 (2026-09-05, nested3_features_v2_verdict) the screen admitted the "
                  "engineered bundle; re-run with it, the cascade chose top-3 with the half-gate "
                  f"ballast and graded $1,379 on these weeks — {BUNDLE_PAGE}.")
NESTED_X_VERDICT = ("Ledger rows nested3_features_v2_frozen_config and nested3_features_v2_verdict "
                    "(2026-09-05). Rule v3.1 admits the screen's keepers iff the top-6 book "
                    "compounds faster with them on the selection window — 10.16 vs 4.03 %/yr on "
                    "508 weeks of 2006–2015 (the earlier bar, a paired HAC t of 2, read 1.61 and "
                    "would have kept them out). With the bundle the cascade chose top-3 (book "
                    "evidence 14.7 / 11.5 / 8.6 %/yr for top-3 / 6 / 10) and the half-gate "
                    "ballast; graded once: $1,379 against the clean page's $521 and SPY's $316. "
                    "The champion carried this bundle from 2026-09-05 to 2026-09-06, when the "
                    "generated bundle replaced it (champion_explorer.html); its book and ballast "
                    "stayed top-6, 70/30.")
SELECTED = ("Ledger row select_2006-01-01_2024-07-18 (2026-09-03), a procedure v2 cascade: its "
            "window and book layers read samples, and it has not been re-run under v3 (every "
            "week). The campaign's record is reports/fill_basis_regrade.md. The owner-declared "
            "champion is a different configuration (top-6, 70/30 ballast, sector cap 2, no "
            "stop): $1,553 on this window as deployed before any bundle; with the generated "
            "bundle, champion_explorer.html.")
CHAMPION_VERDICT = ("The champion as deployed (models/champion_spec.json, PROCEDURE.md): top-6 in "
                    "four sleeves, 70/30 trend ballast, sector cap 2, no stop, the model trained on "
                    "the panel's f_ columns plus the generated bundle of forty features (formulas "
                    "over the panel's raw inputs, chosen on 2006–2015 alone by the OpenFE arm v3; "
                    "adopted 2026-09-06 in place of the screened bundle of seven ideas, whose "
                    "candidates had been written after reading the whole record). In-sample: the "
                    "book, ballast and cap were chosen after reading these years (the 4-week "
                    "rebuild campaign, AGENTS.md), and the bundle's selection read 2006–2015; the "
                    "holdout (2024-07-19 →) is untouched. Before any bundle the same configuration "
                    "graded $1,553 (+15.9%/yr, SR 0.72, DD 62%); with the seven ideas $3,058 "
                    "(+20.2%/yr, SR 0.85, DD 59%)*. The nested test's cascade, given the ideas, "
                    f"preferred top-3 and the half-gate ballast ({BUNDLE_PAGE}); not adopted.")


def spec_config(spec=SPEC):
    """The champion's engine config, read from the spec (the live job mirrors
    the same fields, tests/test_r5.py)."""
    s = json.loads(spec.read_text())
    st, book_pct = s["strategy"], int(s["ballast"]["mix"].split("%")[0])
    return dict(horizon=s["horizon"]["label"].split("_")[1], book=st["book_size"],
                cap=st["sector_cap"], stop=st["stop_loss"], floor=f"{book_pct}/{100 - book_pct}",
                features=list(s["features"]), train_years=s["training_window_years"])


_spec = spec_config()
VARIANTS = {
    "oos": dict(
        rankings=NESTED / "holdings_4w_5y_s0.parquet",
        config=NESTED / "frozen_config.json", train_years=5, screen=NESTED / "screen.json",
        lo="2016-01-01", hi="2024-07-19", out=ROOT / "reports/oos_explorer.html",
        title="Out-of-Sample Explorer", crumb="Out-of-sample chart", scale="linear",
        intro="The selection procedure was run on 2006–2015 alone, reading every week of it; "
              "the configuration it chose ({config}) was then frozen and graded on {span} "
              "against the S&P 500 (SPY), costs included, $100 start.",
        verdict=NESTED_VERDICT),
    "oos_x": dict(
        rankings=NESTED / "holdings_4w_5y_xeeaf48_s0.parquet",
        config=NESTED / "frozen_config_x.json", train_years=5, screen=NESTED / "screen.json",
        lo="2016-01-01", hi="2024-07-19", out=ROOT / "reports/oos_x_explorer.html",
        title="Out-of-Sample Explorer, with the bundle", crumb="Out-of-sample chart (bundle)",
        scale="log",
        intro="The selection procedure was run on 2006–2015 alone, its feature screen admitted "
              "the engineered bundle, and the cascade re-run with it chose {config}; that "
              "configuration was then frozen and graded on {span} against the S&P 500 (SPY), "
              "costs included, $100 start.",
        verdict=NESTED_X_VERDICT),
    "select": dict(
        rankings=CHAMPION / "holdings_4w_5y_s0.parquet",
        config=FROZEN, train_years=5,
        lo="2006-01-01", hi="2024-07-18", out=ROOT / "reports/select_explorer.html",
        title="Selection-Window Explorer", crumb="Selection-window chart", scale="log",
        intro="Run on the whole selection window, the procedure chose {config}. This is "
              "that configuration graded on the same window, {span}, against the S&P 500 "
              "(SPY), costs included, $100 start — in-sample for the selection, which saw "
              "every one of these years; the holdout (2024-07-19 →) is untouched.",
        verdict=SELECTED,
        ledger="select_2006_2024_mechanical_pick_graded",   # the ledger row this grade lands in
        note="{span}, the configuration the mechanical cascade (procedure v2) froze on this "
             "window (ledger select_2006-01-01_2024-07-18) graded on it by selection.simulate: "
             "{s} vs SPY {y}; in-sample for the selection. {out} (app/oos/build.py select)"),
    "champion": dict(
        # the generated bundle's walk (the stem's hash names the spec's features); the seven
        # ideas' record stays on CHAMPION / holdings_4w_5y_xeeaf48_s0 (ledger
        # champion_bundle_2006_2024_graded), history since 2026-09-06
        rankings=GENERATED / f"{holdings_name(_spec['horizon'], _spec['train_years'], _spec['features'])}_s0.parquet",
        config=_spec, train_years=_spec["train_years"], screen=None,
        # the recorded basis (ops/live_emulation.py): ranks before the holdout, every credited
        # week from 2006 counted — including the one the last pre-holdout rank date earns
        lo="2006-01-01", hi="2025-01-01", ranks_before="2024-07-18",
        out=ROOT / "reports/champion_explorer.html",
        title="Champion Explorer", crumb="Champion chart", scale="log",
        intro="The champion as deployed — {config} — graded on {span} against the S&P 500 "
              "(SPY), costs included, $100 start. In-sample: its book, ballast and cap were "
              "chosen after reading these years and its bundle was chosen on 2006–2015; the "
              "holdout (2024-07-19 →) is untouched.",
        verdict=CHAMPION_VERDICT,
        ledger="champion_generated_2006_2024_graded",
        note="{span}, the champion as deployed (models/champion_spec.json: top-6, 70/30, sector "
             "cap 2, no stop, the generated bundle adopted 2026-09-06) graded by selection.simulate "
             "on its recorded basis (the OpenFE arm v3's walk from 2006-01-06, ranks before the "
             "holdout): {s} vs SPY {y}; before any bundle $1,553 (+15.9%/yr, SR 0.72, DD 62%), "
             "with the seven ideas $3,058 (+20.2%/yr, SR 0.85, DD 59%)* "
             "(champion_bundle_2006_2024_graded). In-sample for the book/ballast/cap choices and "
             "for the bundle's selection window; the holdout is untouched. {out} "
             "(app/oos/build.py champion)"),
}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight", 10: "ten"}
FLOOR_WORDS = {"none": "no trend ballast", "halfgate": "half-gate trend ballast"}


def load_config(v):
    """The variant's engine config, read from the cascade's frozen choice (or
    the spec). The page's walk must be the one that saw the config's features:
    the holdings stem carries the bundle's hash."""
    c = v["config"]
    if isinstance(c, Path):
        c = json.loads(c.read_text())
    assert c.get("train_years", v["train_years"]) == v["train_years"], c
    features = list(c.get("features") or [])
    stem = holdings_name(c["horizon"], v["train_years"], features)
    assert v["rankings"].name == f"{stem}_s0.parquet", (v["rankings"].name, stem, features)
    return dict(horizon=c["horizon"], book=c["book"], cap=c["cap"], stop=c["stop"],
                floor=c["floor"], features=features)


def features_line(screen, features=()):
    """The "+ engineered features" line of the notes, from the screen stage's
    record (feature_screen.py, rule v3.1). Asterisk: the candidate ideas were
    written after reading the whole 2006-2024 record, grading years included,
    so the line is reported beside the clean result and never folded into it.
    `features` is the page's own bundle: empty on a clean page."""
    if screen is None:
        return None
    s = json.loads(screen.read_text())
    ex = s["exam"]
    t6, cmp = ex["stats"][ex["primary"]], ex["compounded_pct"]
    if not s["admitted"]:
        tail = "not admitted, so this line equals the clean one on this page"
    elif features:
        assert sorted(features) == sorted(s["admitted"]), (features, s["admitted"])
        tail = "admitted, and this page's walk carries it"
    else:
        tail = f"admitted; the page built with it is {BUNDLE_PAGE}"
    return (f"+ engineered features*: the screen on {s['window'][0][:4]}–{s['window'][1][:4]} "
            f"kept {len(s['keepers'])} of {len(s['candidates'])} candidates "
            f"({', '.join(s['keepers'])}) and examined them as one bundle in a second walk "
            f"with them. The top-{ex['primary'][3:]} picks earned {t6['diff']:+.2%} more per "
            f"{s['kweeks']} weeks with the bundle (HAC t {t6['t']:.2f}, {t6['n']} weeks) and "
            f"the top-{ex['primary'][3:]} book compounded at {cmp['with']:.2f} %/yr with it "
            f"against {cmp['without']:.2f} without, costs included — the rule admits on that "
            f"comparison (v3.1): {tail}. *The candidate ideas were written after reading the "
            f"whole 2006–2024 record, grading years included.")


def describe(config, train_years):
    """'4-week horizon, 5-year training window, top-3 in four staggered sleeves,
    no sector cap, 60/40 trend ballast, -25% stop-loss'"""
    k = HORIZONS[config["horizon"]]["kweeks"]
    cap = f"sector cap {config['cap']}" if config["cap"] else "no sector cap"
    stop = f"{config['stop']:+.0%} stop-loss" if config["stop"] else "no stop-loss"
    floor = FLOOR_WORDS.get(config["floor"], f"{config['floor']} trend ballast")
    s = (f"{k}-week horizon, {train_years}-year training window, top-{config['book']} in "
         f"{WORDS[k]} staggered sleeves, {cap}, {floor}, {stop}")
    if config.get("features"):
        s += ", " + bundle_words(config["features"])
    return s


def bundle_words(features):
    """The page's bundle in a phrase: the generated bundle (g_ columns, chosen
    on the selection window alone — no asterisk) or the screened bundle of
    ideas (asterisk: written after reading the whole record)."""
    if all(f.startswith("g_") for f in features):
        return f"the generated bundle of {len(features)} features"
    return f"the screened bundle of {len(features)} engineered features*"


def ballast_words(floor):
    """How the floor menu's entry parks the money (selection.floor_split)."""
    if floor == "none":
        return "no ballast: the whole of NAV is in the book"
    if floor == "halfgate":
        return ("the book's share of NAV is 100% less half the share of breached SPY trailing "
                "means (30/40/52 weeks) — 100% with none breached, 50% with all three — and "
                "the rest sits in IEF")
    ballast = 100 - int(floor.split("/")[0])
    return (f"{ballast}% of NAV sits in the trend ballast, SPY shifted to IEF one third per "
            f"breached trailing mean (30/40/52 weeks)")


def accounting(config):
    """The weekly-accounting paragraph of the notes, for this configuration."""
    k = HORIZONS[config["horizon"]]["kweeks"]
    cap = f" (at most {WORDS[config['cap']]} per sector)" if config["cap"] else ""
    s = (f"Weekly accounting, the live job's rules: each Friday close the model ranks the "
         f"members; one of {WORDS[k]} sleeves is replaced by the top {WORDS[config['book']]}"
         f"{cap}; the orders fill at the next session's open, 5 bp a side on every trade "
         f"(rebalances under 0.5% of NAV skipped); every name is held {WORDS[k]} weeks; "
         f"{ballast_words(config['floor'])}.")
    if config["stop"]:
        s += (f" A name that has fallen {abs(config['stop']):.0%} from its rotation close "
              f"is sold and its sleeve's money parked in SPY until the sleeve rotates; the "
              f"book shows it as stopped, and the parked money's gain or loss counts as "
              f"ballast.")
    return s + (" Dollar figures are per $100 invested at the start; a company's \"won or "
                "lost\" is its share of the week's NAV change, fees included.")


def build_positions(trace):
    """NAV path and per-name positions from simulate's trace, on a $100
    start at the first credited week.

    The engine's trace already attributes each week's NAV change to the
    names that earned it (`pnl`, fees included, sums to the change) and
    gives each name's price return over the part of the week it was held
    (`vals`). A position is a name's unbroken run in the book (any sleeve);
    `n` counts the sleeves holding it each week, `v` is the stock's weekly
    return, `p` the $ won or lost. The week a name leaves the book it is
    sold at Monday's open: that week's gap and fee close the position as a
    final entry with n = 0. A stopped name stays in its sleeve (the engine
    keeps it listed while its money sits in SPY), so its later weeks carry
    zeros until the sleeve rotates."""
    scale = 100.0 / (trace[0]["nav"] / (1 + trace[0]["r"]))
    nav, positions, open_pos = [100.0], [], {}
    for j, rec in enumerate(trace):
        assert abs(sum(rec["pnl"].values()) - (rec["nav"] - nav[-1] / scale)) < 1e-6 * rec["nav"]
        nav.append(rec["nav"] * scale)
        book = {}
        for names in rec["sleeves"]:
            for n in names:
                book[n] = book.get(n, 0) + 1
        def add(p, n, held):
            p["n"].append(held)
            p["v"].append(rec["vals"].get(n, 0.0))
            p["p"].append(rec["pnl"].get(n, 0.0) * scale)
        for n in [n for n in open_pos if n not in book]:
            p = open_pos.pop(n)
            p["s"] = rec["wk"]
            add(p, n, 0)
            positions.append(p)
        for n in book:
            p = open_pos.get(n)
            if p is None:
                p = open_pos[n] = {"k": n, "j0": j, "b": rec["t"], "s": None,
                                   "n": [], "v": [], "p": []}
            add(p, n, book[n])
    positions.extend(open_pos.values())
    positions.sort(key=lambda p: (p["j0"], p["k"]))
    return nav, positions


def stopped(rec):
    """Names in the book whose sleeve money is parked in SPY: the engine drops
    a stopped name from the target weights but keeps it in the sleeve."""
    return sorted({n for sl in rec["sleeves"] for n in sl if n not in rec["weights"]})


def replay(v, config):
    """The engine's trace over the page's window: rank dates before
    `ranks_before` (default: the window's end) drive the book, and the
    credited weeks in [lo, hi) are kept."""
    from stocks_ml.data.store import DataStore
    world = DataStore(WORLD)
    prices, mem = world.read("prices"), world.read("membership")
    smap = dict(mem.dropna(subset=["sector"]).drop_duplicates("ticker")[["ticker", "sector"]].values)
    daily = prices.sort_values("date")
    ctx = SimpleNamespace(smap=smap, members={}, **price_frames(
        daily.pivot(index="date", columns="ticker", values="close").sort_index(),
        daily.pivot(index="date", columns="ticker", values="open").sort_index()))
    h = pd.read_parquet(v["rankings"]).rename(columns={"tickers": "top15"})
    h["week"] = pd.to_datetime(h.week)
    lo, hi = pd.Timestamp(v["lo"]), pd.Timestamp(v["hi"])
    ranks_before = pd.Timestamp(v.get("ranks_before", v["hi"]))
    trace = []
    rets = simulate(ctx, h[h.week < ranks_before].sort_values("week"), config["horizon"],
                    config["book"], config["cap"], config["stop"], config["floor"], trace=trace)
    trace = [x for x in trace if lo <= x["nxt"] < hi]
    spy = ctx.wret["SPY"].reindex([x["nxt"] for x in trace])
    return trace, rets, spy, smap


def assemble(v, config, trace, rets, spy, smap, names):
    nav, positions = build_positions(trace)
    tick = sorted({n for p in positions for n in [p["k"]]})
    idx = {t: i for i, t in enumerate(tick)}
    iso = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")  # noqa: E731
    lo, hi = pd.Timestamp(v["lo"]), pd.Timestamp(v["hi"])
    weeks, parked = [], []
    for x in trace:
        if x["weights"]:        # a hold week (no weights) keeps the book and its parked names
            parked = [idx[n] for n in stopped(x)]
        weeks.append({"t": iso(x["t"]),
                      "pick": [idx[n] for c in x["rotated"] for n in x["sleeves"][c]],
                      "book": [[idx[n] for n in sl] for sl in x["sleeves"]],
                      "stop": parked, "g": x["g"]})
    spy_nav = list(100 * np.cumprod(1 + spy.fillna(0).values))
    grade = lambda s: {k: metrics(s, lo, hi)[k2] for k, k2 in  # noqa: E731
                       [("end", "terminal_100"), ("cagr", "cagr_pct"),
                        ("sr", "sharpe"), ("dd", "max_dd"), ("n", "n_weeks")]}
    span = f"{iso(lo)[:7]} → {iso(trace[-1]['nxt'])[:7]}"
    return {
        "meta": {"built": str(date.today()), "config": config, "lo": iso(lo), "hi": iso(hi),
                 "first_pick": iso(trace[0]["t"]), "last_pick": iso(trace[-1]["t"]),
                 "title": v["title"], "crumb": v["crumb"], "scale": v["scale"],
                 "intro": v["intro"].format(config=describe(config, v["train_years"]), span=span),
                 "book_word": WORDS[config["book"]], "accounting": accounting(config),
                 "verdict": v["verdict"],
                 "features": features_line(v.get("screen"), config.get("features"))},
        "summary": {"strategy": grade(rets), "spy": grade(spy)},
        "dates": [iso(trace[0]["wk"])] + [iso(x["nxt"]) for x in trace],
        "nav": [round(v, 3) for v in nav],
        "spy": [100.0] + [round(v, 3) for v in spy_nav],
        "weeks": weeks,
        "tickers": [{"t": t, "name": names.get(t, {}).get("name", t),
                     "sector": smap.get(t, "n/a"),
                     "industry": names.get(t, {}).get("industry", "")} for t in tick],
        "positions": [{"k": idx[p["k"]], "j0": p["j0"], "b": iso(p["b"]),
                       "s": iso(p["s"]) if p["s"] is not None else None, "n": p["n"],
                       "v": [round(v, 5) for v in p["v"]],
                       "p": [round(v, 4) for v in p["p"]]} for p in positions],
    }


def render(data, out, template=TEMPLATE):
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = template.read_text()
    assert html.count("__DATA__") == 1
    title = data.get("meta", {}).get("title")
    if title:
        html = html.replace("<title>__TITLE__</title>", f"<title>{title}</title>")
    out.write_text(html.replace("__DATA__", payload))
    return out


def build(name):
    v = VARIANTS[name]
    config = load_config(v)
    trace, rets, spy, smap = replay(v, config)
    t = pd.read_parquet(TICKERS).drop_duplicates("ticker").set_index("ticker")
    names = {k: {"name": r["name"], "industry": r["industry"]} for k, r in t.iterrows()}
    data = assemble(v, config, trace, rets, spy, smap, names)
    out = render(data, v["out"])
    s, y = data["summary"]["strategy"], data["summary"]["spy"]
    print(f"{out}: {describe(config, v['train_years'])}; {len(data['weeks'])} weeks, "
          f"{len(data['tickers'])} tickers, {len(data['positions'])} positions, "
          f"{sum(bool(w['stop']) for w in data['weeks'])} weeks with a stopped name; "
          f"${s['end']:,.0f} ({s['cagr']:+.1f}%/yr, SR {s['sr']:.2f}, DD {s['dd']:.0%}) "
          f"vs SPY ${y['end']:,.0f} ({y['cagr']:+.1f}%/yr, SR {y['sr']:.2f}, DD {y['dd']:.0%})")
    return data


def record(name, data):
    """The page's grade as a ledger row (upserted), for variants that carry one."""
    from stocks_ml.models.trials import record_trials
    v, s, y = VARIANTS[name], data["summary"]["strategy"], data["summary"]["spy"]
    line = lambda m: f"${m['end']:,.0f} ({m['cagr']:+.1f}%/yr, SR {m['sr']:.2f}, DD {m['dd']:.0%})"  # noqa: E731
    span = f"{data['meta']['lo'][:7]} -> {data['dates'][-1][:7]}"
    record_trials([{"kind": "r4w_strategy", "name": v["ledger"],
                    "config": {**data["meta"]["config"], "train_years": v["train_years"],
                               "fills": "next open"},
                    "terminal_100": s["end"], "cagr_pct": s["cagr"], "sharpe": s["sr"],
                    "max_dd": s["dd"], "n_weeks": s["n"],
                    "notes": v["note"].format(span=span, s=line(s), y=line(y),
                                              out=v["out"].relative_to(ROOT))}])


def main(argv):
    for name in argv or list(VARIANTS):
        data = build(name)
        if VARIANTS[name].get("ledger"):
            record(name, data)


if __name__ == "__main__":
    main(sys.argv[1:])
