"""`stocks-ml app`: the champion's backtest as one interactive page.

The page (reports/champion_explorer.html, git-ignored: it carries licensed
prices) replays the deployed spec's walk week by week through
selection.simulate — the engine the backtest, the eval and the live ledger
share — with its trace hook on, and inlines the result into app.html: the
growth of $100 against the S&P 500, every week's picks and sleeves, and each
company's history in the book.

What it reads, all written by earlier steps:

    models/champion_spec.json        the strategy layers and the walk
                                     (`stocks-ml procedure`)
    <walk>/{select,extend}/preds     the K copies' scores (`stocks-ml train`)
    <walk>/eval.json                 the grade, confidence, leak audit
                                     (`stocks-ml eval`), quoted in the notes
    data/r5_live/sharadar_tickers    company names (optional)

and caches the K-copy ranking as <walk>/holdings_k{K}.parquet. Ranks before
the holdout drive the book; credited weeks end at the holdout, exclusive.

    stocks-ml app                        # -> reports/champion_explorer.html
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.backtest import SELECT_START, load_preds, spec_settings
from stocks_ml.eval import segments, spec_walk
from stocks_ml.selection import HOLDOUT_START, HORIZONS, K_COPIES, metrics, simulate

SPEC_PATH = Path("models/champion_spec.json")
STORE = "data/sharadar_world2000_nominal_dl"
TICKERS = Path("data/r5_live/sharadar_tickers.parquet")
TEMPLATE = Path(__file__).with_name("app.html")
OUT = Path("reports/champion_explorer.html")
TITLE = "Champion Explorer"
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight", 10: "ten",
         16: "sixteen"}
FLOOR_WORDS = {"none": "no trend ballast", "halfgate": "half-gate trend ballast"}
LABEL_WORDS = {"label_4w": "the stock's 4-week return minus that week's median member's",
               "label_4w_sector": "the stock's 4-week return minus its sector's median that week"}


def describe(config: dict, train_years: int) -> str:
    """'4-week horizon, 8-year training window, top-10 in four staggered
    sleeves, no sector cap, half-gate trend ballast, no stop-loss'"""
    k = HORIZONS[config["horizon"]]["kweeks"]
    cap = f"sector cap {config['cap']}" if config["cap"] else "no sector cap"
    stop = f"{config['stop']:+.0%} stop-loss" if config["stop"] else "no stop-loss"
    floor = FLOOR_WORDS.get(config["floor"], f"{config['floor']} trend ballast")
    return (f"{k}-week horizon, {train_years}-year training window, top-{config['book']} in "
            f"{WORDS[k]} staggered sleeves, {cap}, {floor}, {stop}")


def ballast_words(floor: str) -> str:
    """How the floor parks the money (ledger.floor_split)."""
    if floor == "none":
        return "no ballast: the whole of NAV is in the book"
    if floor == "halfgate":
        return ("the book's share of NAV is 100% less half the share of breached SPY trailing "
                "means (30/40/52 weeks) — 100% with none breached, 50% with all three — and "
                "the rest sits in IEF")
    ballast = 100 - int(floor.split("/")[0])
    return (f"{ballast}% of NAV sits in the trend ballast, SPY shifted to IEF one third per "
            f"breached trailing mean (30/40/52 weeks)")


def accounting(config: dict) -> str:
    """The weekly-accounting paragraph of the notes."""
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


def verdict(spec: dict, ev: dict) -> str:
    """The notes' verdict paragraph, read off the spec and the eval record
    (never typed): what the model is, how its numbers were reached, the
    confidence band, the leak audit."""
    k = spec["ensemble"]["k_copies"]
    label = spec["horizon"]["label"]
    tab, ci = ev["table"][ev["label"]], ev.get("confidence") or {}
    spy = ev["table"]["sp500"]
    dollars = lambda w: f"${tab[w]['terminal_100']:,.0f} vs ${spy[w]['terminal_100']:,.0f}"  # noqa: E731
    s = (f"The champion as deployed (models/champion_spec.json, PROCEDURE.md): the model is trained on "
         f"the panel's f_ columns on the nominal price basis, target {label} ({LABEL_WORDS.get(label, label)}), "
         f"a {spec['training_window_years']}-year trailing window refit every week, {WORDS.get(k, k)} copies "
         f"averaged, labels that grade a delisting to its final print. The book, ballast, stop and cap were "
         f"decided by stocks-ml procedure's code on the walk's 2006–2015 segment; the label and window were "
         f"chosen on 2006–2015 alone and 2016–2024 was read once (stocks-ml eval, {ev['evaluated_at'][:10]}): "
         f"2006–2015 {dollars('2006-2015')}, 2016–2024 {dollars('2016-2024')}, 2006–2024 {dollars('2006-2024')} "
         f"against the S&P 500; paired weekly t vs the S&P 500 {tab['paired_t_vs_sp500']['2006-2024']:+.2f} "
         f"(2006–2024) / {tab['paired_t_vs_sp500']['2016-2024']:+.2f} (2016–2024).")
    if ev.get("falsification"):
        f = ev["falsification"]["2016-2024"]
        s += (f" Falsification against the incumbent ({Path(ev['incumbent']['walk']).name}): paired weekly "
              f"t {f['t']:+.2f} on {f['weeks']} weeks of 2016–2024, {ev['falsification']['verdict']}.")
    if ci:
        c16, c06 = ci["2016-2024"]["excess_cagr_vs_sp500_pct"], ci["2006-2024"]["excess_cagr_vs_sp500_pct"]
        s += (f" 95% nested interval on the excess return over the S&P 500: {c16['nested_95'][0]:+.1f}..{c16['nested_95'][1]:+.1f}%/yr "
              f"around {c16['point']:+.1f} (2016–2024), {c06['nested_95'][0]:+.1f}..{c06['nested_95'][1]:+.1f} around "
              f"{c06['point']:+.1f} (2006–2024); {ci['2016-2024']['p_excess_positive_nested']:.0%} of resampled "
              f"2016–2024 histories beat the S&P 500.")
    s += (f" Leak audit {ev['leak_audit']['VERDICT']}. In-sample: the book, ballast and cap were chosen on "
          f"2006–2015, which this page includes; the holdout ({HOLDOUT_START.date()} →) is untouched.")
    return s


def holdings(sel, ctx, walk: Path, k: int, log=print) -> pd.DataFrame:
    """The walk's K-copy ranking at every rank week before the holdout, as
    the engine ranks it (selection.ensemble_holdings), cached beside the walk."""
    out = walk / f"holdings_k{k}.parquet"
    if not out.exists():
        preds = load_preds(segments(walk))
        log(f"app: ranking {preds.week.nunique()} weeks at K={k} -> {out}")
        rows, _ = sel.ensemble_holdings(ctx, preds, range(1, k + 1))
        rows.sort_values("week").reset_index(drop=True).to_parquet(out, index=False)
    h = pd.read_parquet(out)
    h["week"] = pd.to_datetime(h["week"])
    return h.sort_values("week").reset_index(drop=True)


def replay(ctx, h: pd.DataFrame, config: dict, lo=SELECT_START, hi=HOLDOUT_START):
    """The engine's trace: ranks before `hi` drive the book; the credited
    weeks in [lo, hi) are kept."""
    trace = []
    rets = simulate(ctx, h[h.week < hi], config["horizon"], config["book"], config["cap"],
                    config["stop"], config["floor"], trace=trace)
    trace = [x for x in trace if lo <= x["nxt"] < hi]
    spy = ctx.wret["SPY"].reindex([x["nxt"] for x in trace])
    return trace, rets, spy


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


def assemble(meta: dict, config: dict, trace, rets, spy, smap, names, lo=SELECT_START, hi=HOLDOUT_START):
    """The page's data: the meta paragraphs, the two NAV paths, every week's
    picks and sleeves, the tickers, the positions."""
    nav, positions = build_positions(trace)
    tick = sorted({p["k"] for p in positions})
    idx = {t: i for i, t in enumerate(tick)}
    iso = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")  # noqa: E731
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
    return {
        "meta": {**meta, "built": str(date.today()), "config": config, "lo": iso(lo), "hi": iso(hi),
                 "first_pick": iso(trace[0]["t"]), "last_pick": iso(trace[-1]["t"]),
                 "book_word": WORDS[config["book"]], "accounting": accounting(config)},
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


def render(data: dict, out: Path, template: Path = TEMPLATE) -> Path:
    """Inline the data into the template; `</` is escaped so the JSON cannot
    close its own script tag."""
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = template.read_text()
    assert html.count("__DATA__") == 1
    title = data.get("meta", {}).get("title")
    if title:
        html = html.replace("<title>__TITLE__</title>", f"<title>{title}</title>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("__DATA__", payload))
    return out


def company_names(path: Path = TICKERS) -> dict:
    if not Path(path).exists():
        return {}
    t = pd.read_parquet(path).drop_duplicates("ticker").set_index("ticker")
    return {k: {"name": r["name"], "industry": r["industry"]} for k, r in t.iterrows()}


def build(store: str = STORE, out: Path = OUT, spec_path: Path = SPEC_PATH, k: int = K_COPIES,
          tickers: Path = TICKERS, log=print) -> dict:
    """Build the champion's page from the spec's walk; returns the page data."""
    from stocks_ml.models.trials import record_trials
    from stocks_ml.train import context
    spec = json.loads(Path(spec_path).read_text())
    walk = spec_walk(spec_path)
    ev_path = walk / "eval.json"
    if not ev_path.exists():
        raise SystemExit(f"{ev_path} missing: run stocks-ml eval first (the notes quote it)")
    ev = json.loads(ev_path.read_text())
    st = spec_settings(spec_path)
    config = dict(horizon="4w", **st)
    train_years = spec["training_window_years"]
    sel, ctx, _ = context(store)
    h = holdings(sel, ctx, walk, k, log=log)
    trace, rets, spy = replay(ctx, h, config)
    span = f"{SELECT_START.strftime('%Y-%m')} → {pd.Timestamp(trace[-1]['nxt']).strftime('%Y-%m')}"
    meta = {"title": TITLE, "crumb": "Champion chart", "scale": "log", "store": store,
            "intro": (f"The champion as deployed — {describe(config, train_years)}, {WORDS.get(k, k)} copies "
                      f"averaged, on the nominal price basis with delisting-honest labels — graded on {span} "
                      f"against the S&P 500 (SPY), costs included, $100 start. In-sample: its book, ballast "
                      f"and cap were chosen after reading these years; the holdout ({HOLDOUT_START.date()} →) "
                      f"is untouched."),
            "verdict": verdict(spec, ev)}
    data = assemble(meta, config, trace, rets, spy, ctx.smap, company_names(tickers))
    out = render(data, Path(out))
    s, y = data["summary"]["strategy"], data["summary"]["spy"]
    line = lambda m: f"${m['end']:,.0f} ({m['cagr']:+.1f}%/yr, SR {m['sr']:.2f}, DD {m['dd']:.0%})"  # noqa: E731
    log(f"{out}: {describe(config, train_years)}; {len(data['weeks'])} weeks, "
        f"{len(data['tickers'])} tickers, {len(data['positions'])} positions, "
        f"{sum(bool(w['stop']) for w in data['weeks'])} weeks with a stopped name; "
        f"{line(s)} vs SPY {line(y)}")
    record_trials([{"kind": "app", "name": f"app_champion_{walk.name}_k{k}",
                    "config": {**config, "train_years": train_years, "fills": "next open"},
                    "terminal_100": s["end"], "cagr_pct": s["cagr"], "sharpe": s["sr"],
                    "max_dd": s["dd"], "n_weeks": s["n"],
                    "notes": f"{span}, the spec's walk {walk} replayed by selection.simulate at K={k} "
                             f"for {out}: {line(s)} vs SPY {line(y)} (stocks-ml app)"}])
    return data
