"""Build reports/oos_explorer.html: the nested OOS test (procedure selected on
2006-2015, graded 2016 -> 2024-06) as a self-contained explorer.

The replay goes through selection.simulate itself (with its trace hook), so
the app's curve is the engine's curve; this file only attributes each week's
return to the names that earned it and inlines the result into app.html.

    .venv/bin/python app/oos/build.py            # writes reports/oos_explorer.html
"""
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from stocks_ml.selection import COST, metrics, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WORLD = ROOT / "data/sharadar_world2000"
RANKINGS = ROOT / "data/experiments/nested2_v1/rankings_5y.parquet"
TICKERS = ROOT / "data/r5_live/sharadar_tickers.parquet"
TEMPLATE = Path(__file__).with_name("app.html")
OUT = ROOT / "reports/oos_explorer.html"

CONFIG = dict(horizon="4w", book=10, cap=2, stop=None, floor="60/40")
BALLAST = 0.4                      # the 60/40 floor: 40% of NAV in the trend ballast
LO, HI = pd.Timestamp("2016-01-01"), pd.Timestamp("2024-07-18")
AS_GRADED = ("The ledger's as-graded verdict (nested2_verdict, 2026-09-01) read $497 vs "
             "$295: it paid the 13 Thursday-dated picks in this window for the week that "
             "had already ended. nested2_verdict_amended carries the numbers shown here.")


def build_positions(trace, ballast=BALLAST, cost=COST):
    """NAV path and per-name positions from simulate's trace.

    Each credited week's return is split: every sleeve's names share the
    book's 1-ballast slice equally, the rotated sleeves' names share that
    rotation's cost, and the ballast earns its floor return. The split is
    asserted to re-assemble the engine's return each week. A position is a
    name's unbroken run in the book (any sleeve); `n` counts the sleeves
    holding it each week, `v` is the stock's weekly return, `p` the $ won
    or lost on a $100 start."""
    a = 1 - ballast
    ncoh = len(trace[0]["sleeves"])
    nav, positions, open_pos = [100.0], [], {}
    for j, rec in enumerate(trace):
        nav_prev = nav[-1]
        pnl, nsl, val = {}, {}, {}
        for c, (names, vals) in enumerate(zip(rec["sleeves"], rec["vals"])):
            w = a / (ncoh * len(names))
            share = a * (cost / ncoh) / len(names) if c in rec["rotated"] else 0.0
            for n, v in zip(names, vals):
                pnl[n] = pnl.get(n, 0.0) + nav_prev * (w * v - share)
                nsl[n] = nsl.get(n, 0) + 1
                val[n] = v
        ballast_pnl = nav_prev * ballast * rec["fr"]
        assert abs(sum(pnl.values()) + ballast_pnl - nav_prev * rec["r"]) < 1e-9 * nav_prev
        nav.append(nav_prev * (1 + rec["r"]))
        for n in [n for n in open_pos if n not in nsl]:
            p = open_pos.pop(n)
            p["s"] = rec["wk"]
            positions.append(p)
        for n in nsl:
            p = open_pos.get(n)
            if p is None:
                p = open_pos[n] = {"k": n, "j0": j, "b": rec["t"], "s": None,
                                   "n": [], "v": [], "p": []}
            p["n"].append(nsl[n])
            p["v"].append(val[n])
            p["p"].append(pnl[n])
    positions.extend(open_pos.values())
    positions.sort(key=lambda p: (p["j0"], p["k"]))
    return nav, positions


def replay():
    from stocks_ml.data.store import DataStore
    world = DataStore(WORLD)
    prices, mem = world.read("prices"), world.read("membership")
    smap = dict(mem.dropna(subset=["sector"]).drop_duplicates("ticker")[["ticker", "sector"]].values)
    cw = (prices.pivot(index="date", columns="ticker", values="close")
          .sort_index().ffill().resample("W-FRI").last())
    ctx = SimpleNamespace(cw=cw, wret=cw.pct_change(fill_method=None), spy_w=cw["SPY"], smap=smap)
    h = pd.read_parquet(RANKINGS).rename(columns={"tickers": "top15"})
    h["week"] = pd.to_datetime(h.week)
    trace = []
    rets = simulate(ctx, h, CONFIG["horizon"], CONFIG["book"], CONFIG["cap"],
                    CONFIG["stop"], CONFIG["floor"], trace=trace)
    trace = [x for x in trace if LO <= x["nxt"] < HI]
    spy = ctx.wret["SPY"].reindex([x["nxt"] for x in trace])
    return trace, rets, spy, smap


def assemble(trace, rets, spy, smap, names):
    nav, positions = build_positions(trace)
    tick = sorted({n for p in positions for n in [p["k"]]})
    idx = {t: i for i, t in enumerate(tick)}
    iso = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")  # noqa: E731
    weeks = [{"t": iso(x["t"]),
              "pick": [idx[n] for c in x["rotated"] for n in x["sleeves"][c]],
              "book": [[idx[n] for n in sl] for sl in x["sleeves"]],
              "g": x["g"]} for x in trace]
    spy_nav = list(100 * np.cumprod(1 + spy.fillna(0).values))
    grade = lambda s: {k: metrics(s, LO, HI)[k2] for k, k2 in  # noqa: E731
                       [("end", "terminal_100"), ("cagr", "cagr_pct"),
                        ("sr", "sharpe"), ("dd", "max_dd"), ("n", "n_weeks")]}
    return {
        "meta": {"built": str(date.today()), "config": CONFIG, "lo": iso(LO), "hi": iso(HI),
                 "first_pick": iso(trace[0]["t"]), "last_pick": iso(trace[-1]["t"]),
                 "as_graded": AS_GRADED},
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


def render(data, template=TEMPLATE, out=OUT):
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = template.read_text()
    assert html.count("__DATA__") == 1
    out.write_text(html.replace("__DATA__", payload))
    return out


def main():
    trace, rets, spy, smap = replay()
    t = pd.read_parquet(TICKERS).drop_duplicates("ticker").set_index("ticker")
    names = {k: {"name": r["name"], "industry": r["industry"]} for k, r in t.iterrows()}
    data = assemble(trace, rets, spy, smap, names)
    out = render(data)
    s, y = data["summary"]["strategy"], data["summary"]["spy"]
    print(f"{out}: {len(data['weeks'])} weeks, {len(data['tickers'])} tickers, "
          f"{len(data['positions'])} positions; ${s['end']:.0f} vs SPY ${y['end']:.0f}")


if __name__ == "__main__":
    main()
