"""Survivorship torture test: quantify how much of the backtest's headline
performance survives when index-removal outcomes are penalized using
*empirically measured* post-removal returns, without buying delisted-price data.

Pipeline: classify each real S&P 500 removal's stated reason (`classify_reason`)
-> measure what actually happened to the stock's price after removal
(`measure_post_removal`) -> turn those empirical outcomes into a haircut per
removal event (`compute_haircuts`) -> feed those haircuts into the simulator's
`removal_haircuts` hook and re-run the strategies (`run_torture`), reporting the
tortured results against the committed (non-tortured) baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.backtest.metrics import summarize
from stocks_ml.backtest.simulator import run_backtest
from stocks_ml.backtest.strategies import make_strategies
from stocks_ml.models.champion import load_champion

TRUNCATION_DAYS = 60          # fewer trading days of post-removal data than this -> unmeasurable
GFC_WINDOW = ("2008-09-01", "2009-03-31")

_ACQUISITION_KEYWORDS = ("acquir", "merg", "taken private", "purchas", "bought",
                        "private equity", "buyout")
_DECLINE_KEYWORDS = ("market cap", "bankrupt", "chapter", "receiver", "delist",
                     "liquidat", "financial distress", "moved to")
_RESTRUCTURING_KEYWORDS = ("spun off", "spin", "split")


def classify_reason(reason) -> str:
    """Classify a Wikipedia S&P 500 change-table 'Reason' string into
    'acquisition' | 'decline' | 'restructuring' | 'unknown'.

    Order matters: acquisition keywords are checked first, so e.g. "acquired
    after bankruptcy" classifies as acquisition, not decline -- acquisition is
    the less punitive, empirically-grounded outcome and wins ties.
    """
    if pd.isna(reason):
        return "unknown"
    text = str(reason).lower()
    if any(k in text for k in _ACQUISITION_KEYWORDS):
        return "acquisition"
    if any(k in text for k in _DECLINE_KEYWORDS):
        return "decline"
    if any(k in text for k in _RESTRUCTURING_KEYWORDS):
        return "restructuring"
    return "unknown"


def measure_post_removal(prices: pd.DataFrame, removals: pd.DataFrame,
                         horizon_days: int = 126) -> pd.DataFrame:
    """For each removal event with price data, measure what actually happened
    to the price afterward.

    anchor = last close on/before the removal date; outcome = close
    `horizon_days` trading days after removal, or the final available close if
    the series ends sooner. `truncated` flags events whose price series ends
    within `TRUNCATION_DAYS` trading days of removal -- the ticker's data
    effectively disappears, which is itself evidence of a bad outcome, but the
    magnitude is unmeasurable from this (free) price source. Events with NO
    post-removal price data at all (the clearest delisting/bankruptcy cases)
    are still recorded here -- truncated=True, post_ret=NaN -- so they appear
    in the evidence counts and receive a (fallback-derived, see
    compute_haircuts) haircut instead of silently vanishing and leaking
    optimism into exactly the cases this test exists to penalize.
    """
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    rows = []
    for _, ev in removals.iterrows():
        ticker = ev["ticker"]
        if ticker not in close.columns:
            # No price data for this ticker anywhere in the store: it could
            # never have been bought or held by the simulator in the first
            # place, so there is nothing to measure or penalize -- correctly
            # (and only) skipped case.
            continue
        rdate = pd.Timestamp(ev["date"])
        series = close[ticker].dropna()
        before = series[series.index <= rdate]
        if before.empty:
            # No anchor price exists (the series starts after the removal
            # date) -- cannot compute even an unmeasurable event; skip.
            continue
        anchor = before.iloc[-1]
        reason_class = classify_reason(ev.get("reason"))
        after = series[series.index > rdate]
        if after.empty:
            # The clearest delisting case: price data stops at/before the
            # removal date. Record it -- unmeasurable magnitude (NaN
            # post_ret) but a very real, maximally truncated event.
            rows.append({"ticker": ticker, "date": rdate, "reason_class": reason_class,
                        "post_ret": float("nan"), "truncated": True})
            continue
        outcome = after.iloc[horizon_days - 1] if len(after) >= horizon_days else after.iloc[-1]
        rows.append({
            "ticker": ticker,
            "date": rdate,
            "reason_class": reason_class,
            "post_ret": float(outcome / anchor - 1),
            "truncated": bool(len(after) < TRUNCATION_DAYS),
        })
    return pd.DataFrame(rows, columns=["ticker", "date", "reason_class", "post_ret", "truncated"])


def _class_stats(measured: pd.DataFrame) -> dict:
    """Per reason_class: n events, n truncated, median/q25 post_ret of the
    non-truncated (reliably measured) events. NaN median/q25 means the class
    has zero non-truncated events (nothing of its own is measurable) --
    compute_haircuts resolves that through a fallback chain, never silently
    as 0.0.
    """
    stats = {}
    for cls, grp in measured.groupby("reason_class"):
        non_trunc = grp.loc[~grp["truncated"], "post_ret"]
        stats[cls] = {
            "n": int(len(grp)),
            "n_truncated": int(grp["truncated"].sum()),
            "median": float(non_trunc.median()) if len(non_trunc) else float("nan"),
            "q25": float(non_trunc.quantile(0.25)) if len(non_trunc) else float("nan"),
        }
    return stats


def compute_haircuts(measured: pd.DataFrame) -> dict:
    """Turn measured post-removal outcomes into a haircut per removal event.

    Per class: haircut = max(0, -median(post_ret of non-truncated events)) --
    classes whose empirical median post-removal return is non-negative
    (expected: acquisition, restructuring) get 0.0, the empirical outcome, not
    an assumption. Truncated events (price series disappears near/at removal
    -- unmeasurable, and disappearing correlates with worse outcomes) are
    instead assigned a more punitive 25th-percentile-loss haircut, resolved
    through a fallback chain when the event's own class has no measurable
    non-truncated events to compute a q25 from (e.g. a class that is 100%
    truncated): own class q25 -> 'decline' class q25 -> global q25 (across all
    non-truncated events of any class) -> 0.0, logged as an explicit warning
    (this is the one case that UNDERSTATES risk -- nothing anywhere is
    measurable).

    Returns {"class_haircuts": {...}, "per_event": DataFrame[ticker, date,
    haircut], "class_stats": {cls: {n, n_truncated, median, q25, fallback,
    haircut}}}. class_stats records which fallback tier each class actually
    used, surfaced in the torture report's evidence table.
    """
    stats = _class_stats(measured)

    def _haircut(q):
        return max(0.0, -q) if q == q else float("nan")

    global_non_trunc = measured.loc[~measured["truncated"], "post_ret"]
    global_q25 = float(global_non_trunc.quantile(0.25)) if len(global_non_trunc) else float("nan")
    decline_q25 = stats.get("decline", {}).get("q25", float("nan"))

    truncated_haircut, fallback = {}, {}
    for cls, s in stats.items():
        if s["q25"] == s["q25"]:
            truncated_haircut[cls], fallback[cls] = _haircut(s["q25"]), "own"
        elif decline_q25 == decline_q25:
            truncated_haircut[cls], fallback[cls] = _haircut(decline_q25), "decline"
        elif global_q25 == global_q25:
            truncated_haircut[cls], fallback[cls] = _haircut(global_q25), "global"
        else:
            truncated_haircut[cls], fallback[cls] = 0.0, "none"
            print(f"[survivorship] WARNING: class={cls} has no measurable q25 at any "
                  "fallback level (own/decline/global all unmeasurable) -- its "
                  "truncated events get a 0.0 haircut, UNDERSTATING risk")

    median_haircut = {}
    for cls, s in stats.items():
        if s["median"] == s["median"]:
            median_haircut[cls] = _haircut(s["median"])
        else:
            # A class with zero non-truncated events has no median of its
            # own; reporting 0.0 here would misleadingly read as "no penalty"
            # even though every event in the class needed the fallback chain
            # above -- surface the resolved fallback haircut instead.
            median_haircut[cls] = truncated_haircut[cls]

    class_stats = {cls: {**s, "fallback": fallback[cls], "haircut": median_haircut[cls]}
                   for cls, s in stats.items()}

    for cls, s in class_stats.items():
        fb_note = "" if s["fallback"] == "own" else f" fallback={s['fallback']}"
        print(f"[survivorship] class={cls} n={s['n']} n_truncated={s['n_truncated']} "
              f"median_post_ret={s['median']:.2%} q25_post_ret={s['q25']:.2%} "
              f"haircut={s['haircut']:.2%}{fb_note}")

    per_event = measured[["ticker", "date"]].copy()
    per_event["haircut"] = [
        truncated_haircut[cls] if truncated else median_haircut[cls]
        for cls, truncated in zip(measured["reason_class"], measured["truncated"])
    ]
    return {"class_haircuts": median_haircut, "per_event": per_event, "class_stats": class_stats}


def _parse_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _find_table(lines: list[str], header_marker: str) -> tuple[list[str], list[list[str]]]:
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and header_marker in line:
            header = _parse_row(line)
            rows = []
            for row_line in lines[i + 2:]:          # i+1 is the '|---|---|' separator
                if not row_line.strip().startswith("|"):
                    break
                rows.append(_parse_row(row_line))
            return header, rows
    raise ValueError(f"no markdown table with header containing {header_marker!r} found")


def parse_baseline_report(path) -> dict:
    """Extract the committed baseline's headline and GFC-stress numbers,
    verbatim as display strings, from reports/backtest.md.

    The torture run reuses these rather than re-running the (hours-long)
    non-tortured backtest: same code/panel/seed makes the committed report a
    valid control.
    """
    lines = Path(path).read_text().splitlines()
    head_header, head_rows = _find_table(lines, "$100")
    headline = {row[0]: dict(zip(head_header, row)) for row in head_rows}
    stress_header, stress_rows = _find_table(lines, "window")
    stress = {row[0]: dict(zip(stress_header, row)) for row in stress_rows}
    return {"headline": headline, "gfc": stress.get("GFC", {})}


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.1%}"


def _window_return(nav: pd.Series, start, end) -> float:
    sub = nav[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]
    return float(sub.iloc[-1] / sub.iloc[0] - 1) if len(sub) > 1 else float("nan")


def build_torture_report(class_stats: dict, results: dict, baseline: dict,
                         champion_name: str) -> str:
    lines = ["# Survivorship torture test", "",
             f"Champion model: **{champion_name}** (same fitted estimator as the "
             "committed baseline). Empirically measured removal-haircut stress "
             "test: how much of the backtest's headline performance survives when "
             "every simulated index-removal liquidation is penalized using "
             "empirically measured post-removal returns, without buying "
             "delisted-price data.", ""]

    lines += ["## Measured post-removal outcomes (the empirical evidence)", "",
              "| reason class | n events | n truncated | median post_ret | q25 post_ret | "
              "haircut | fallback |",
              "|---|---|---|---|---|---|---|"]
    for cls in sorted(class_stats):
        s = class_stats[cls]
        fb_note = "–" if s["fallback"] == "own" else f"fallback: {s['fallback']}"
        lines.append(f"| {cls} | {s['n']} | {s['n_truncated']} | {_fmt_pct(s['median'])} | "
                     f"{_fmt_pct(s['q25'])} | {_fmt_pct(s['haircut'])} | {fb_note} |")

    lines += ["", "## Headline: committed baseline (reports/backtest.md) vs. tortured", "",
              "| strategy | baseline $100→ | tortured $100→ | baseline CAGR | "
              "tortured CAGR | baseline max DD | tortured max DD |",
              "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        s = summarize(r.nav)
        b = baseline.get("headline", {}).get(name, {})
        lines.append(f"| {name} | {b.get('$100 →', '–')} | ${s['terminal_100']:,.0f} | "
                     f"{b.get('CAGR', '–')} | {_fmt_pct(s['cagr'])} | "
                     f"{b.get('max DD', '–')} | {_fmt_pct(s['max_drawdown'])} |")

    lines += ["", "## GFC stress window (total return): baseline vs. tortured", "",
              "| strategy | baseline | tortured |", "|---|---|---|"]
    for name, r in results.items():
        tortured = _window_return(r.nav, *GFC_WINDOW)
        lines.append(f"| {name} | {baseline.get('gfc', {}).get(name, '–')} | "
                     f"{_fmt_pct(tortured)} |")

    lines += ["", "## Interpretation", "",
              "- Haircuts fire only when the simulated strategy is still holding a "
              "ticker at the moment it is actually removed from the S&P 500 (within "
              "a 35-day window of the real removal date); they cannot simulate "
              "holding through the tickers that are fully missing from the free "
              "price source and so never entered the panel at all -- those "
              "casualties are not represented here at all.",
              "- The haircut is applied as a proceeds reduction at the liquidation "
              "exec day, including cases where the base (non-tortured) simulator "
              "would otherwise mark the position at a stale forward-filled price.",
              "- Truncated removal events (price history ends within "
              f"{TRUNCATION_DAYS} trading days of removal, i.e. the ticker's data "
              "disappears near the event -- typically the worst outcomes) are "
              "assigned their class's empirical 25th-percentile loss rather than "
              "their own unmeasurable return.",
              "- Classes whose empirical median post-removal return is non-negative "
              "(acquisitions, restructurings) get a 0.0 haircut -- the measured "
              "outcome, not an assumption."]
    return "\n".join(lines)


def run_torture(store, cfg, models_dir="models", out_dir="reports",
                baseline_report="reports/backtest.md") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel, prices, removals = store.read("panel"), store.read("prices"), store.read("removals")
    champ_name, estimator = load_champion(models_dir)

    measured = measure_post_removal(prices, removals)
    haircuts = compute_haircuts(measured)

    strategies = make_strategies(cfg)
    results = {name: run_backtest(panel, prices, strat, estimator, cfg,
                                  removal_haircuts=haircuts["per_event"])
              for name, strat in strategies.items()}

    baseline = parse_baseline_report(baseline_report)
    report = build_torture_report(haircuts["class_stats"], results, baseline, champ_name)
    path = out / "survivorship_torture.md"
    path.write_text(report)
    return path
