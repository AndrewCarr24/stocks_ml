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
    magnitude is unmeasurable from this (free) price source.
    """
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    rows = []
    for _, ev in removals.iterrows():
        ticker = ev["ticker"]
        if ticker not in close.columns:
            continue
        rdate = pd.Timestamp(ev["date"])
        series = close[ticker].dropna()
        before = series[series.index <= rdate]
        after = series[series.index > rdate]
        if before.empty or after.empty:
            continue
        anchor = before.iloc[-1]
        outcome = after.iloc[horizon_days - 1] if len(after) >= horizon_days else after.iloc[-1]
        rows.append({
            "ticker": ticker,
            "date": rdate,
            "reason_class": classify_reason(ev.get("reason")),
            "post_ret": float(outcome / anchor - 1),
            "truncated": bool(len(after) < TRUNCATION_DAYS),
        })
    return pd.DataFrame(rows, columns=["ticker", "date", "reason_class", "post_ret", "truncated"])


def _class_stats(measured: pd.DataFrame) -> dict:
    """Per reason_class: n events, n truncated, median/q25 post_ret of the
    non-truncated (reliably measured) events. NaN median/q25 (no non-truncated
    events in the class) reads as 0.0 downstream, never as a punitive haircut.
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
    an assumption. Truncated events (price series disappears near removal --
    unmeasurable, and disappearing correlates with worse outcomes) are instead
    assigned their class's more punitive 25th-percentile loss,
    max(0, -quantile(post_ret, 0.25)).
    """
    stats = _class_stats(measured)
    median_haircut = {cls: (max(0.0, -s["median"]) if s["median"] == s["median"] else 0.0)
                      for cls, s in stats.items()}
    q25_haircut = {cls: (max(0.0, -s["q25"]) if s["q25"] == s["q25"] else 0.0)
                  for cls, s in stats.items()}

    for cls, s in stats.items():
        print(f"[survivorship] class={cls} n={s['n']} n_truncated={s['n_truncated']} "
              f"median_post_ret={s['median']:.2%} q25_post_ret={s['q25']:.2%} "
              f"haircut={median_haircut[cls]:.2%}")

    per_event = measured[["ticker", "date"]].copy()
    per_event["haircut"] = [
        q25_haircut[cls] if truncated else median_haircut[cls]
        for cls, truncated in zip(measured["reason_class"], measured["truncated"])
    ]
    return {"class_haircuts": median_haircut, "per_event": per_event}


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


def build_torture_report(measured: pd.DataFrame, class_haircuts: dict, results: dict,
                         baseline: dict, champion_name: str) -> str:
    lines = ["# Survivorship torture test", "",
             f"Champion model: **{champion_name}** (same fitted estimator as the "
             "committed baseline). Empirically measured removal-haircut stress "
             "test: how much of the backtest's headline performance survives when "
             "every simulated index-removal liquidation is penalized using "
             "empirically measured post-removal returns, without buying "
             "delisted-price data.", ""]

    lines += ["## Measured post-removal outcomes (the empirical evidence)", "",
              "| reason class | n events | n truncated | median post_ret | q25 post_ret | haircut |",
              "|---|---|---|---|---|---|"]
    stats = _class_stats(measured)
    for cls in sorted(stats):
        s = stats[cls]
        lines.append(f"| {cls} | {s['n']} | {s['n_truncated']} | {_fmt_pct(s['median'])} | "
                     f"{_fmt_pct(s['q25'])} | {_fmt_pct(class_haircuts.get(cls, 0.0))} |")

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
    report = build_torture_report(measured, haircuts["class_haircuts"], results, baseline, champ_name)
    path = out / "survivorship_torture.md"
    path.write_text(report)
    return path
