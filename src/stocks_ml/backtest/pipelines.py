"""Multi-pipeline league: different targets/strategies/cadences, one exam.

Owner's design: pipelines may differ in training objective (regression /
learning-to-rank / classification), investment strategy, and rebalance cadence,
but every one is judged by the identical criterion — $100 through the same
cost-aware walk-forward simulator. Cadence differences are automatically priced
because turnover costs are modeled.

Guardrails (agreed before wave 1):
- Wave-1 challengers use conventional untuned settings; tuning happens only if
  a challenger looks promising, and only on the pre-holdout CV window.
- The holdout section leads the report (iron rule 0) but is graded only for
  these pre-declared pipelines; the league table itself grows the effective
  trial count that Sharpe deflation must answer for.
- The shadow ledger, not this league, makes the final live-strategy call.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from stocks_ml.backtest.metrics import summarize
from stocks_ml.backtest.report import benchmark_navs
from stocks_ml.backtest.simulator import run_backtest, walk_forward_predictions
from stocks_ml.backtest.strategies import ConfidenceTopK, EqualWeightTopK, SpyFloor
from stocks_ml.models.candidates import (TopQuintileClassifier, WeekGroupedXGBRanker,
                                         make_tuned)
from stocks_ml.models.champion import holdout_start_date

# The 4-week label spans ~29 calendar days of future prices; its pipelines must
# purge at least that plus the weekly label's usual 10-day safety margin.
PURGE_DAYS_4W = 42


@dataclass
class Pipeline:
    name: str
    estimator: object
    strategy: object
    description: str
    label_col: str = "label"
    purge_days: int | None = None
    rebalance_every: int = 1


def wave1_pipelines(cfg, models_dir: str = "models") -> list[Pipeline]:
    pipes = []
    incumbent = make_tuned("xgb", models_dir)
    if incumbent is not None:
        pipes.append(Pipeline(
            "incumbent_weekly", incumbent, SpyFloor(EqualWeightTopK(cfg.top_k)),
            "champion regression (RMSE-trained), weekly, topk_spy — the baseline"))
    pipes.append(Pipeline(
        "ltr_weekly", WeekGroupedXGBRanker(), SpyFloor(EqualWeightTopK(cfg.top_k)),
        "learning-to-rank (rank:ndcg, week = query group), weekly, topk_spy"))
    monthly = make_tuned("xgb", models_dir)
    if monthly is not None:
        # inner early-stop split must also respect the longer label span
        monthly.set_params(early_stop_purge_days=PURGE_DAYS_4W)
        pipes.append(Pipeline(
            "monthly_reg", monthly, SpyFloor(EqualWeightTopK(cfg.top_k)),
            "champion hyperparameters on the 4-week label, monthly cadence, topk_spy",
            label_col="label_4w", purge_days=PURGE_DAYS_4W, rebalance_every=4))
    pipes.append(Pipeline(
        "quintile_prob_weekly", TopQuintileClassifier(),
        SpyFloor(ConfidenceTopK(cfg.top_k, floor=0.5)),
        "P(top-quintile) classifier, weekly; only >50% confidence fills a slot, "
        "rest in SPY"))
    return pipes


def _section(lines, title, navs, bench_names, lo=None, hi=None, note=""):
    lines += ["", f"## {title}", ""]
    if note:
        lines += [note, ""]
    lines += ["| pipeline | $100 → | total return | ann. Sharpe | max DD | worst week |",
              "|---|---|---|---|---|---|"]
    for name, nav in navs.items():
        sub = nav
        if lo is not None:
            sub = sub[sub.index >= lo]
        if hi is not None:
            sub = sub[sub.index < hi]
        if len(sub) < 10:
            continue
        sub = 100.0 * sub / sub.iloc[0]
        s = summarize(sub)
        tag = " (benchmark)" if name in bench_names else ""
        lines.append(f"| {name}{tag} | ${s['terminal_100']:,.0f} | "
                     f"{s['terminal_100'] / 100 - 1.0:+.1%} | {s['sharpe']:.2f} | "
                     f"{s['max_drawdown']:.1%} | {s['worst_week']:.1%} |")


def run_league(store, cfg, models_dir: str = "models", out_dir: str = "reports") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel, prices, fred = store.read("panel"), store.read("prices"), store.read("fred")
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    hold_start = holdout_start_date(dates, cfg.holdout_years)
    start = pd.Timestamp(cfg.eval_start)

    pipes = wave1_pipelines(cfg, models_dir)
    navs, meta = {}, {}
    for p in pipes:
        wf = walk_forward_predictions(
            panel, p.estimator, cfg, start=start, label_col=p.label_col,
            purge_days=p.purge_days, rebalance_every=p.rebalance_every)
        res = run_backtest(
            panel, prices, p.strategy, p.estimator, cfg, start=start,
            predictions=wf, label_col=p.label_col, purge_days=p.purge_days,
            rebalance_every=p.rebalance_every)
        navs[p.name] = res.nav
        meta[p.name] = res
        print(f"done {p.name}: ${res.nav.iloc[-1]:,.0f} "
              f"({res.n_fits} fits, costs ${res.total_costs:.2f})", flush=True)

    bench = benchmark_navs(prices, fred, next(iter(navs.values())).index)
    navs.update(bench)

    lines = ["# Pipeline league — one exam, many recipes", "",
             f"All pipelines: $100 walked forward from {start.date()} through the same "
             f"cost-aware simulator (5bps one-way), staggered-refit ensembles, tie guard. "
             f"They differ in training objective, strategy, and cadence:", ""]
    for p in pipes:
        lines.append(f"- **{p.name}**: {p.description}")
    lines += ["", f"Trials to date across the project (feeds Sharpe-deflation "
                  f"honesty): each league row adds one."]

    _section(lines, f"Holdout (≥ {pd.Timestamp(hold_start).date()}) — the primary comparison",
             navs, set(bench), lo=hold_start,
             note="Never used for tuning or selection of any pipeline.")
    _section(lines, f"Selection window ({start.date()} → {pd.Timestamp(hold_start).date()})",
             navs, set(bench), lo=None, hi=hold_start,
             note="Overlaps the incumbent's CV/tuning window — context, not verdict.")

    lines += ["", "## Cost and fit accounting", "",
              "| pipeline | rebalances/cadence | model fits | costs $ |", "|---|---|---|---|"]
    for p in pipes:
        r = meta[p.name]
        cadence = f"every {p.rebalance_every} wk" if p.rebalance_every > 1 else "weekly"
        lines.append(f"| {p.name} | {cadence} | {r.n_fits} | {r.total_costs:,.2f} |")
    lines += ["", "Honesty notes:",
              "- Wave-1 challengers are deliberately untuned (conventional defaults); "
              "the incumbent had an Optuna budget. A challenger that competes while "
              "untuned is the interesting signal.",
              "- Every pipeline graded here spends holdout novelty; the shadow ledger "
              "makes the final call.",
              "", "![pipeline holdout equity](equity_pipelines.png)"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, nav in navs.items():
        sub = nav[nav.index >= hold_start]
        if len(sub) < 2:
            continue
        sub = 100.0 * sub / sub.iloc[0]
        style = "--" if name in bench else "-"
        ax.plot(sub.index, sub.values, style, label=name, alpha=0.7 if name in bench else 1.0)
    ax.set_ylabel(f"$100 at holdout start ({pd.Timestamp(hold_start).date()})")
    ax.set_title("Pipeline league — holdout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "equity_pipelines.png", dpi=120)
    plt.close(fig)

    pd.DataFrame(navs).to_csv(out / "pipelines_navs.csv", index_label="date")
    path = out / "pipelines.md"
    path.write_text("\n".join(lines))
    return path
