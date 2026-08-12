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

import json
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
from stocks_ml.features.panel import LABEL_4W_PURGE_DAYS as PURGE_DAYS_4W
from stocks_ml.models.candidates import (TimeTailEarlyStopXGB, TopQuintileClassifier,
                                         WeekGroupedXGBRanker, make_tuned)
from stocks_ml.models.champion import holdout_start_date


def _optuna_params(models_dir: str, family: str) -> dict | None:
    """CV-selected Optuna params for a league family, if that search has run."""
    path = Path(models_dir) / f"{family}_optuna.json"
    return json.loads(path.read_text()) if path.exists() else None


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
    ltr_params = _optuna_params(models_dir, "ltr")
    pipes.append(Pipeline(
        "ltr_weekly", WeekGroupedXGBRanker(**(ltr_params or {})),
        SpyFloor(EqualWeightTopK(cfg.top_k)),
        "learning-to-rank (rank:ndcg, week = query group), weekly, topk_spy"
        + (" — CV-tuned" if ltr_params else " — untuned defaults")))
    m4_params = _optuna_params(models_dir, "xgb4w")
    monthly = (TimeTailEarlyStopXGB(**m4_params) if m4_params
               else make_tuned("xgb", models_dir))
    if monthly is not None:
        # inner early-stop split must also respect the longer label span
        monthly.set_params(early_stop_purge_days=PURGE_DAYS_4W)
        pipes.append(Pipeline(
            "monthly_reg", monthly, SpyFloor(EqualWeightTopK(cfg.top_k)),
            ("CV-tuned 4-week-label regression" if m4_params
             else "champion hyperparameters on the 4-week label")
            + ", monthly cadence, topk_spy",
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


def render_combined_charts(out_dir: str = "reports") -> list[Path]:
    """Every model and strategy on shared equity charts (full history + holdout).

    Merges the strategy zoo's navs.csv with the league's pipelines_navs.csv —
    valid only because both are generated from the same panel vintage (the
    2026-08-12 reconciliation: incumbent_weekly == topk_spy to the week, so the
    duplicate row is dropped). Regenerate both CSVs before trusting the output
    if either run is stale."""
    out = Path(out_dir)
    frames = []
    zoo_path, lg_path = out / "navs.csv", out / "pipelines_navs.csv"
    if zoo_path.exists():
        frames.append(pd.read_csv(zoo_path, parse_dates=["date"], index_col="date"))
    if lg_path.exists():
        lg = pd.read_csv(lg_path, parse_dates=["date"], index_col="date")
        if frames:  # zoo present: drop the league's duplicates of it
            lg = lg.drop(columns=["incumbent_weekly", "spy_hold", "cash"],
                         errors="ignore")
        frames.append(lg)
    if not frames:
        return []
    navs = pd.concat(frames, axis=1)
    navs = navs.loc[:, ~navs.columns.duplicated()]
    bench = {"spy_hold", "cash"}

    written = []
    specs = [("equity_all.png", None, "All models and strategies — $100 at start (log scale)", True),
             ("equity_all_holdout.png", pd.Timestamp("2024-07-19"),
              "All models and strategies — holdout, $100 at 2024-07-19", False)]
    for fname, lo, title, log in specs:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for name in navs.columns:
            nav = navs[name].dropna()
            if lo is not None:
                nav = nav[nav.index >= lo]
            if len(nav) < 2:
                continue
            nav = 100.0 * nav / nav.iloc[0]
            style = "--" if name in bench else "-"
            ax.plot(nav.index, nav.values, style, label=name,
                    alpha=0.6 if name in bench else 1.0,
                    linewidth=1.1 if name in bench else 1.5)
        if log:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("$100 →")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path = out / fname
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def run_league(store, cfg, models_dir: str = "models", out_dir: str = "reports") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel, prices, fred = store.read("panel"), store.read("prices"), store.read("fred")
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    hold_start = holdout_start_date(dates, cfg.holdout_years)
    # Same exam as the strategy zoo (owner-mandated): full history from
    # backtest_start, not just the CV-era window.
    start = pd.Timestamp(cfg.backtest_start)

    pipes = wave1_pipelines(cfg, models_dir)
    navs, meta, ledger_rows = {}, {}, []
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
        pre = res.nav[res.nav.index < hold_start] if hold_start is not None else res.nav
        pre_sr = summarize(pre)["sharpe"]
        ledger_rows.append({"kind": "league_pipeline", "name": p.name,
                            "pre_holdout_sharpe": (round(pre_sr, 4)
                                                   if pd.notna(pre_sr) else None)})
        print(f"done {p.name}: ${res.nav.iloc[-1]:,.0f} "
              f"({res.n_fits} fits, costs ${res.total_costs:.2f})", flush=True)

    from stocks_ml.models.trials import record_trials
    record_trials(ledger_rows, Path(models_dir) / "trials_ledger.json")

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
    _section(lines, f"Since {start.date()} (full history)",
             navs, set(bench),
             note="Same exam as the strategy zoo. Pre-holdout years overlap the "
                  "tuned pipelines' CV/tuning windows — context, not verdict; "
                  "the holdout section above is the clean test.")

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
    lines += ["", "Combined view of every model and strategy: "
                  "![all, full history](equity_all.png) "
                  "![all, holdout](equity_all_holdout.png)"]
    path = out / "pipelines.md"
    path.write_text("\n".join(lines))
    render_combined_charts(out_dir)
    return path
