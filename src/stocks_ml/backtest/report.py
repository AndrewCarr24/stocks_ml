from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats

from stocks_ml.backtest.metrics import regime_flags, regime_summaries, summarize
from stocks_ml.backtest.simulator import run_backtest, walk_forward_predictions
from stocks_ml.backtest.strategies import make_strategies
from stocks_ml.models.candidates import get_candidates
from stocks_ml.models.champion import load_champion, holdout_start_date


def benchmark_navs(prices, fred, index: pd.DatetimeIndex) -> dict:
    spy = (prices[prices.ticker == "SPY"].set_index("date")["close"]
           .sort_index().reindex(index).ffill())
    first_valid = spy.first_valid_index()
    base = spy.loc[first_valid] if first_valid is not None else np.nan
    spy_nav = 100.0 * (spy / base)
    dtb3 = (fred[fred.series == "DTB3"].set_index("date")["value"]
            .sort_index().reindex(index).ffill().fillna(0.0))
    daily = (1 + dtb3 / 100.0) ** (1 / 252)
    cash_nav = 100.0 * daily.cumprod() / daily.iloc[0]
    return {"spy_hold": spy_nav, "cash": cash_nav}


def stress_windows():
    return [("GFC", "2008-09-01", "2009-03-31"),
            ("Q4 2018", "2018-10-01", "2018-12-31"),
            ("COVID crash", "2020-02-15", "2020-04-15"),
            ("2022 bear", "2022-01-01", "2022-12-31")]


def _window_return(nav: pd.Series, start, end):
    sub = nav[(nav.index >= pd.Timestamp(start)) & (nav.index <= pd.Timestamp(end))]
    return sub.iloc[-1] / sub.iloc[0] - 1 if len(sub) > 1 else np.nan


def _fmt(x, pct=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.1%}" if pct else f"{x:,.2f}"


def build_report(results: dict, bench: dict, flags: pd.DataFrame, n_trials: int,
                 champion_name: str, holdout_start,
                 cross_trial_var: float | None = None) -> str:
    lines = [f"# Backtest report", "",
             f"Champion model: **{champion_name}** · strategies × candidates tried: "
             f"**{n_trials}** (used to deflate Sharpe)", ""]

    all_navs = {**{k: r.nav for k, r in results.items()}, **bench}
    lines += ["## Headline ($100 invested at start)", "",
              "| strategy | $100 → | CAGR | Sharpe | Deflated Sharpe | Sortino | max DD "
              "| worst week | underwater (d) | costs $ | fits |", "|---|" + "---|" * 10]
    for name, r in results.items():
        s = summarize(r.nav, n_trials=n_trials, cross_trial_var=cross_trial_var)
        lines.append(
            f"| {name} | ${s['terminal_100']:,.0f} | {_fmt(s['cagr'], True)} | "
            f"{_fmt(s['sharpe'])} | {_fmt(s['deflated_sharpe'])} | {_fmt(s['sortino'])} | "
            f"{_fmt(s['max_drawdown'], True)} | {_fmt(s['worst_week'], True)} | "
            f"{s['longest_underwater_days']} | {_fmt(r.total_costs)} | {r.n_fits} |")
    for name, nav in bench.items():
        s = summarize(nav)
        lines.append(
            f"| {name} | ${s['terminal_100']:,.0f} | {_fmt(s['cagr'], True)} | "
            f"{_fmt(s['sharpe'])} | – | {_fmt(s['sortino'])} | "
            f"{_fmt(s['max_drawdown'], True)} | {_fmt(s['worst_week'], True)} | "
            f"{s['longest_underwater_days']} | – | – |")

    lines += ["", "## Regime-sliced performance (ann. Sharpe / hit rate)", "",
              "| strategy | bull | bear | high_vol | low_vol |", "|---|---|---|---|---|"]
    for name, nav in all_navs.items():
        rs = regime_summaries(nav, flags)
        cells = " | ".join(f"{_fmt(rs[k]['sharpe'])} / {_fmt(rs[k]['hit_rate'], True)}"
                           for k in ("bull", "bear", "high_vol", "low_vol"))
        lines.append(f"| {name} | {cells} |")

    lines += ["", "## Stress windows (total return)", "",
              "| window | " + " | ".join(all_navs) + " |",
              "|---|" + "---|" * len(all_navs)]
    for label, start, end in stress_windows():
        row = " | ".join(_fmt(_window_return(nav, start, end), True)
                         for nav in all_navs.values())
        lines.append(f"| {label} | {row} |")

    if holdout_start is not None:
        # THE decision table: the only window fully clean of model selection.
        # Strategy comparisons should lead with these numbers, not the
        # since-2005 headline (whose window overlaps CV/tuning).
        lines += ["", f"## Holdout period (≥ {pd.Timestamp(holdout_start).date()}, "
                      "never used for model selection — the primary strategy comparison)",
                  "", "| strategy | $100 → | total return | ann. Sharpe | max DD | worst week |",
                  "|---|---|---|---|---|---|"]
        for name, nav in all_navs.items():
            sub = nav[nav.index >= holdout_start]
            if len(sub) < 10:
                continue
            sub = 100.0 * (sub / sub.iloc[0])
            s = summarize(sub)
            lines.append(f"| {name} | ${s['terminal_100']:,.0f} | "
                         f"{_fmt(s['terminal_100'] / 100.0 - 1.0, True)} | {_fmt(s['sharpe'])} | "
                         f"{_fmt(s['max_drawdown'], True)} | {_fmt(s['worst_week'], True)} |")
        lines += ["", "![holdout equity curves](equity_holdout.png)"]

    # Recent five years, each NAV rescaled to $100 at the window start — "what
    # would $100 have done had I started then". Note the champion was selected
    # using CV folds that overlap most of this window, so unlike the holdout
    # section these figures are not fully out-of-sample.
    recent_start = max(nav.index.max() for nav in all_navs.values()) - pd.DateOffset(years=5)
    lines += ["", f"## Recent five years ($100 at {recent_start.date()}; "
                  "overlaps model-selection window — see holdout for the clean test)", "",
              "| strategy | $100 → | CAGR | Sharpe | max DD |", "|---|---|---|---|---|"]
    for name, nav in all_navs.items():
        sub = nav[nav.index >= recent_start]
        if len(sub) < 10:
            continue
        sub = 100.0 * (sub / sub.iloc[0])
        s = summarize(sub)
        lines.append(f"| {name} | ${s['terminal_100']:,.0f} | {_fmt(s['cagr'], True)} | "
                     f"{_fmt(s['sharpe'])} | {_fmt(s['max_drawdown'], True)} |")

    # MinTRL: how many live shadow-ledger weeks certify a strategy at this
    # trial count (Bailey-LdP; turns "let it accumulate" into a threshold).
    mintrl_note = "- MinTRL unavailable (needs a live strategy row and ledger variance)."
    live = results.get(getattr(build_report, "_live_strategy", "topk_spy"))
    if live is not None and holdout_start is not None and cross_trial_var is not None:
        from stocks_ml.backtest.metrics import expected_max_sr, min_track_record
        hold = live.nav[live.nav.index >= holdout_start]
        wk = hold.resample("W-FRI").last().pct_change().dropna()
        if len(wk) > 10 and wk.std() > 0:
            sr_w = float(wk.mean() / wk.std())
            sr0_w = expected_max_sr(n_trials, cross_trial_var / 52.0)
            mtrl = min_track_record(sr_w, sr0_w,
                                    skew=float(stats.skew(wk)),
                                    kurt=float(stats.kurtosis(wk, fisher=False)))
            mintrl_note = (
                f"- MinTRL: with N={n_trials} trials, the expected-max weekly SR of a "
                f"zero-skill strategy is {sr0_w:.3f}; a live strategy matching the "
                f"holdout live-strategy weekly SR ({sr_w:.3f}) needs "
                + (f"**≥ {mtrl:.0f} weeks** of shadow ledger to certify SR>0 at 95%."
                   if np.isfinite(mtrl) else
                   "**no finite record** — it does not clear the expected-max bar."))
    lines += ["", "## Honesty notes", "", mintrl_note,
              f"- Sharpe deflation uses N={n_trials} trials and "
              + ("the ledger's empirical cross-trial SR variance "
                 f"({cross_trial_var:.4f})." if cross_trial_var is not None else
                 "the single-path variance proxy (trials ledger too thin; "
                 "under-deflates diverse trials)."),
              "- Regime flags (SPY 200d SMA, VIX median) use full-sample statistics; "
              "they are reporting lenses, not tradable signals.",
              "- Fundamentals are sparse before ~2009 (EDGAR XBRL phase-in).",
              "- Delisted tickers missing from the free price source are absent from "
              "the panel; residual survivorship bias is reported in the ingestion manifest.",
              "- Positions whose ticker stops trading are liquidated at the last "
              "available (forward-filled) price — optimistic for bankruptcies.",
              "", "![equity curves](equity.png)"]
    return "\n".join(lines)


def run_all_backtests(store, cfg, models_dir="models", out_dir="reports") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel, prices, fred = store.read("panel"), store.read("prices"), store.read("fred")
    champ_name, estimator = load_champion(models_dir)

    strategies = make_strategies(cfg)
    # One prediction walk serves every strategy: predictions depend only on the
    # estimator, and the staggered ensemble makes them start-anchor independent.
    predictions = walk_forward_predictions(panel, estimator, cfg)
    results = {name: run_backtest(panel, prices, strat, estimator, cfg,
                                  predictions=predictions)
               for name, strat in strategies.items()}

    any_nav = next(iter(results.values())).nav
    bench = benchmark_navs(prices, fred, any_nav.index)
    spy = (prices[prices.ticker == "SPY"].set_index("date")["close"]
           .sort_index().reindex(any_nav.index).ffill())
    vix_raw = fred[fred.series == "VIXCLS"].set_index("date")["value"].sort_index()
    flags = regime_flags(spy, vix_raw)

    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    hold_start = holdout_start_date(dates, cfg.holdout_years)
    # Trial accounting: the ledger is the census (every tuning trial, tournament
    # candidate, league pipeline, strategy variant). The old strategies-times-
    # candidates heuristic remains a floor for installs with no ledger yet.
    from stocks_ml.models.trials import ledger_stats, record_trials

    rows = []
    for name, res in results.items():
        pre = res.nav[res.nav.index < hold_start] if hold_start is not None else res.nav
        sr = summarize(pre)["sharpe"]
        rows.append({"kind": "strategy_variant", "name": f"{champ_name}+{name}",
                     "pre_holdout_sharpe": round(sr, 4) if np.isfinite(sr) else None})
    ledger_path = Path(models_dir) / "trials_ledger.json"
    record_trials(rows, ledger_path)
    n_ledger, cross_var = ledger_stats(ledger_path)
    n_candidates = len(get_candidates(cfg, models_dir))
    n_trials = max(n_ledger, len(strategies) * n_candidates)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r in results.items():
        ax.plot(r.nav.index, r.nav.values, label=name)
    for name, nav in bench.items():
        ax.plot(nav.index, nav.values, label=name, linestyle="--", alpha=0.7)
    ax.set_yscale("log")
    ax.set_ylabel("NAV ($100 start, log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=120)
    plt.close(fig)

    # Persist the daily NAV series so charts/analysis can be regenerated without
    # re-running the multi-hour simulation.
    all_navs = {**{k: r.nav for k, r in results.items()}, **bench}
    pd.DataFrame(all_navs).to_csv(out / "navs.csv", index_label="date")

    # Holdout-only chart: the clean test. Every curve rescaled to $100 at the
    # holdout start; solid = strategies, dashed = benchmarks. Strategy evaluation
    # should lead with this view — the full-history chart overlaps the
    # model-selection window.
    if hold_start is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, nav in all_navs.items():
            sub = nav[nav.index >= hold_start]
            if len(sub) < 2:
                continue
            sub = 100.0 * (sub / sub.iloc[0])
            style = "--" if name in bench else "-"
            ax.plot(sub.index, sub.values, style, label=name,
                    alpha=0.7 if name in bench else 1.0)
        ax.set_ylabel(f"$100 at holdout start ({pd.Timestamp(hold_start).date()})")
        ax.set_title("Holdout period — never used for model selection")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "equity_holdout.png", dpi=120)
        plt.close(fig)

    report = build_report(results, bench, flags, n_trials, champ_name, hold_start,
                          cross_trial_var=cross_var)
    path = out / "backtest.md"
    path.write_text(report)
    return path
