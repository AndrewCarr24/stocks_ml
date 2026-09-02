from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from stocks_ml.config import load_config
from stocks_ml.data.store import DataStore


def _store(cfg) -> DataStore:
    return DataStore(cfg.data_dir)


def cmd_ingest(args, cfg):
    from stocks_ml.data.edgar import ingest_edgar
    from stocks_ml.data.fred import ingest_fred
    from stocks_ml.data.insiders import ingest_form4
    from stocks_ml.data.membership import ingest_membership, members_asof
    from stocks_ml.data.prices import all_tickers_ever, ingest_prices
    from stocks_ml.data.sec8k import ingest_sec8k
    from stocks_ml.data.shortint import ingest_shortint
    from stocks_ml.features.panel import build_panel

    if args.full:
        for f in Path(cfg.data_dir).glob("*.parquet"):
            f.unlink()
    store = _store(cfg)
    mem = ingest_membership(store, cfg)
    print(f"membership: {store.manifest['membership']}")
    start = cfg.backtest_start - pd.Timedelta(days=450)
    print(f"prices: {ingest_prices(store, all_tickers_ever(mem), start)}")
    print(f"fred: {ingest_fred(store, cfg.fred_series, cfg.user_agent)}")
    current = members_asof(mem, pd.Timestamp.today())
    print(f"edgar: {ingest_edgar(store, current, cfg.edgar_concepts, cfg.user_agent)}")
    sec_tickers = [ticker for ticker in all_tickers_ever(mem) if ticker != "SPY"]
    print(f"sec8k: {ingest_sec8k(store, sec_tickers, cfg.user_agent)}")
    print(f"form4: {ingest_form4(store, cfg.user_agent)}")
    print(f"shortint: {ingest_shortint(store, cfg.user_agent)}")
    panel = build_panel(store, cfg)
    print(f"panel: {len(panel)} rows, {panel.date.min().date()} → {panel.date.max().date()}")


def cmd_train(args, cfg):
    from stocks_ml.models.champion import run_training

    run_training(_store(cfg), cfg)
    print("wrote models/champion.joblib, models/champion.json, models/selection.md")


_DEFAULT_SAMPLES = {"xgb": 40, "lgbm": 40, "catboost": 12, "enet": 12}
_DEFAULT_TRIALS = {"xgb": 100, "lgbm": 100, "catboost": 60, "enet": 40,
                   "ltr": 60, "xgb4w": 60}
_OPTUNA_ONLY = {"ltr", "xgb4w"}  # league families: no random-search grid


def cmd_tune(args, cfg):
    if args.optuna:
        from stocks_ml.models.optuna_tuning import tune_optuna

        n = args.trials if args.trials is not None else _DEFAULT_TRIALS[args.family]
        result = tune_optuna(_store(cfg), cfg, family=args.family, n_trials=n)
        status = "SELECTED" if result["selected"] else "no eligible trial"
        artifact = (f" and models/{args.family}_optuna.json"
                if result["selected"] else "")
        print(f"wrote models/optuna_{args.family}.md{artifact}; "
            f"best CV IC {result['best_cv_ic']:.4f} | {status}")
        return
    if args.family in _OPTUNA_ONLY:
        raise SystemExit(f"family {args.family!r} tunes via Optuna only; add --optuna")
    from stocks_ml.models.tuning import tune_model

    n = args.samples if args.samples is not None else _DEFAULT_SAMPLES[args.family]
    results = tune_model(_store(cfg), cfg, family=args.family, n_samples=n)
    print(f"wrote models/{args.family}_tuned.json and models/tuning_{args.family}.md")
    print(results.head(5)[["name", "mean_ic", "n_test_weeks"]].to_string(index=False))


def cmd_backtest(args, cfg):
    from stocks_ml.backtest.report import run_all_backtests

    path = run_all_backtests(_store(cfg), cfg)
    print(f"wrote {path}")


def cmd_pipelines(args, cfg):
    from stocks_ml.backtest.pipelines import run_league

    path = run_league(_store(cfg), cfg)
    print(f"wrote {path}")


def cmd_torture(args, cfg):
    from stocks_ml.backtest.survivorship import run_torture

    path = run_torture(_store(cfg), cfg)
    print(f"wrote {path}")


def cmd_leaderboard(args, cfg):
    from stocks_ml.backtest.leaderboard import build_leaderboard

    print(build_leaderboard(holdout=args.holdout))


def cmd_select(args, cfg):
    from stocks_ml.selection import run_select

    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    run_select(args.sel_start, args.sel_end, args.eval_start, args.eval_end,
               name=args.name, stage=args.stage, shard=shard)


def cmd_procedure_card(args, cfg):
    from stocks_ml.procedure_card import CARD_PATH, write_card

    write_card()
    print(f"wrote {CARD_PATH} from models/champion_spec.json")


def cmd_signals(args, cfg):
    from stocks_ml.live.ledger import CHALLENGER_TAG, Ledger
    from stocks_ml.live.signals import generate_signals

    store = _store(cfg)
    out = Path("signals")
    out.mkdir(exist_ok=True)

    ledger = Ledger.load("ledger.json")
    md, trades = generate_signals(store, cfg, ledger)
    path = out / f"{date.today()}.md"
    path.write_text(md)
    trades_path = out / f"{date.today()}-trades.json"
    trades_path.write_text(json.dumps([[t, d, p] for t, d, p in trades]))
    print(f"wrote {path} and {trades_path} ({len(trades)} trades suggested)")

    # Shadow race: the LTR challenger runs the same strategy on its own paper
    # ledger, giving a true out-of-sample head-to-head against the champion.
    from stocks_ml.backtest.pipelines import _optuna_params
    from stocks_ml.models.candidates import WeekGroupedXGBRanker

    from stocks_ml.backtest.strategies import EqualWeightTopK, SpyFloor

    ltr_params = _optuna_params("models", "ltr")
    challenger = WeekGroupedXGBRanker(**(ltr_params or {}))
    ledger2 = Ledger.load(f"ledger_{CHALLENGER_TAG}.json")
    if not (ledger2.positions or ledger2.cash):
        ledger2 = Ledger(cash=100.0)          # first run: same $100 start
        ledger2.save(f"ledger_{CHALLENGER_TAG}.json")
    md2, trades2 = generate_signals(
        store, cfg, ledger2, estimator=challenger,
        model_name=f"ltr{' (CV-tuned)' if ltr_params else ' (untuned)'}",
        strategy=SpyFloor(EqualWeightTopK(cfg.challenger_top_k)))
    path2 = out / f"{date.today()}-{CHALLENGER_TAG}.md"
    path2.write_text(md2)
    trades2_path = out / f"{date.today()}-{CHALLENGER_TAG}-trades.json"
    trades2_path.write_text(json.dumps([[t, d, p] for t, d, p in trades2]))
    print(f"wrote {path2} and {trades2_path} ({len(trades2)} challenger trades)")

    from stocks_ml.live.ledger import race_status
    status = race_status(ledger, ledger2)
    path.write_text(md + "\n\n" + status)
    print(status.splitlines()[-1])


def cmd_ledger(args, cfg):
    from stocks_ml.live.ledger import Ledger, find_latest_trades, latest_closes

    path = Path(args.ledger)
    ledger = Ledger.load(path)
    if args.action == "init":
        ledger = Ledger(cash=args.cash)
        ledger.save(path)
        print(f"initialized ledger with ${args.cash:,.2f}")
    elif args.action == "mark":
        store = _store(cfg)
        prices = store.read("prices")
        nav = ledger.mark(latest_closes(prices), prices["date"].max())
        ledger.save(path)
        print(f"NAV: ${nav:,.2f}")
    elif args.action == "apply":
        if args.file:
            tpath = Path(args.file)
        else:
            tpath = find_latest_trades(tag=args.tag)
            if tpath is None:
                raise SystemExit("no matching signals/*-trades.json found; "
                                 "run `stocks-ml signals` first")
        if tpath.name in ledger.applied_files and not args.force:
            print(f"already applied {tpath.name}; skipping (use --force to re-apply)")
            return
        trades = [(t, float(d), float(p)) for t, d, p in json.loads(tpath.read_text())]
        ledger.apply_trades(trades, date.today(), cost_bps=cfg.cost_bps if cfg else 0.0)
        ledger.applied_files.append(tpath.name)
        ledger.save(path)
        print(f"applied {len(trades)} trades from {tpath}")
        print(f"cash: ${ledger.cash:,.2f}  positions: {ledger.positions}")
    else:  # show
        print(f"cash: ${ledger.cash:,.2f}  positions: {ledger.positions}")
        for d, n in ledger.nav_history[-10:]:
            print(f"  {d}: ${n:,.2f}")


def cmd_r5_weekly(args, cfg):
    """The champion's weekly signal (live/r5.py): refresh the live world,
    rank, rotate a sleeve, fill last week's orders, write signals_r5/ and
    ledger_r5.json. Runs on GitHub Actions (.github/workflows/champion.yml);
    ops/r5_weekly.sh runs the same cycle on the owner's Mac."""
    from stocks_ml.live.r5 import run_weekly
    rep = run_weekly(args.live_dir, cfg, as_of=args.as_of, refresh=not args.no_refresh,
                     sec=not args.no_sec, dry_run=args.dry_run, capital=args.capital,
                     out_dir=args.out_dir, ledger_path=args.ledger)
    if args.commit and not args.dry_run:
        commit_outputs(rep["signal"]["date"], [args.out_dir, args.ledger])


def commit_outputs(t: str, paths: list[str], run=None) -> bool:
    """git add/commit/push the champion's outputs. A rerun on the same
    Friday regenerates identical files: nothing staged, nothing committed.
    Rebases onto origin before pushing so a commit that landed during the
    ~20-minute cycle does not reject the push."""
    import subprocess
    run = run or (lambda cmd, **kw: subprocess.run(cmd, **kw))
    run(["git", "add", *paths], check=True)
    if run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        print(f"r5: signal {t} unchanged; nothing to commit")
        return False
    run(["git", "commit", "-q", "-m", f"r5: signal {t}"], check=True)
    run(["git", "pull", "-q", "--rebase"], check=True)
    run(["git", "push", "-q"], check=True)
    print(f"committed and pushed r5: signal {t}")
    return True


def main():
    parser = argparse.ArgumentParser(
        prog="stocks-ml",
        description="ML stock forecasting: ingest | train | tune | backtest | signals | ledger | torture | r5-weekly")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="fetch data and build the panel")
    p_ingest.add_argument("--full", action="store_true")
    sub.add_parser("train", help="champion model selection")
    p_tune = sub.add_parser("tune", help="hyperparameter tuning via walk-forward CV")
    p_tune.add_argument("--family", choices=["xgb", "lgbm", "catboost", "enet",
                                             "ltr", "xgb4w"], default="xgb")
    p_tune.add_argument("--samples", type=int, default=None,
                        help="random-search configs (default: per-family — xgb/lgbm 40, catboost/enet 12)")
    p_tune.add_argument("--optuna", action="store_true",
                        help="use Optuna TPE search selected only on pre-holdout CV")
    p_tune.add_argument("--trials", type=int, default=None,
                        help="Optuna trials (default: per-family — xgb/lgbm 100, catboost 60, enet 40)")
    sub.add_parser("backtest", help="run all strategies and write the report")
    sub.add_parser("pipelines", help="multi-pipeline league: different targets/strategies/"
                                     "cadences, one $100 walk-forward exam")
    sub.add_parser("signals", help="generate this week's trade signals")
    p_sel = sub.add_parser("select", help="run the full selection procedure "
                           "on a window (stages: grid, wsweep, holdings, cascade)")
    p_sel.add_argument("--sel-start", required=True)
    p_sel.add_argument("--sel-end", required=True)
    p_sel.add_argument("--eval-start", default=None)
    p_sel.add_argument("--eval-end", default=None)
    p_sel.add_argument("--name", default=None)
    p_sel.add_argument("--stage", default="all",
                       choices=["all", "grid", "wsweep", "holdings", "cascade"])
    p_sel.add_argument("--shard", default=None, help="i/n to split stage weeks")
    sub.add_parser("procedure-card", help="regenerate PROCEDURE.md from "
                   "models/champion_spec.json")
    p_lb = sub.add_parser("leaderboard", help="the formal leaderboard (owner "
                          "format: earnings-ranked 3-2-1, 2001-2024, ensembles only)")
    p_lb.add_argument("--holdout", action="store_true")
    sub.add_parser("torture", help="survivorship torture test: empirical removal haircuts "
                                   "(requires `ingest` to have been re-run once for the "
                                   "removals dataset)")
    p_led = sub.add_parser("ledger", help="paper ledger operations")
    p_led.add_argument("action", choices=["init", "mark", "show", "apply"])
    p_led.add_argument("--cash", type=float, default=100.0)
    p_led.add_argument("--file", default=None,
                       help="trades JSON to apply (default: newest matching signals/*-trades.json)")
    p_led.add_argument("--force", action="store_true",
                       help="re-apply a trades file even if already recorded in applied_files")
    p_led.add_argument("--ledger", default="ledger.json",
                       help="ledger file (challenger shadow race uses ledger_ltr.json)")
    p_led.add_argument("--tag", default=None,
                       help="trades-file tag for default apply lookup (challenger: ltr)")

    p_r5 = sub.add_parser("r5-weekly", help="the r5 champion's weekly signal: refresh the "
                          "live world (Sharadar + SEC), rank, rotate a sleeve, keep the "
                          "paper ledger (signals_r5/, ledger_r5.json)")
    p_r5.add_argument("--live-dir", default="data/r5_live")
    p_r5.add_argument("--as-of", default=None, help="signal date (a panel Friday); "
                      "default: the latest panel date, which must be last Friday")
    p_r5.add_argument("--no-refresh", action="store_true",
                      help="rank on the stored panel without refreshing any data")
    p_r5.add_argument("--no-sec", action="store_true",
                      help="refresh Sharadar only (skip EDGAR/8-K/short interest/FRED)")
    p_r5.add_argument("--dry-run", action="store_true",
                      help="print the signal; write neither the ledger nor signals_r5/")
    p_r5.add_argument("--capital", type=float, default=100.0, help="paper capital at first run")
    p_r5.add_argument("--out-dir", default="signals_r5")
    p_r5.add_argument("--ledger", default="ledger_r5.json")
    p_r5.add_argument("--commit", action="store_true",
                      help="git add/commit/push the signal and ledger after a real run")

    args = parser.parse_args()
    cfg = load_config(args.config)
    {"ingest": cmd_ingest, "train": cmd_train, "tune": cmd_tune, "backtest": cmd_backtest,
     "leaderboard": cmd_leaderboard, "procedure-card": cmd_procedure_card,
     "select": cmd_select,
     "pipelines": cmd_pipelines, "signals": cmd_signals, "ledger": cmd_ledger,
     "torture": cmd_torture, "r5-weekly": cmd_r5_weekly}[args.command](args, cfg)


if __name__ == "__main__":
    main()
