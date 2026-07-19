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
    print(f"form4: {ingest_form4(store, cfg.user_agent)}")
    print(f"shortint: {ingest_shortint(store, cfg.user_agent)}")
    panel = build_panel(store, cfg)
    print(f"panel: {len(panel)} rows, {panel.date.min().date()} → {panel.date.max().date()}")


def cmd_train(args, cfg):
    from stocks_ml.models.champion import run_training

    run_training(_store(cfg), cfg)
    print("wrote models/champion.joblib, models/champion.json, models/selection.md")


_DEFAULT_SAMPLES = {"xgb": 40, "lgbm": 40, "catboost": 12, "enet": 12}
_DEFAULT_TRIALS = {"xgb": 100, "lgbm": 100, "catboost": 60, "enet": 40}


def cmd_tune(args, cfg):
    if args.optuna:
        from stocks_ml.models.optuna_tuning import tune_optuna

        n = args.trials if args.trials is not None else _DEFAULT_TRIALS[args.family]
        result = tune_optuna(_store(cfg), cfg, family=args.family, n_trials=n)
        print(f"wrote models/optuna_{args.family}.md"
              + (f" and models/{args.family}_optuna.json" if result["adopted"] else ""))
        print(f"best CV IC {result['best_cv_ic']:.4f} | "
              f"candidate holdout IC {result['candidate_holdout_ic']:.4f} | "
              f"incumbent holdout IC {result['incumbent_holdout_ic']:.4f} | "
              f"{'ADOPTED' if result['adopted'] else 'rejected'}")
        return
    from stocks_ml.models.tuning import tune_model

    n = args.samples if args.samples is not None else _DEFAULT_SAMPLES[args.family]
    results = tune_model(_store(cfg), cfg, family=args.family, n_samples=n)
    print(f"wrote models/{args.family}_tuned.json and models/tuning_{args.family}.md")
    print(results.head(5)[["name", "mean_ic", "n_test_weeks"]].to_string(index=False))


def cmd_backtest(args, cfg):
    from stocks_ml.backtest.report import run_all_backtests

    path = run_all_backtests(_store(cfg), cfg)
    print(f"wrote {path}")


def cmd_torture(args, cfg):
    from stocks_ml.backtest.survivorship import run_torture

    path = run_torture(_store(cfg), cfg)
    print(f"wrote {path}")


def cmd_signals(args, cfg):
    from stocks_ml.live.ledger import Ledger
    from stocks_ml.live.signals import generate_signals

    ledger = Ledger.load("ledger.json")
    md, trades = generate_signals(_store(cfg), cfg, ledger)
    out = Path("signals")
    out.mkdir(exist_ok=True)
    path = out / f"{date.today()}.md"
    path.write_text(md)
    trades_path = out / f"{date.today()}-trades.json"
    trades_path.write_text(json.dumps([[t, d, p] for t, d, p in trades]))
    print(f"wrote {path} and {trades_path} ({len(trades)} trades suggested)")


def cmd_ledger(args, cfg):
    from stocks_ml.live.ledger import Ledger, latest_closes

    path = Path("ledger.json")
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
            candidates = sorted(Path("signals").glob("*-trades.json"))
            if not candidates:
                raise SystemExit("no signals/*-trades.json found; run `stocks-ml signals` first")
            tpath = candidates[-1]
        if tpath.name in ledger.applied_files and not args.force:
            print(f"already applied {tpath.name}; skipping (use --force to re-apply)")
            return
        trades = [(t, float(d), float(p)) for t, d, p in json.loads(tpath.read_text())]
        ledger.apply_trades(trades, date.today())
        ledger.applied_files.append(tpath.name)
        ledger.save(path)
        print(f"applied {len(trades)} trades from {tpath}")
        print(f"cash: ${ledger.cash:,.2f}  positions: {ledger.positions}")
    else:  # show
        print(f"cash: ${ledger.cash:,.2f}  positions: {ledger.positions}")
        for d, n in ledger.nav_history[-10:]:
            print(f"  {d}: ${n:,.2f}")


def main():
    parser = argparse.ArgumentParser(
        prog="stocks-ml",
        description="ML stock forecasting: ingest | train | tune | backtest | signals | ledger | torture")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="fetch data and build the panel")
    p_ingest.add_argument("--full", action="store_true")
    sub.add_parser("train", help="champion model selection")
    p_tune = sub.add_parser("tune", help="hyperparameter tuning via walk-forward CV")
    p_tune.add_argument("--family", choices=["xgb", "lgbm", "catboost", "enet"], default="xgb")
    p_tune.add_argument("--samples", type=int, default=None,
                        help="random-search configs (default: per-family — xgb/lgbm 40, catboost/enet 12)")
    p_tune.add_argument("--optuna", action="store_true",
                        help="use Optuna TPE search, adopting the winner only if it beats "
                             "the random-search config on the untouched holdout")
    p_tune.add_argument("--trials", type=int, default=None,
                        help="Optuna trials (default: per-family — xgb/lgbm 100, catboost 60, enet 40)")
    sub.add_parser("backtest", help="run all strategies and write the report")
    sub.add_parser("signals", help="generate this week's trade signals")
    sub.add_parser("torture", help="survivorship torture test: empirical removal haircuts "
                                   "(requires `ingest` to have been re-run once for the "
                                   "removals dataset)")
    p_led = sub.add_parser("ledger", help="paper ledger operations")
    p_led.add_argument("action", choices=["init", "mark", "show", "apply"])
    p_led.add_argument("--cash", type=float, default=100.0)
    p_led.add_argument("--file", default=None,
                       help="trades JSON to apply (default: newest signals/*-trades.json)")
    p_led.add_argument("--force", action="store_true",
                       help="re-apply a trades file even if already recorded in applied_files")

    args = parser.parse_args()
    cfg = load_config(args.config)
    {"ingest": cmd_ingest, "train": cmd_train, "tune": cmd_tune, "backtest": cmd_backtest,
     "signals": cmd_signals, "ledger": cmd_ledger, "torture": cmd_torture}[args.command](args, cfg)


if __name__ == "__main__":
    main()
