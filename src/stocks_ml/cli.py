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
    from stocks_ml.data.membership import ingest_membership, members_asof
    from stocks_ml.data.prices import all_tickers_ever, ingest_prices
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
    panel = build_panel(store, cfg)
    print(f"panel: {len(panel)} rows, {panel.date.min().date()} → {panel.date.max().date()}")


def cmd_train(args, cfg):
    from stocks_ml.models.champion import run_training

    run_training(_store(cfg), cfg)
    print("wrote models/champion.joblib, models/champion.json, models/selection.md")


def cmd_backtest(args, cfg):
    from stocks_ml.backtest.report import run_all_backtests

    path = run_all_backtests(_store(cfg), cfg)
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
        trades = [(t, float(d), float(p)) for t, d, p in json.loads(tpath.read_text())]
        ledger.apply_trades(trades, date.today())
        ledger.save(path)
        print(f"applied {len(trades)} trades from {tpath}")
        print(f"cash: ${ledger.cash:,.2f}  positions: {ledger.positions}")
    else:  # show
        print(f"cash: ${ledger.cash:,.2f}  positions: {ledger.positions}")
        for d, n in ledger.nav_history[-10:]:
            print(f"  {d}: ${n:,.2f}")


def main():
    parser = argparse.ArgumentParser(prog="stocks-ml",
                                     description="ML stock forecasting: ingest | train | backtest | signals | ledger")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="fetch data and build the panel")
    p_ingest.add_argument("--full", action="store_true")
    sub.add_parser("train", help="champion model selection")
    sub.add_parser("backtest", help="run all strategies and write the report")
    sub.add_parser("signals", help="generate this week's trade signals")
    p_led = sub.add_parser("ledger", help="paper ledger operations")
    p_led.add_argument("action", choices=["init", "mark", "show", "apply"])
    p_led.add_argument("--cash", type=float, default=100.0)
    p_led.add_argument("--file", default=None,
                       help="trades JSON to apply (default: newest signals/*-trades.json)")

    args = parser.parse_args()
    cfg = load_config(args.config)
    {"ingest": cmd_ingest, "train": cmd_train, "backtest": cmd_backtest,
     "signals": cmd_signals, "ledger": cmd_ledger}[args.command](args, cfg)


if __name__ == "__main__":
    main()
