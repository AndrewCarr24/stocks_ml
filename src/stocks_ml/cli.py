"""stocks-ml: the champion's command line — one command per process.

  world           refresh a world store (Sharadar + SEC) and rebuild its panel   data/world.py
  train           the walk: the model refit every week, K copies, saved        train.py
  backtest        a walk through the strategy: the table vs the S&P 500        backtest.py
  procedure       the strategy layers decided on a walk, written to the spec   procedure.py
  procedure-card  PROCEDURE.md from the spec                                    procedure_card.py
  eval            the one look: grade, confidence, falsification, leak audit,
                  charts                                                        eval.py
  app             the interactive explorer (reports/champion_explorer.html)    app/build.py
  r5-weekly       the live weekly signal (GitHub Actions, every Saturday)      live/r5.py
"""
from __future__ import annotations

import argparse

from stocks_ml.config import load_config

STORE = "data/sharadar_world2000_nominal_dl"


def cmd_world(args, cfg):
    """Refresh the store's inputs (unless --no-refresh) and rebuild
    panel.parquet + panel_sf.parquet under config.yaml's price basis and
    delisting rule. The live job runs the same two steps every Saturday."""
    from stocks_ml.data.world import build_world_panel, refresh_world
    if not args.no_refresh:
        refresh_world(args.dir, cfg, sec=not args.no_sec)
    build_world_panel(args.dir, cfg)


def cmd_train(args, cfg):
    from stocks_ml.train import check_reproduces, walk
    if args.check:
        res = check_reproduces(args.store, args.check, weeks=args.check_weeks, copies=(1,))
        if not all(res.values()):
            raise SystemExit(f"the walk does not reproduce: {res}")
        return
    walk(args.store, args.start, args.end, args.label, args.train_years, args.k, args.out,
         every=args.every)


def cmd_backtest(args, cfg):
    from stocks_ml.backtest import run, spec_settings
    st = spec_settings()
    for key in ("book", "floor", "stop", "cap"):
        v = getattr(args, key)
        if v is not None:
            st[key] = v
    run(args.preds, args.store, st, args.k, name=args.name)


def cmd_procedure(args, cfg):
    """The strategy layers (book, floor, stop, cap) decided on a saved K-copy
    walk of the selection window, and the model fields (label, training
    window) read from the walk's own record, written into
    models/champion_spec.json — the only way those fields change."""
    from stocks_ml.procedure import run
    run(args.preds, store=args.store, k=args.k, lo=args.sel_start, hi=args.sel_end,
        check=args.check)


def cmd_procedure_card(args, cfg):
    from stocks_ml.procedure_card import CARD_PATH, write_card
    write_card()
    print(f"wrote {CARD_PATH} from models/champion_spec.json")


def cmd_eval(args, cfg):
    from stocks_ml.eval import run
    run(args.walk, args.incumbent, store=args.store, k=args.k, ci_draws=args.ci_draws,
        charts=not args.no_charts)


def cmd_app(args, cfg):
    from stocks_ml.app.build import build
    build(store=args.store, out=args.out)


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


def _floor(v: str):
    from stocks_ml.ledger import FLOORS
    if v not in FLOORS:
        raise argparse.ArgumentTypeError(f"floor must be one of {tuple(FLOORS)}")
    return v


def _optional_int(v: str):
    return None if v.lower() in ("none", "null") else int(v)


def _optional_float(v: str):
    return None if v.lower() in ("none", "null") else float(v)


def main():
    parser = argparse.ArgumentParser(
        prog="stocks-ml",
        description="world | train | backtest | procedure | procedure-card | eval | app | r5-weekly")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("world", help="refresh a world store (Sharadar + SEC) and rebuild its panel")
    p.add_argument("--dir", default="data/r5_live", help="the store (default: the live world)")
    p.add_argument("--no-refresh", action="store_true", help="rebuild the panel only")
    p.add_argument("--no-sec", action="store_true", help="refresh Sharadar only")

    p = sub.add_parser("train", help="the walk: refit the model every week of a window, K copies")
    p.add_argument("--out", help="directory for preds.parquet + spec.json (a walk segment)")
    p.add_argument("--start", default="2006-01-01")
    p.add_argument("--end", default="2015-12-31")
    p.add_argument("--label", default="label_4w_sector")
    p.add_argument("--train-years", type=int, default=8)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--every", type=int, default=1, help="every Nth week: a sample, exploration only")
    p.add_argument("--store", default=STORE)
    p.add_argument("--check", metavar="PREDS", help="refit copy 1 of a saved walk's first weeks and "
                   "compare; writes nothing")
    p.add_argument("--check-weeks", type=int, default=2)

    p = sub.add_parser("backtest", help="a walk through the strategy: the table vs the S&P 500")
    p.add_argument("--preds", nargs="+", required=True, help="walk segments (preds.parquet files)")
    p.add_argument("--store", default=STORE)
    p.add_argument("--k", type=int, default=None, help="copies to average (default: all)")
    p.add_argument("--book", type=int, default=None)
    p.add_argument("--floor", type=_floor, default=None)
    p.add_argument("--stop", type=_optional_float, default=None, help="e.g. -0.25 or none")
    p.add_argument("--cap", type=_optional_int, default=None, help="names per sector, or none")
    p.add_argument("--name", default="walk")

    p = sub.add_parser("procedure", help="decide the strategy layers on a walk of the selection "
                       "window and write them, with the walk's model recipe, into the spec (+ PROCEDURE.md)")
    p.add_argument("--preds", required=True, help="<walk>/select/preds.parquet")
    p.add_argument("--store", default=STORE)
    p.add_argument("--k", type=int, default=None, help="copies (default selection.K_COPIES)")
    p.add_argument("--sel-start", default="2006-01-01")
    p.add_argument("--sel-end", default="2015-12-31")
    p.add_argument("--check", action="store_true",
                   help="recompute and compare; write nothing; exit 1 if the spec drifted")

    sub.add_parser("procedure-card", help="regenerate PROCEDURE.md from models/champion_spec.json")

    p = sub.add_parser("eval", help="the one look at a walk: grade, confidence, falsification, "
                       "leak audit, charts")
    p.add_argument("--walk", default=None, help="walk directory (default: the spec's walk)")
    p.add_argument("--incumbent", default=None, help="the incumbent's walk directory")
    p.add_argument("--store", default=STORE)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--ci-draws", type=int, default=200, help="seed draws for the intervals; 0 skips")
    p.add_argument("--no-charts", action="store_true")

    p = sub.add_parser("app", help="build the interactive explorer of the champion's backtest")
    p.add_argument("--store", default=STORE)
    p.add_argument("--out", default="reports/champion_explorer.html")

    p = sub.add_parser("r5-weekly", help="the champion's weekly signal: refresh the live world, "
                       "rank, rotate a sleeve, keep the paper ledger (signals_r5/, ledger_r5.json)")
    p.add_argument("--live-dir", default="data/r5_live")
    p.add_argument("--as-of", default=None, help="signal date (a panel Friday); "
                   "default: the latest panel date, which must be last Friday")
    p.add_argument("--no-refresh", action="store_true",
                   help="rank on the stored panel without refreshing any data")
    p.add_argument("--no-sec", action="store_true",
                   help="refresh Sharadar only (skip EDGAR/8-K/short interest/FRED)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the signal; write neither the ledger nor signals_r5/")
    p.add_argument("--capital", type=float, default=100.0, help="paper capital at first run")
    p.add_argument("--out-dir", default="signals_r5")
    p.add_argument("--ledger", default="ledger_r5.json")
    p.add_argument("--commit", action="store_true",
                   help="git add/commit/push the signal and ledger after a real run")

    args = parser.parse_args()
    if args.command == "train" and not args.check and not args.out:
        parser.error("train needs --out (or --check PREDS)")
    cfg = load_config(args.config)
    {"world": cmd_world, "train": cmd_train, "backtest": cmd_backtest,
     "procedure": cmd_procedure, "procedure-card": cmd_procedure_card,
     "eval": cmd_eval, "app": cmd_app, "r5-weekly": cmd_r5_weekly}[args.command](args, cfg)


if __name__ == "__main__":
    main()
