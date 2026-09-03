"""stocks-ml: the champion's command line.

  r5-weekly       the weekly signal (GitHub Actions runs it every Saturday)
  select          the pre-registered selection procedure on a window
  procedure-card  regenerate PROCEDURE.md from models/champion_spec.json
"""
from __future__ import annotations

import argparse

from stocks_ml.config import load_config


def cmd_select(args, cfg):
    from stocks_ml.selection import run_select

    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else (0, 1)
    run_select(args.sel_start, args.sel_end, args.eval_start, args.eval_end,
               name=args.name, stage=args.stage, shard=shard)


def cmd_procedure_card(args, cfg):
    from stocks_ml.procedure_card import CARD_PATH, write_card

    write_card()
    print(f"wrote {CARD_PATH} from models/champion_spec.json")


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
        description="the r5 champion: r5-weekly | select | procedure-card")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

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

    args = parser.parse_args()
    cfg = load_config(args.config)
    {"r5-weekly": cmd_r5_weekly, "select": cmd_select,
     "procedure-card": cmd_procedure_card}[args.command](args, cfg)


if __name__ == "__main__":
    main()
