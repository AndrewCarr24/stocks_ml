"""stocks-ml: the champion's command line — one command per process.

  world           refresh a world store (Sharadar + SEC) and rebuild its panel   data/world.py
  train           the walk: the model refit every week, K copies, saved        train.py
  backtest        a walk through the strategy: the table vs the S&P 500        backtest.py
  procedure       the strategy layers decided on a walk, written to the spec   procedure.py
  procedure-card  PROCEDURE.md from the spec                                    procedure_card.py
  eval            the one look: grade, confidence, falsification, leak audit,
                  charts                                                        eval.py
  app             the interactive explorer (reports/champion_explorer.html)    app/build.py
  challenge       the challenger protocol: candidate recipes vs the incumbent  challenge.py
  challenge-fast  the prototype: candidates on a stratified random sample, K=16 challenge.py
  explain         Shapley feature importance of the champion's yearly fits     explain.py
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
    if args.top:
        from stocks_ml.data.world import build_research_store, derive_research_store
        if args.dir == "data/r5_live":
            raise SystemExit("--top builds a NEW research world: give it its own --dir")
        if args.derive_from:
            derive_research_store(args.dir, args.derive_from, cfg, n=args.top, membership_from=args.membership_from,
                                  top_up=args.top_up, sized=args.sized)
            return
        from stocks_ml.data.sharadar import api_key
        build_research_store(args.dir, api_key(), cfg, n=args.top, sec=not args.no_sec)
        return
    if args.append_sf:
        from stocks_ml.data.world import append_sf_columns
        append_sf_columns(args.dir, cfg)
        return
    if args.append_sr:
        from stocks_ml.data.world import append_sector_relative
        append_sector_relative(args.dir)
        return
    if args.append_dv:
        from stocks_ml.data.world import append_clean_dollar_volume
        append_clean_dollar_volume(args.dir)
        return
    if args.extras:
        from stocks_ml.data.sharadar import api_key
        from stocks_ml.data.store import DataStore
        from stocks_ml.data.world import refresh_extras
        refresh_extras(DataStore(args.dir), api_key())
        return
    if not args.no_refresh:
        refresh_world(args.dir, cfg, sec=not args.no_sec)
    build_world_panel(args.dir, cfg)


def _params(v: str | None) -> dict:
    """--params learning_rate=0.01,n_estimators=300 (typed by selection.MODEL_PARAMS)."""
    if not v:
        return {}
    return dict(kv.split("=", 1) for kv in v.split(","))


def cmd_train(args, cfg):
    from stocks_ml.train import check_reproduces, walk
    if args.check:
        res = check_reproduces(args.store, args.check, weeks=args.check_weeks, copies=(1,))
        if not all(res.values()):
            raise SystemExit(f"the walk does not reproduce: {res}")
        return
    copies = None
    if args.copies:
        a, b = (int(x) for x in args.copies.split("-"))
        copies = list(range(a, b + 1))
    walk(args.store, args.start, args.end, args.label, args.train_years,
         len(copies) if copies else args.k, args.out,
         every=args.every, features=[f for f in (args.features or "").split(",") if f],
         params=_params(args.params), workers=args.workers, copies=copies,
         drop=[f for f in (args.drop or "").split(",") if f], train_top=args.train_top,
         refit_every=args.refit_every)


def cmd_challenge(args, cfg):
    from stocks_ml.challenge import incumbent_recipe, parse_candidate, run
    base = incumbent_recipe(args.incumbent, allow_other=args.incumbent_recipe_ok)
    cands = [parse_candidate(c, base) for c in args.candidate]
    run(cands, args.incumbent, args.out, store=args.store, k16=args.k16, lo=args.sel_start,
        hi=args.sel_end, workers=args.workers, adjudicate_window=args.adjudicate, refit_every=args.refit_every,
        incumbent_recipe_ok=args.incumbent_recipe_ok)


def cmd_challenge_fast(args, cfg):
    from stocks_ml.challenge import incumbent_recipe, parse_candidate, run_fast
    base = incumbent_recipe(args.incumbent, allow_other=args.incumbent_recipe_ok)
    cands = [parse_candidate(c, base) for c in args.candidate]
    run_fast(cands, args.incumbent, args.out, store=args.store, per_year=args.per_year,
             seed=args.seed, lo=args.sel_start, hi=args.sel_end, workers=args.workers,
             refit_every=args.refit_every, k=args.k, incumbent_recipe_ok=args.incumbent_recipe_ok)


def cmd_backtest(args, cfg):
    from stocks_ml.backtest import run, spec_settings
    if args.rolling is not None:
        from stocks_ml.rolling import parse_lookback
        from stocks_ml.rolling import run as run_rolling
        run_rolling(args.preds, args.store, parse_lookback(args.rolling), cadence=args.cadence,
                    min_years=args.min_years, sel_lo=args.sel_start, sel_hi=args.sel_end,
                    out=args.rolling_out, k=args.k, workers=args.workers)
        return
    given = {key: getattr(args, key) for key in ("book", "floor", "stop", "cap", "vol_cut")
             if getattr(args, key) is not None}
    if given.get("vol_cut") in ("none", "null"):
        given["vol_cut"] = None
    st = {**spec_settings(), **given} if given else None    # None: the walk's own settings
    run(args.preds, args.store, st, args.k, name=args.name)


def cmd_procedure(args, cfg):
    """The strategy layers (book, floor, stop, cap) decided on a saved K-copy
    walk of the selection window, and the model fields (label, training
    window) read from the walk's own record, written into
    models/champion_spec.json — the only way those fields change."""
    if args.lookback:
        from stocks_ml.rolling import choose
        choose(args.lookback, args.sel_start, args.sel_end)
        return
    if not args.preds:
        raise SystemExit("procedure needs --preds (or --lookback <rolling json files>)")
    from stocks_ml.procedure import run
    run(args.preds, store=args.store, k=args.k, lo=args.sel_start, hi=args.sel_end,
        check=args.check)


def cmd_procedure_card(args, cfg):
    from stocks_ml.procedure_card import CARD_PATH, README_PATH, write_card
    write_card()
    print(f"wrote {CARD_PATH} and the champion block of {README_PATH} from models/champion_spec.json "
          "and the champion walk's eval.json")


def cmd_eval(args, cfg):
    from stocks_ml.eval import run
    run(args.walk, args.incumbent, store=args.store, k=args.k, ci_draws=args.ci_draws,
        charts=not args.no_charts)


def cmd_app(args, cfg):
    from stocks_ml.app.build import build
    build(store=args.store, out=args.out)


def cmd_challenger2(args, cfg):
    from stocks_ml.challenger2 import run
    run(args.candidate, name=args.name, store=args.store, seed=args.seed, min_iters=args.min_iters,
        max_iters=args.max_iters, se_tol=args.se_tol, workers=args.workers, copies=args.copies)


def cmd_tune(args, cfg):
    from stocks_ml.tune import run
    run(args.name, trials=args.trials, copies=args.copies, batch=args.batch, workers=args.workers, seed=args.seed,
        phase=args.phase, store=args.store, universe=args.universe, cfg=cfg)


def cmd_audit_live(args, cfg):
    """Archived live rows vs the same rows in a later panel build (the leak detector)."""
    from stocks_ml.leak_audit import live_vs_rebuilt
    panel = pd.read_parquet(Path(args.panel))
    t = live_vs_rebuilt(args.live_dir, panel)
    if t.empty:
        print("nothing archived yet (the live job archives its rows from 2026-09-19 on)")
        return
    print(t.to_string(index=False))
    worst = t[t["rank_agreement"] < 0.98]
    print(f"\n{len(worst)} feature(s) with rank agreement below 0.98" + (": " + ", ".join(worst["feature"]) if len(worst) else ""))


def cmd_explain(args, cfg):
    from stocks_ml.explain import run
    a, b = (int(x) for x in args.years.split("-"))
    split = lambda v: None if v is None else [f for f in v.split(",") if f]  # noqa: E731
    run(args.store, range(a, b + 1), copies=tuple(int(x) for x in args.copies.split(",")),
        features=split(args.features), drop=split(args.drop), tag=args.tag or "")


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
        description="world | train | backtest | procedure | procedure-card | eval | app | challenge | challenge-fast | explain | r5-weekly")
    parser.add_argument("--config", default="config/config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("world", help="refresh a world store (Sharadar + SEC) and rebuild its panel")
    p.add_argument("--dir", default="data/r5_live", help="the store (default: the live world)")
    p.add_argument("--no-refresh", action="store_true", help="rebuild the panel only")
    p.add_argument("--no-sec", action="store_true", help="refresh Sharadar only")
    p.add_argument("--extras", action="store_true",
                   help="pull the extra Sharadar tables (tickers metadata, holdings, high/low, the wider "
                        "fundamentals) into the store and stop")
    p.add_argument("--top", type=int, default=None, metavar="N",
                   help="build a NEW research world at --dir for the top-N universe (domestic common stock "
                        "by market cap at each quarter end), from scratch; refuses an existing world")
    p.add_argument("--append-dv", action="store_true",
                   help="append the split-consistent dollar volume (x_dollar_vol) to the panel (append-only)")
    p.add_argument("--append-sr", action="store_true",
                   help="append sector-relative ranks of the admitted features (x_sr_*) to the panel, derived from "
                        "its own ranked columns (append-only)")
    p.add_argument("--append-sf", action="store_true",
                   help="append the Sharadar feature columns a frozen panel lacks, after verifying every "
                        "existing one recomputes exactly (append-only; never rebuilds)")
    p.add_argument("--sized", action="store_true",
                   help="with --membership-from and --top N: N names each quarter end — the index's largest N members, "
                        "or the index topped up with the largest non-members (the tuner's universe dial)")
    p.add_argument("--top-up", action="store_true",
                   help="with --membership-from and --top N: add the largest non-members by market cap at each quarter "
                        "end until the universe holds N names (the index plus the next names by size)")
    p.add_argument("--membership-from", default=None, metavar="STORE",
                   help="with --derive-from: take the membership stints from STORE instead of the top-N cut "
                        "(a control: the champion's names on the new world's tables and build)")
    p.add_argument("--derive-from", default=None, metavar="STORE",
                   help="with --top N: carve the top-N world out of a built top-M world (N <= M): tables "
                        "shared by symlink, membership cut at N, the panel rebuilt for this universe; no pull")

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
    p.add_argument("--features", default=None, help="extra panel columns, comma-separated")
    p.add_argument("--params", default=None, help="MODEL_PARAMS overrides, name=value,...")
    p.add_argument("--workers", type=int, default=7, help="fits in parallel processes (1 = in-process); STOCKS_ML_CORES=N caps the cores used")
    p.add_argument("--drop", default=None, help="admitted f_ columns withheld, comma-separated")
    p.add_argument("--train-top", type=int, default=None, metavar="N",
                   help="fit on the largest N names by market cap at each row's date (a top-N world); score every member")
    p.add_argument("--refit-every", type=int, default=1, metavar="N",
                   help="screening cadence: one fit per N weeks, every week scored (the record stays weekly)")
    p.add_argument("--copies", default=None, metavar="A-B",
                   help="walk copies (seeds) A..B instead of 1..K, e.g. 17-32 for a seed twin")

    p = sub.add_parser("challenge", help="the challenger protocol: candidate recipes vs the incumbent "
                       "(sample, every-week comparison, leak audit; --k16 adds the one look)")
    p.add_argument("--incumbent-recipe-ok", action="store_true",
                   help="accept an incumbent walked with a recipe other than the spec's (refused otherwise: the 2026-09-20 "
                        "leaky-yardstick guard)")
    p.add_argument("--out", required=True, help="directory for the candidates' walks and challenge.json")
    p.add_argument("--candidate", action="append", required=True, metavar="RECIPE",
                   help="label=..,train_years=..,features=a+b,params=name:value+..; unspecified "
                        "fields come from the incumbent's recipe (repeatable)")
    p.add_argument("--incumbent", default=None,
                   help="the incumbent's select walk (default: the spec's)")
    p.add_argument("--store", default=STORE)
    p.add_argument("--sel-start", default="2006-01-01")
    p.add_argument("--sel-end", default="2015-12-31")
    p.add_argument("--refit-every", type=int, default=1, metavar="N",
                   help="screening cadence for every walk of this run (incumbent twin, candidates, adjudication): "
                        "one fit per N weeks; the incumbent's own walk must be at the same cadence")
    p.add_argument("--adjudicate", action="store_true",
                   help="after stage 2, the challenger's winner meets the incumbent once on 2016-2019 "
                        "(neither selected there): the incumbent's extend walk, its seed twin walked on the "
                        "window, the candidate walked on the window; the model score decides")
    p.add_argument("--k16", action="store_true", help="the winner at K=16 on both segments + eval")
    p.add_argument("--workers", type=int, default=7, help="fits in parallel processes (1 = in-process); STOCKS_ML_CORES=N caps the cores used")

    p = sub.add_parser("challenge-fast", help="the prototype: candidate recipes vs the incumbent on a "
                       "stratified random sample of the selection window at K=16; ranks only")
    p.add_argument("--incumbent-recipe-ok", action="store_true",
                   help="accept an incumbent walked with a recipe other than the spec's (refused otherwise: the 2026-09-20 "
                        "leaky-yardstick guard)")
    p.add_argument("--out", required=True, help="directory for the candidates' sample walks and fast_s<seed>.json")
    p.add_argument("--candidate", action="append", required=True, metavar="RECIPE",
                   help="as for challenge (repeatable)")
    p.add_argument("--incumbent", default=None, help="the incumbent's select walk (default: the spec's)")
    p.add_argument("--store", default=STORE)
    p.add_argument("--sel-start", default="2006-01-01")
    p.add_argument("--sel-end", default="2015-12-31")
    p.add_argument("--per-year", type=int, default=26, help="weeks drawn from each year")
    p.add_argument("--seed", type=int, default=0, help="the draw's seed (the same weeks for every candidate)")
    p.add_argument("--workers", type=int, default=7, help="fits in parallel processes (1 = in-process); STOCKS_ML_CORES=N caps the cores used")
    p.add_argument("--refit-every", type=int, default=1, metavar="N",
                   help="screening cadence: the sample is drawn in runs of N consecutive weeks, one fit per run; "
                        "the incumbent's walk must be at the same cadence")
    p.add_argument("--k", type=int, default=16, help="copies per candidate (the null is calibrated at this K)")

    p = sub.add_parser("backtest", help="a walk through the strategy: the table vs the S&P 500 at the "
                       "walk's own settings (the spec's for the champion; the procedure's decision on "
                       "2006-2015 for any other walk; or the flags)")
    p.add_argument("--preds", nargs="+", required=True, help="walk segments (preds.parquet files)")
    p.add_argument("--store", default=STORE)
    p.add_argument("--k", type=int, default=None, help="copies to average (default: all)")
    p.add_argument("--book", type=int, default=None)
    p.add_argument("--floor", type=_floor, default=None)
    p.add_argument("--stop", type=_optional_float, default=None, help="e.g. -0.25 or none")
    p.add_argument("--cap", type=_optional_int, default=None, help="names per sector, or none")
    p.add_argument("--vol-cut", dest="vol_cut", default=None, help="none | abs | abs_or_sector (ledger.VOL_CUTS)")
    p.add_argument("--name", default="walk")
    p.add_argument("--rolling", default=None, metavar="YEARS|expanding",
                   help="the rolling procedure: re-decide the layers at every rank week on the "
                        "trailing YEARS (or everything so far); graded on the selection window only")
    p.add_argument("--cadence", type=int, default=1, help="decide every Nth rank week (rolling)")
    p.add_argument("--min-years", type=int, default=3, help="expanding: years before the first decision")
    p.add_argument("--sel-start", default="2006-01-01", help="rolling: the graded span")
    p.add_argument("--sel-end", default="2015-12-31")
    p.add_argument("--rolling-out", default=None, help="rolling: the json (default <walk>/rolling/<rule>.json)")
    p.add_argument("--workers", type=int, default=None, help="rolling: parallel decision weeks")

    p = sub.add_parser("procedure", help="decide the strategy layers on a walk of the selection "
                       "window and write them, with the walk's model recipe, into the spec (+ PROCEDURE.md)")
    p.add_argument("--preds", default=None, help="<walk>/select/preds.parquet")
    p.add_argument("--store", default=STORE)
    p.add_argument("--k", type=int, default=None, help="copies (default selection.K_COPIES)")
    p.add_argument("--sel-start", default="2006-01-01")
    p.add_argument("--sel-end", default="2015-12-31")
    p.add_argument("--check", action="store_true",
                   help="recompute and compare; write nothing; exit 1 if the spec drifted")
    p.add_argument("--lookback", nargs="+", default=None, metavar="ROLLING_JSON",
                   help="choose among rolling rules (backtest --rolling outputs) on [--sel-start, "
                        "--sel-end] by the rule's compounded %/yr; then the one look at the choice")

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

    p = sub.add_parser("explain", help="Shapley feature importance of the champion: one fit per year, "
                       "exact TreeSHAP on the scored members; reports/champion_shap.{png,md}")
    p.add_argument("--years", default="2007-2024", help="A-B: one fit at each year's first rank week")
    p.add_argument("--copies", default="1", help="copies (seeds) to average, comma-separated")
    p.add_argument("--store", default=STORE)
    p.add_argument("--features", default=None, help="a variant: extra panel columns, comma-separated")
    p.add_argument("--drop", default=None, help="a variant: admitted f_ columns withheld, comma-separated")
    p.add_argument("--tag", default=None, help="a variant writes reports/champion_shap_<tag>.* instead")

    p = sub.add_parser("challenger2", help="the champion vs one altered recipe, paired on random weeks and seeds "
                       "until the running means converge; challenger2_convergence/<name>.png")
    p.add_argument("--candidate", required=True, metavar="RECIPE",
                   help="what differs from the champion, e.g. 'params=max_depth:5', 'train_years=12', 'label=label_4w_sector_log'")
    p.add_argument("--name", default=None, help="experiment name (default: from the candidate text)")
    p.add_argument("--store", default=STORE)
    p.add_argument("--seed", type=int, default=0, help="the draw's seed (weeks and model seeds)")
    p.add_argument("--min-iters", type=int, default=50)
    p.add_argument("--max-iters", type=int, default=500)
    p.add_argument("--se-tol", type=float, default=0.35, help="stop when the paired difference's SE is at or below this (pp per hold)")
    p.add_argument("--workers", type=int, default=4, help="paired fits in parallel processes; STOCKS_ML_CORES=N caps the cores")
    p.add_argument("--copies", type=int, default=1, help="seeded copies per model per iteration, averaged (16 = the deployed ensemble)")

    p = sub.add_parser("tune", help="hyperparameter search inside challenger2: trials vs the champion on common weeks and "
                       "seeds, pruned; challenger2_convergence/tune_<name>.png, data/tune/<name>.db (resumable)")
    p.add_argument("--name", required=True)
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--copies", type=int, default=4, help="seeded copies per model per week")
    p.add_argument("--batch", type=int, default=26, help="weeks per batch between prune checks")
    p.add_argument("--phase", type=int, default=0, help="which of the four 4-week phases supplies the weeks (non-overlapping holds)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--store", default=STORE)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--universe", action="store_true",
                   help="also search the universe (universe_n names around the index, carved on demand), the training "
                        "window and the label family; the champion stays the deployed model on its own data")

    p = sub.add_parser("audit-live", help="compare the rows the live job scored (its archive) with the same rows "
                       "in a later panel build: a feature whose history is rewritten by later data is a leak")
    p.add_argument("--live-dir", default="data/r5_live")
    p.add_argument("--panel", default="data/r5_live/panel_sf.parquet", help="the later build to compare against")

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
    if args.command in ("challenge", "challenge-fast") and not args.incumbent:
        import json
        from stocks_ml.procedure import SPEC_PATH
        args.incumbent = json.loads(SPEC_PATH.read_text())["procedure"]["preds"]["path"]
    cfg = load_config(args.config)
    {"world": cmd_world, "train": cmd_train, "backtest": cmd_backtest, "audit-live": cmd_audit_live,
     "challenger2": cmd_challenger2, "tune": cmd_tune,
     "procedure": cmd_procedure, "procedure-card": cmd_procedure_card,
     "eval": cmd_eval, "app": cmd_app, "challenge": cmd_challenge,
     "challenge-fast": cmd_challenge_fast, "explain": cmd_explain,
     "r5-weekly": cmd_r5_weekly}[args.command](args, cfg)


if __name__ == "__main__":
    main()
