"""Stage-2 model exam for the tails-probe keepers.

tails_features_probe_v1 screened 30 candidates built from both tails of the
2006-2024 OOS record plus the 18 never-ablated pending panel features; eight
cleared the registered bar (|NW t| >= 2, same IC sign in 2002-12 and 2013-24):
the expected-earnings window, log nominal price, cash runway, 8-K officer-change
and distress counts, the mean of the last four earnings-day reactions,
accruals, and f_overnight_12w. None is a rebranded existing feature (max
weekly Spearman with any panel feature 0.57). About one of the eight is the
expected false keeper of a 48-way screen; this exam is the second gate.

Pre-registered tails_exam_v1 (2026-09-04): the eight, ranked per week and
neutral-filled like every other feature, are added as one bundle to the
champion's inputs. On the same ~115 sampled weeks 2006-01 -> 2024-06-14 as
leverage_exam_v1 (selection.sample_weeks seed 11, >= 28 days apart, thinned
seed 7) the canonical K=4 ensemble (selection.ensemble_preds: champion params,
5-year window, label_4w, purge 35) is fit fresh per week WITH and WITHOUT the
bundle, identical weeks and seeds. Statistics: paired difference (with -
without) in the top-6 4-week return (PRIMARY: the champion holds six), the
top-10 4-week return and hit10; PASS iff the primary's paired t > 2 (plain t,
the weeks are independent). Raw open-to-open returns, no costs: a model-layer
test. A pass on the sample earns a within-bundle ablation and then a population
confirmation before anything touches the champion; a fail closes the bundle.

  PYTHONPATH=src:. .venv/bin/python ops/tails_exam.py --register
  PYTHONPATH=src:. .venv/bin/python ops/tails_exam.py [--weeks N]   # rows checkpointed under data/experiments/tails_exam_v1
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

import stocks_ml.selection as sel
from ops.leverage_exam import HI, LO, exam_weeks, one_week, paired
from ops.tails_features_probe import candidate_features
from stocks_ml.data.store import DataStore
from stocks_ml.features.ranking import rank_normalize
from stocks_ml.models.trials import record_trials

NAME = "tails_exam_v1"
OUT = Path("data/experiments/tails_exam_v1")
REPORT = Path("reports/tails_exam.md")
BUNDLE = {"c_earn_window": "f_earn_window", "c_price_level": "f_price_level",
          "c_cash_runway": "f_sf_cash_runway", "c_8k_officer_26w": "f_8k_officer_26w",
          "c_earn_react_mean4": "f_earn_react_mean4", "c_8k_distress_26w": "f_8k_distress_26w",
          "c_accrual": "f_sf_accrual"}
PENDING_COPY = {"f_overnight_12w": "f_overnight_12w_exam"}   # already ranked in the panel; copied past the PENDING gate
FEATURES = list(BUNDLE.values()) + list(PENDING_COPY.values())
HORIZON, TRAIN_YEARS, T_BAR = "4w", 5, 2.0
SPEC = (f"add {FEATURES} (the tails_features_probe_v1 keepers; ranked per week, neutral-filled) as one bundle "
        f"to the champion inputs; ~115 sampled weeks {LO.date()} -> {HI.date()} "
        f"(leverage_exam_v1's weeks: selection.sample_weeks seed 11, >=28d apart, thinned seed 7); "
        f"selection.ensemble_preds K=4, champion params, {TRAIN_YEARS}y window, label_4w purge 35, fit fresh per "
        f"week WITH and WITHOUT the bundle on identical weeks/seeds; statistics = paired diff (with - without) in "
        f"top-6 4w return (PRIMARY: the champion holds six), top-10 4w return and hit10 (secondary, reported); "
        f"PASS iff the primary's paired t > {T_BAR}; raw returns, no costs. Sample pass -> within-bundle ablation, "
        f"then population confirmation before adoption; fail -> the bundle is closed.")


def register():
    record_trials([{"kind": "preregistration", "name": NAME, "notes": SPEC}])
    print("registered", NAME)


def panels(ctx):
    """(without, with): the champion panel and the same panel plus the bundle."""
    world = DataStore("data/sharadar_world2000")
    feats = candidate_features(world, ctx.pan[["date", "ticker"]])[list(BUNDLE)].rename(columns=BUNDLE)
    with_bundle = pd.concat([ctx.pan, feats], axis=1)
    with_bundle = rank_normalize(with_bundle, list(BUNDLE.values()))
    for src, dst in PENDING_COPY.items():
        with_bundle[dst] = with_bundle[src]
    return ctx.pan, with_bundle


def report(df):
    res = pd.DataFrame([paired(df, s) for s in ("top6", "top10", "hit10")])
    passed = bool(res.set_index("stat").loc["top6", "t"] > T_BAR)
    lines = ["# Tails exam: the eight probe keepers in the champion's inputs", "",
             f"Pre-registered `{NAME}` (2026-09-04). {SPEC}", "",
             f"{len(df)} weeks {df.week.min().date()} -> {df.week.max().date()}; SPY 4w mean "
             f"{df.spy.mean():+.2%}, member mean {df.rand_mean.mean():+.2%}. Returns are 4-week, "
             f"open-to-open, raw.", "",
             "| statistic | without | with | diff (with - without) | t | weeks with > without |",
             "|---|---|---|---|---|---|"]
    for r in res.to_dict("records"):
        f = (lambda v: f"{v:+.2%}") if r["stat"] != "hit10" else (lambda v: f"{v:.3f}")
        lines.append(f"| {r['stat']}{' (primary)' if r['stat'] == 'top6' else ''} | {f(r['without'])} | "
                     f"{f(r['with'])} | {f(r['diff'])} | {r['t']:+.2f} | {r['wins']:.0%} |")
    d = df["top6_with"] - df["top6_without"]
    era = {k: d[m] for k, m in (("2006-12", df.week < "2013-01-01"), ("2013-24", df.week >= "2013-01-01"))}
    lines += ["", "Primary by era: " + "; ".join(
        f"{k} diff {v.mean():+.2%}, t {v.mean() / (v.std(ddof=1) / len(v) ** 0.5):+.2f} (n {len(v)})"
        for k, v in era.items() if len(v) > 2), "",
        "## Verdict", "",
        (f"PASS on the sample (primary top-6 paired t > {T_BAR}): within-bundle ablation next, then "
         f"population confirmation, on the owner's go." if passed else
         f"FAIL: the primary top-6 paired diff does not clear t > {T_BAR}. The bundle adds nothing the "
         f"trees can use at a 4-week horizon; it is closed. The champion is unchanged."), ""]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))
    record_trials([{"kind": "feature_exam", "name": NAME,
                    "config": {"features": FEATURES, "weeks": int(len(df)), "horizon": HORIZON,
                               "train_years": TRAIN_YEARS},
                    **{f"{r['stat']}_diff": r["diff"] for r in res.to_dict("records")},
                    **{f"{r['stat']}_t": r["t"] for r in res.to_dict("records")},
                    "passed": passed, "notes": f"{'PASS' if passed else 'FAIL'} on {len(df)} sampled "
                    f"weeks; {REPORT}"}])
    return res


def run(n_weeks=None):
    ctx = sel.Ctx()
    without, with_bundle = panels(ctx)
    weeks = exam_weeks(ctx)[: n_weeks or None]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rows.parquet"
    t0 = time.time()
    n = sel._stage_loop(ctx, weeks, path, lambda t: one_week(ctx, without, with_bundle, t), checkpoint=5)
    print(f"{n} weeks in {(time.time() - t0) / 60:.1f} min")
    df = pd.read_parquet(path)
    df["week"] = pd.to_datetime(df["week"])
    return report(df)


if __name__ == "__main__":
    if "--register" in sys.argv:
        register()
    else:
        n = int(sys.argv[sys.argv.index("--weeks") + 1]) if "--weeks" in sys.argv else None
        run(n)
