"""Stage-2 model exam for the two cash-flow leverage features.

decline_features_probe_v1 screened nine decline/distress candidates by
bivariate 4-week IC; none cleared the registered bar. The owner sent the two
same-sign near-misses on to the model layer, where interactions can matter
(bivariate IC cannot see them): EBITDA / debt (signed, t 1.6) and FCF / cash
(t 1.8). Net debt / market cap flipped sign between eras and was dropped.

Pre-registered leverage_exam_v1 (2026-09-04): the two candidates, ranked per
week and neutral-filled like every other feature, are added to the champion's
inputs. On ~115 sampled weeks 2006-01 -> 2024-06-14 (selection.sample_weeks,
>= 28 days apart so the 4-week labels do not overlap; seed 11, thinned with
seed 7) the canonical K=4 ensemble (selection.ensemble_preds: champion params,
5-year window, label_4w, purge 35) is fit fresh per week WITH and WITHOUT the
pair, identical weeks and seeds. Statistics: paired difference (with - without)
in the top-6 4-week return, the top-10 4-week return and hit10 (share of the
top 10 beating the member median); PASS iff the top-6 diff (primary: the
champion holds six) has t > 2 (plain t, the weeks are independent). Raw open-to-open returns, no costs: a model-layer test.
A pass on the sample earns a population confirmation before anything touches
the champion; a fail closes the question.

  PYTHONPATH=src .venv/bin/python ops/leverage_exam.py --register
  PYTHONPATH=src .venv/bin/python ops/leverage_exam.py [--weeks N]   # rows checkpointed under data/experiments/leverage_exam_v1
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import stocks_ml.selection as sel
from ops.decline_features_probe import candidate_features
from stocks_ml.data.store import DataStore
from stocks_ml.features.ranking import rank_normalize
from stocks_ml.models.trials import record_trials

NAME = "leverage_exam_v1"
OUT = Path("data/experiments/leverage_exam_v1")
REPORT = Path("reports/leverage_exam.md")
LO, HI = pd.Timestamp("2006-01-01"), pd.Timestamp("2024-06-14")
PAIR = {"c_ebitda_debt": "f_sf_ebitda_debt", "c_fcf_cash": "f_sf_fcf_cash"}
N_WEEKS = 115
HORIZON, TRAIN_YEARS = "4w", 5
T_BAR = 2.0
SPEC = (f"add {list(PAIR.values())} (EBITDA/debt signed, FCF/cash; ranked per week, neutral-filled) "
        f"to the champion inputs; {N_WEEKS} sampled weeks {LO.date()} -> {HI.date()} "
        f"(selection.sample_weeks seed 11, >=28d apart, thinned seed 7); selection.ensemble_preds "
        f"K=4, champion params, {TRAIN_YEARS}y window, label_4w purge 35, fit fresh per week WITH "
        f"and WITHOUT the pair on identical weeks/seeds; statistics = paired diff (with - without) "
        f"in top-6 4w return (PRIMARY: the champion holds six), top-10 4w return and hit10 (secondary, "
        f"reported); PASS iff the primary's paired t > {T_BAR}; raw returns, no costs. Sample pass -> "
        f"population confirmation before adoption; fail -> closed.")


def register():
    record_trials([{"kind": "preregistration", "name": NAME, "notes": SPEC}])
    print("registered", NAME)


def panels(ctx):
    """(without, with): the champion panel and the same panel plus the pair."""
    base = ctx.pan[["date", "ticker"]].copy()
    base["sector"] = base["ticker"].map(ctx.smap)
    world = DataStore("data/sharadar_world2000")
    feats = candidate_features(world, base)[list(PAIR)].rename(columns=PAIR)
    with_pair = pd.concat([ctx.pan, feats], axis=1)
    with_pair = rank_normalize(with_pair, list(PAIR.values()))
    return ctx.pan, with_pair


def exam_weeks(ctx):
    weeks = sel.sample_weeks(ctx.weeks, LO, HI)
    if len(weeks) > N_WEEKS:
        keep = np.sort(np.random.default_rng(7).choice(len(weeks), N_WEEKS, replace=False))
        weeks = [weeks[i] for i in keep]
    return weeks


def hit10(ctx, t, preds):
    wk = sel.week_slot(ctx.fwd[HORIZON].index, t)
    r = ctx.fwd[HORIZON].loc[wk]
    uni = [x for x in ctx.members[t] if x in r.index and not pd.isna(r[x])]
    p = preds.loc[preds.index.intersection(pd.Index(uni))]
    top = p.sort_values(ascending=False).index[:10]
    return float((r.loc[top] > r.loc[uni].median()).mean())


def one_week(ctx, without, with_pair, t):
    row = {"week": t}
    for arm, pan in (("without", without), ("with", with_pair)):
        ctx.pan = pan
        p = sel.ensemble_preds(ctx, t, HORIZON, TRAIN_YEARS)
        if p is None:
            return None
        s = sel.slice_row(ctx, t, HORIZON, p)
        if s is None:
            return None
        row[f"top6_{arm}"] = s["top6"]
        row[f"top10_{arm}"] = s["top10"]
        row[f"hit10_{arm}"] = hit10(ctx, t, p)
        row[f"top15_{arm}"] = s["top15"]
        row["spy"], row["rand_mean"] = s["spy"], s["rand_mean"]
    return row


def paired(df, stat):
    d = df[f"{stat}_with"] - df[f"{stat}_without"]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 else np.nan
    return {"stat": stat, "without": df[f"{stat}_without"].mean(), "with": df[f"{stat}_with"].mean(),
            "diff": d.mean(), "t": t, "n": len(d), "wins": float((d > 0).mean())}


def report(df):
    res = pd.DataFrame([paired(df, s) for s in ("top6", "top10", "hit10")])
    passed = bool(res.set_index("stat").loc["top6", "t"] > T_BAR)
    lines = ["# Leverage exam: EBITDA/debt + FCF/cash in the champion's inputs", "",
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
    lines += ["", "## Verdict", "",
              (f"PASS on the sample (primary top-6 paired t > {T_BAR}): population confirmation "
               f"next, on the owner's go." if passed else
               f"FAIL: the primary top-6 paired diff does not clear t > {T_BAR}. The pair adds nothing "
               f"the trees can use at a 4-week horizon; the question is closed. The champion is "
               f"unchanged."), ""]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))
    record_trials([{"kind": "feature_exam", "name": NAME, "config": {"features": list(PAIR.values()),
                    "weeks": int(len(df)), "horizon": HORIZON, "train_years": TRAIN_YEARS},
                    **{f"{r['stat']}_diff": r["diff"] for r in res.to_dict("records")},
                    **{f"{r['stat']}_t": r["t"] for r in res.to_dict("records")},
                    "passed": passed, "notes": f"{'PASS' if passed else 'FAIL'} on {len(df)} sampled "
                    f"weeks; {REPORT}"}])
    return res


def run(n_weeks=None):
    ctx = sel.Ctx()
    without, with_pair = panels(ctx)
    weeks = exam_weeks(ctx)[: n_weeks or None]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rows.parquet"
    t0 = time.time()
    n = sel._stage_loop(ctx, weeks, path, lambda t: one_week(ctx, without, with_pair, t),
                        checkpoint=5)
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
