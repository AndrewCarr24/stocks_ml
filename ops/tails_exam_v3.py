"""Third and final sample for the tails bundle: 116 random in-between weeks.

tails_exam_v1 (114 spaced weeks): top-6 paired diff +0.51%/4w, t +1.03.
tails_exam_v2 (the other 118 spaced weeks, independent): +0.90%, t +1.59;
pooled 232: +0.71%, t +1.88 -- under the 2.0 bar, effect size held. The
owner's rule for this round (2026-09-04): add 116 more weeks, picked at random
from the ~730 Fridays of 2006-01-06 -> 2024-06-14 that neither exam ran; if the
effect remains and the t-stat gets above 2, the bundle is kept.

Pre-registered tails_exam_v3 (2026-09-04): 116 weeks drawn uniformly without
replacement (numpy default_rng seed 3) from the panel Fridays in
[2006-01-01, 2024-06-14] not in v1 or v2; identical design (same bundle,
selection.ensemble_preds K=4, champion params, 5y window, label_4w purge 35,
WITH vs WITHOUT on identical weeks/seeds; raw 4w open-to-open returns, no
costs). These weeks sit between the spaced ones, so 4-week labels overlap:
every t below is a calendar HAC t -- Bartlett kernel on calendar distance with
a 28-day bandwidth (pairs >= 28 days apart get zero weight; on the spaced
232 it reduces to the plain t). Statistics: REPLICATION = the 116 new weeks
alone (diff, t, reported); PRIMARY = pooled v1 + v2 + v3 (348 weeks): PASS iff
the pooled top-6 paired HAC t > 2 AND the new weeks' top-6 diff > 0. This is
the third look at one hypothesis and the pooled t includes the two earlier
samples; the owner set the rule knowing that. PASS -> the bundle is admitted
and adoption follows the standing structural re-selection procedure (full
walk, champion spec) on the owner's go; FAIL -> closed for good.

  PYTHONPATH=src:. .venv/bin/python ops/tails_exam_v3.py --register
  PYTHONPATH=src:. .venv/bin/python ops/tails_exam_v3.py               # ~15 min; rows under data/experiments/tails_exam_v3
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import stocks_ml.selection as sel
from ops.leverage_exam import HI, LO, one_week
from ops.tails_exam import FEATURES, HORIZON, OUT as OUT_V1, T_BAR, TRAIN_YEARS, panels
from ops.tails_exam_ext import OUT as OUT_V2
from stocks_ml.models.trials import record_trials

NAME = "tails_exam_v3"
OUT = Path("data/experiments/tails_exam_v3")
REPORT = Path("reports/tails_exam_v3.md")
N_NEW, SEED, BANDWIDTH_DAYS = 116, 3, 28
SPEC = (f"third sample for the tails bundle {FEATURES}: {N_NEW} weeks drawn uniformly without replacement (seed "
        f"{SEED}) from the panel Fridays in [{LO.date()}, {HI.date()}] not in tails_exam_v1/v2; identical design "
        f"(selection.ensemble_preds K=4, champion params, {TRAIN_YEARS}y window, label_4w purge 35, WITH vs WITHOUT "
        f"on identical weeks/seeds; raw 4w returns, no costs). Labels overlap, so every t is a calendar HAC t "
        f"(Bartlett kernel, {BANDWIDTH_DAYS}-day bandwidth; equals the plain t on the spaced 232). REPLICATION = new "
        f"weeks alone (reported); PRIMARY = pooled v1+v2+v3 (~348 weeks): PASS iff pooled top-6 paired t > {T_BAR} "
        f"AND the new weeks' top-6 diff > 0. Third look at one hypothesis, pooled with the earlier samples, by the "
        f"owner's rule. PASS -> bundle admitted, adoption via the standing structural re-selection on the owner's "
        f"go; FAIL -> closed for good.")


def register():
    record_trials([{"kind": "preregistration", "name": NAME, "notes": SPEC}])
    print("registered", NAME)


def prior_rows():
    dfs = []
    for p in (OUT_V1, OUT_V2):
        d = pd.read_parquet(p / "rows.parquet")
        d["week"] = pd.to_datetime(d["week"])
        dfs.append(d)
    return pd.concat(dfs).sort_values("week").reset_index(drop=True)


def new_weeks(ctx):
    done = set(prior_rows()["week"])
    pool = [w for w in ctx.weeks if LO <= w <= HI and w not in done]
    pick = np.sort(np.random.default_rng(SEED).choice(len(pool), N_NEW, replace=False))
    return [pool[i] for i in pick]


def hac_t(d: pd.Series, weeks: pd.Series, bandwidth_days: int = BANDWIDTH_DAYS) -> float:
    """t of the mean of d with a Bartlett kernel on calendar distance."""
    x = d.to_numpy(dtype=float) - d.mean()
    days = weeks.to_numpy(dtype="datetime64[D]").astype(np.int64)
    dist = np.abs(days[:, None] - days[None, :])
    w = np.clip(1 - dist / bandwidth_days, 0, None)
    var = (x[:, None] * x[None, :] * w).sum() / len(x) ** 2
    return float(d.mean() / np.sqrt(var)) if var > 0 else np.nan


def paired(df, stat):
    d = df[f"{stat}_with"] - df[f"{stat}_without"]
    return {"stat": stat, "without": df[f"{stat}_without"].mean(), "with": df[f"{stat}_with"].mean(),
            "diff": d.mean(), "t": hac_t(d, df["week"]), "n": len(d), "wins": float((d > 0).mean())}


def block(df, title):
    res = pd.DataFrame([paired(df, s) for s in ("top6", "top10", "hit10")])
    lines = [f"## {title}: {len(df)} weeks {df.week.min().date()} -> {df.week.max().date()}", "",
             f"SPY 4w mean {df.spy.mean():+.2%}, member mean {df.rand_mean.mean():+.2%}.", "",
             "| statistic | without | with | diff (with - without) | HAC t | weeks with > without |",
             "|---|---|---|---|---|---|"]
    for r in res.to_dict("records"):
        f = (lambda v: f"{v:+.2%}") if r["stat"] != "hit10" else (lambda v: f"{v:.3f}")
        lines.append(f"| {r['stat']}{' (primary)' if r['stat'] == 'top6' else ''} | {f(r['without'])} | "
                     f"{f(r['with'])} | {f(r['diff'])} | {r['t']:+.2f} | {r['wins']:.0%} |")
    d = df["top6_with"] - df["top6_without"]
    era = {k: df[m] for k, m in (("2006-12", df.week < "2013-01-01"), ("2013-24", df.week >= "2013-01-01"))}
    lines += ["", "Top-6 by era: " + "; ".join(
        f"{k} diff {(e.top6_with - e.top6_without).mean():+.2%}, t "
        f"{hac_t(e.top6_with - e.top6_without, e.week):+.2f} (n {len(e)})" for k, e in era.items() if len(e) > 2)
        + f". Plain-t 95% CI on the top-6 diff {d.mean() - 1.96 * d.std(ddof=1) / len(d) ** 0.5:+.2%} .. "
          f"{d.mean() + 1.96 * d.std(ddof=1) / len(d) ** 0.5:+.2%} (narrower than the HAC interval).", ""]
    return res, lines


def report(new, pooled):
    r_new, l_new = block(new, "Replication (the 116 new weeks only)")
    r_all, l_all = block(pooled, "Primary (pooled v1 + v2 + v3)")
    t_all = float(r_all.set_index("stat").loc["top6", "t"])
    t_new = float(r_new.set_index("stat").loc["top6", "t"])
    d_new = float(r_new.set_index("stat").loc["top6", "diff"])
    d_all = float(r_all.set_index("stat").loc["top6", "diff"])
    passed = bool(t_all > T_BAR and d_new > 0)
    verdict = (f"PASS: pooled top-6 paired HAC t {t_all:+.2f} > {T_BAR} and the new weeks' diff {d_new:+.2%} > 0. "
               f"The bundle is admitted; adoption follows the standing structural re-selection (full walk with the "
               f"bundle, champion spec) on the owner's go." if passed else
               f"FAIL: pooled top-6 paired HAC t {t_all:+.2f} (diff {d_all:+.2%}); new weeks' diff {d_new:+.2%} "
               f"(t {t_new:+.2f}). The rule needed t > {T_BAR} and a positive replication. The bundle is closed for "
               f"good. The champion is unchanged.")
    lines = ["# Tails exam, third sample: 116 random in-between weeks", "",
             f"Pre-registered `{NAME}` (2026-09-04). {SPEC}", "", *l_new, *l_all, "## Verdict", "", verdict, ""]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))
    record_trials([{"kind": "feature_exam", "name": NAME,
                    "config": {"features": FEATURES, "weeks_new": int(len(new)), "weeks_pooled": int(len(pooled)),
                               "seed": SEED, "hac_bandwidth_days": BANDWIDTH_DAYS, "horizon": HORIZON,
                               "train_years": TRAIN_YEARS},
                    **{f"{r['stat']}_diff": r["diff"] for r in r_all.to_dict("records")},
                    **{f"{r['stat']}_t": r["t"] for r in r_all.to_dict("records")},
                    "top6_diff_new": d_new, "top6_t_new": t_new,
                    "passed": passed, "notes": f"{'PASS' if passed else 'FAIL'} pooled {len(pooled)} weeks "
                    f"(new {len(new)} weeks diff {d_new:+.2%}, t {t_new:+.2f}); {REPORT}"}])


def run():
    ctx = sel.Ctx()
    without, with_bundle = panels(ctx)
    weeks = new_weeks(ctx)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rows.parquet"
    t0 = time.time()
    n = sel._stage_loop(ctx, weeks, path, lambda t: one_week(ctx, without, with_bundle, t), checkpoint=5)
    print(f"{n} weeks in {(time.time() - t0) / 60:.1f} min")
    new = pd.read_parquet(path)
    new["week"] = pd.to_datetime(new["week"])
    pooled = pd.concat([prior_rows(), new]).sort_values("week").reset_index(drop=True)
    assert pooled["week"].is_unique
    report(new, pooled)


if __name__ == "__main__":
    register() if "--register" in sys.argv else run()
