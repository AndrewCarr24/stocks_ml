"""Extension of tails_exam_v1 to the full 4-weekly sample (~234 weeks).

tails_exam_v1 failed on 114 weeks: top-6 paired diff +0.51%/4w, t +1.03,
95% CI -0.46% .. +1.47%. The paired diff's 5.3%/4w standard deviation puts
that design's detectable effect at ~1%/4w; a +0.5% effect is invisible to it.
The owner asked for the sample to be doubled to see whether the effect size
holds.

Pre-registered tails_exam_v2 (2026-09-04): the 120 weeks of
selection.sample_weeks(2006-01-01 -> 2024-06-14; seed 11, >= 28 days apart)
that the seed-7 thinning left out of v1 -- disjoint from and >= 28 days away
from every v1 week -- are run with the identical design (same bundle, same
K=4 ensemble, same seeds, WITH vs WITHOUT). Two statistics, both fixed here:
(1) REPLICATION: the new weeks alone, the independent test -- reported with
its own diff and t; (2) PRIMARY: pooled v1 + new (~234 weeks), PASS iff the
top-6 paired t > 2. The pooled test is not independent of v1: the decision to
extend was taken after seeing v1's +0.51%, so the pooled t carries a modest
optional-continuation inflation; the replication statistic does not, and a
pooled pass with a negative replication diff is reported as such. A pass earns
a within-bundle ablation, then a population confirmation before adoption; a
fail closes the bundle for good.

  PYTHONPATH=src:. .venv/bin/python ops/tails_exam_ext.py --register
  PYTHONPATH=src:. .venv/bin/python ops/tails_exam_ext.py               # ~15 min; rows under data/experiments/tails_exam_v2
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import stocks_ml.selection as sel
from ops.leverage_exam import HI, LO, one_week, paired
from ops.tails_exam import FEATURES, HORIZON, OUT as OUT_V1, T_BAR, TRAIN_YEARS, panels
from stocks_ml.models.trials import record_trials

NAME = "tails_exam_v2"
OUT = Path("data/experiments/tails_exam_v2")
REPORT = Path("reports/tails_exam_ext.md")
SPEC = (f"extension of tails_exam_v1: the ~120 selection.sample_weeks({LO.date()} -> {HI.date()}, seed 11, >=28d "
        f"apart) weeks not in v1 (disjoint, >=28d from every v1 week), identical design (bundle {FEATURES}, "
        f"selection.ensemble_preds K=4, champion params, {TRAIN_YEARS}y window, label_4w purge 35, WITH vs WITHOUT "
        f"on identical weeks/seeds); statistics: REPLICATION = new weeks alone (diff, t, reported); PRIMARY = pooled "
        f"v1 + new (~234 weeks), PASS iff top-6 paired t > {T_BAR}; raw 4w open-to-open returns, no costs. The "
        f"pooled t is not independent of v1 (extension decided after seeing v1); the replication is. Pass -> "
        f"within-bundle ablation, then population confirmation before adoption; fail -> bundle closed for good.")


def register():
    record_trials([{"kind": "preregistration", "name": NAME, "notes": SPEC}])
    print("registered", NAME)


def new_weeks(ctx):
    done = set(pd.to_datetime(pd.read_parquet(OUT_V1 / "rows.parquet")["week"]))
    return [w for w in sel.sample_weeks(ctx.weeks, LO, HI) if w not in done]


def block(df, title):
    res = pd.DataFrame([paired(df, s) for s in ("top6", "top10", "hit10")])
    lines = [f"## {title}: {len(df)} weeks {df.week.min().date()} -> {df.week.max().date()}", "",
             f"SPY 4w mean {df.spy.mean():+.2%}, member mean {df.rand_mean.mean():+.2%}.", "",
             "| statistic | without | with | diff (with - without) | t | weeks with > without |",
             "|---|---|---|---|---|---|"]
    for r in res.to_dict("records"):
        f = (lambda v: f"{v:+.2%}") if r["stat"] != "hit10" else (lambda v: f"{v:.3f}")
        lines.append(f"| {r['stat']}{' (primary)' if r['stat'] == 'top6' else ''} | {f(r['without'])} | "
                     f"{f(r['with'])} | {f(r['diff'])} | {r['t']:+.2f} | {r['wins']:.0%} |")
    d = df["top6_with"] - df["top6_without"]
    era = {k: d[m] for k, m in (("2006-12", df.week < "2013-01-01"), ("2013-24", df.week >= "2013-01-01"))}
    lines += ["", "Top-6 by era: " + "; ".join(
        f"{k} diff {v.mean():+.2%}, t {v.mean() / (v.std(ddof=1) / len(v) ** 0.5):+.2f} (n {len(v)})"
        for k, v in era.items() if len(v) > 2)
        + f". 95% CI on the top-6 diff {d.mean() - 1.96 * d.std(ddof=1) / len(d) ** 0.5:+.2%} .. "
          f"{d.mean() + 1.96 * d.std(ddof=1) / len(d) ** 0.5:+.2%}.", ""]
    return res, lines


def report(new, pooled):
    r_new, l_new = block(new, "Replication (new weeks only)")
    r_all, l_all = block(pooled, "Primary (pooled v1 + new)")
    t_all = float(r_all.set_index("stat").loc["top6", "t"])
    t_new = float(r_new.set_index("stat").loc["top6", "t"])
    d_new = float(r_new.set_index("stat").loc["top6", "diff"])
    passed = bool(t_all > T_BAR)
    verdict = (f"PASS on the pooled sample (top-6 paired t {t_all:+.2f} > {T_BAR}); the replication alone shows "
               f"{d_new:+.2%} (t {t_new:+.2f}). Within-bundle ablation next, then population confirmation, on the "
               f"owner's go." if passed else
               f"FAIL: pooled top-6 paired t {t_all:+.2f} does not clear {T_BAR}; the replication alone shows "
               f"{d_new:+.2%} (t {t_new:+.2f}). The bundle is closed. The champion is unchanged.")
    lines = ["# Tails exam, extended: the eight probe keepers on the full 4-weekly sample", "",
             f"Pre-registered `{NAME}` (2026-09-04). {SPEC}", "", *l_new, *l_all, "## Verdict", "", verdict, ""]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))
    record_trials([{"kind": "feature_exam", "name": NAME,
                    "config": {"features": FEATURES, "weeks_new": int(len(new)), "weeks_pooled": int(len(pooled)),
                               "horizon": HORIZON, "train_years": TRAIN_YEARS},
                    **{f"{r['stat']}_diff": r["diff"] for r in r_all.to_dict("records")},
                    **{f"{r['stat']}_t": r["t"] for r in r_all.to_dict("records")},
                    "top6_diff_new": d_new, "top6_t_new": t_new,
                    "passed": passed, "notes": f"{'PASS' if passed else 'FAIL'} pooled {len(pooled)} weeks "
                    f"(replication {len(new)} weeks diff {d_new:+.2%}, t {t_new:+.2f}); {REPORT}"}])


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
    v1 = pd.read_parquet(OUT_V1 / "rows.parquet")
    for df in (new, v1):
        df["week"] = pd.to_datetime(df["week"])
    pooled = pd.concat([v1, new]).sort_values("week").reset_index(drop=True)
    assert pooled["week"].is_unique and pooled["week"].diff().dt.days.min() >= 28
    report(new, pooled)


if __name__ == "__main__":
    register() if "--register" in sys.argv else run()
