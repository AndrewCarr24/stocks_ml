"""The champion with the screened bundle, graded on its recorded basis.

The owner adopted the 7-feature engineered bundle into the champion on
2026-09-05 (models/champion_spec.json `features`; the feature screen under
rule v3.1, data/experiments/nested3_v2/screen.md). The champion's official
record ($1,553, ledger champion_r5_7030_cap2_regraded) is the clean 5y walk
from 2001-06 graded as deployed; this script grades the same configuration
(4w / 5y / top-6 / 70-30 trend ballast / no stop / cap 2) on the with-bundle
walk extended over the same span, so the two records share a basis, and
puts the bundle line on the package charts.

  grade    data/experiments/champion_2006_2024/holdings_4w_5y_xeeaf48_s0.parquet
           (the bundle walk: nested3_v2's 2006-2024 rows extended back to 2001-06-01)
           simulated by selection.simulate on the live ledger's rules, ranks before
           the holdout, the live-emulation windows; paired with the clean record on
           the clean walk's weeks; the nested cascade's own with-bundle pick
           (top-3 / half-gate) as the option it is
           -> champion_bundle_grades.json, reports/champion_bundle_regrade.md,
              the `champion_bundle` column of chart_series.csv
  charts   the two package PNGs with the bundle line (needs matplotlib)

Run from the repo root: .venv/bin/python ops/champion_bundle_regrade.py {grade,charts,all}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
NEW = Path("data/experiments/champion_2006_2024")
SPEC = Path("models/champion_spec.json")
REPORT = Path("reports/champion_bundle_regrade.md")
HI = pd.Timestamp("2024-07-18")                       # holdout starts 2024-07-19
CHAMPION = dict(horizon="4w", book=6, cap=2, stop=None, floor="70/30")
WINDOWS = {"pre_holdout": ("2006", "2025"), "2006_2015": ("2006", "2016"), "2016_2024": ("2016", "2025"),
           "2006_2012": ("2006", "2013"), "2013_2024": ("2013", "2025"), "2021_2024": ("2021", "2025")}
SPAN = {"pre_holdout": "2006-01 -> 2024-07 (pre-holdout)", "2006_2015": "2006-2015", "2016_2024": "2016-01 -> 2024-07",
        "2006_2012": "2006-2012", "2013_2024": "2013-01 -> 2024-07", "2021_2024": "2021-01 -> 2024-07"}


def campaign():
    spec = importlib.util.spec_from_file_location("regrade_campaign", HERE / "regrade_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def walks():
    import stocks_ml.selection as sel
    feats = json.loads(SPEC.read_text())["features"]
    clean = pd.read_parquet(NEW / "holdings_4w_5y_s0.parquet")
    bundle = pd.read_parquet(NEW / f"{sel.holdings_name('4w', 5, feats)}_s0.parquet")
    for h in (clean, bundle):
        h["week"] = pd.to_datetime(h.week)
    return sel, feats, clean, bundle


def grade_walk(sel, ctx, h, **kw):
    """Every window's metrics for the champion's settings (kw overrides) on
    the walk's rank weeks before the holdout; the SPY row beside each."""
    h = h[h.week < HI].sort_values("week")
    s = sel.simulate(ctx, h, **{**CHAMPION, **kw})
    spy = ctx.wret["SPY"].reindex(s.index)
    out = {w: {"config": sel.metrics(s, pd.Timestamp(a), pd.Timestamp(b)),
               "sp500": sel.metrics(spy, pd.Timestamp(a), pd.Timestamp(b))} for w, (a, b) in WINDOWS.items()}
    return {"rank_weeks": int(len(h)), "first_rank_week": str(h.week.min().date()),
            "last_rank_week": str(h.week.max().date()), **out}, s


def gl(m):
    return f"${m['terminal_100']:,.0f} ({m['cagr_pct']:+.1f}%/yr, SR {m['sharpe']:.2f}, DD {m['max_dd']:.0%})"


def grade():
    sel, feats, clean, bundle = walks()
    ctx = sel.Ctx("data/sharadar_world2000")
    res, series = {}, {}
    for key, h, kw in (("clean_recorded", clean, {}),
                       ("bundle_recorded", bundle, {}),
                       ("bundle_clean_weeks", bundle[bundle.week.isin(set(clean.week))], {}),
                       ("bundle_top3_halfgate", bundle, dict(book=3, floor="halfgate"))):
        res[key], series[key] = grade_walk(sel, ctx, h, **kw)
        print(key, res[key]["rank_weeks"], "rank weeks", res[key]["first_rank_week"], "->",
              res[key]["last_rank_week"], gl(res[key]["pre_holdout"]["config"]), flush=True)
    res["features"] = feats
    (NEW / "champion_bundle_grades.json").write_text(json.dumps(res, indent=1))
    # the bundle line for the package charts, on its own weeks (the chart's cumprod
    # treats a week a series lacks as flat, so every line ends at its record); SPY
    # runs on to the bundle walk's end, the clean lines stop at their 2024-06 cutoff
    chart = pd.read_csv(NEW / "chart_series.csv", index_col="week", parse_dates=True)
    chart = chart.drop(columns=[c for c in chart.columns if c == "champion_bundle"])
    chart = chart.join(series["bundle_recorded"].rename("champion_bundle"), how="outer")
    chart["sp500"] = chart["sp500"].fillna(ctx.wret["SPY"].reindex(chart.index))
    chart.index.name = "week"
    chart.to_csv(NEW / "chart_series.csv")
    b, bp, cl, t3 = (res[k] for k in ("bundle_recorded", "bundle_clean_weeks", "clean_recorded", "bundle_top3_halfgate"))
    md = ["# The champion with the screened bundle, as deployed", "",
          f"Generated by `ops/champion_bundle_regrade.py` on {pd.Timestamp.today().date()}. The champion's "
          "settings (4w / 5y / top-6 / 70-30 trend ballast / no stop / cap 2) graded by `selection.simulate` "
          "on the live ledger's rules (fills at the next open, 5 bp a side), on the walk that trains with the "
          f"bundle ({', '.join(feats)}; `holdings_4w_5y_xeeaf48_s0`, 2001-06 -> 2024-07-19) and on the "
          "clean walk of record (`holdings_4w_5y_s0`, 2001-06 -> 2024-06-14). Rank weeks before the "
          "holdout (2024-07-19); each window's metrics on the credited weeks inside it; SPY beside every "
          "line. Asterisk on every bundle line: the thirty candidate ideas were written after reading the "
          "whole 2006-2024 record, grading years included; only the screen's statistics were bounded to "
          "2006-2015 (data/experiments/nested3_v2/screen.md).", "",
          f"The bundle walk has {b['rank_weeks']} rank weeks before the holdout, the clean walk of record "
          f"{cl['rank_weeks']}: the clean walk lacks 8 weeks of 2010-2013 and ends 2024-06-14 (the campaign's "
          "label-end cutoff), the bundle walk runs to 2024-07-12. The paired line grades the bundle on the "
          "clean walk's weeks only.", "",
          "## Champion settings, every window", "",
          "| window | clean (record) | bundle* (record) | bundle* on the clean weeks | SPY |", "|---|---|---|---|---|"]
    for w in WINDOWS:
        md.append(f"| {SPAN[w]} | {gl(cl[w]['config'])} | {gl(b[w]['config'])} | {gl(bp[w]['config'])} | "
                  f"{gl(b[w]['sp500'])} |")
    md += ["", "SPY's row is on the bundle walk's weeks; on the clean walk's weeks it reads "
           f"{gl(cl['pre_holdout']['sp500'])} over 2006-01 -> 2024-06.", "",
           "## The nested cascade's own with-bundle pick (top-3 / half-gate / no stop / cap 2), same walk", "",
           "| window | top-3 half-gate, bundle* | champion settings, bundle* | SPY |", "|---|---|---|---|"]
    for w in WINDOWS:
        md.append(f"| {SPAN[w]} | {gl(t3[w]['config'])} | {gl(b[w]['config'])} | {gl(t3[w]['sp500'])} |")
    md += ["", "Selected by the nested cascade on 2006-2015 with the bundle (nested3_features_v2_frozen_config), "
           "graded once out of sample at $1,379 vs the clean line $521 vs SPY $316 (2016 -> 2024-07-19). "
           "The owner kept the champion's book and ballast (2026-09-05); this table records the option.", ""]
    REPORT.write_text("\n".join(md))
    print("\n".join(md))
    print("wrote", REPORT, "and", NEW / "champion_bundle_grades.json")


def charts():
    rc = campaign()
    rc.CHART_NOTE = "as deployed: fills at the next open, 5 bp a side"
    rc.charts()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["grade", "charts", "all"])
    step = ap.parse_args().step
    for name, fn in (("grade", grade), ("charts", charts)):
        if step in (name, "all"):
            print(f"== {name}", flush=True)
            fn()
