"""Feature probe: nine "secular decline / distress" candidates vs the 4-week label.

Motivated by the OOS losers (Big Lots, Lumen, DISH: leveraged decliners with
a dividend the market was pricing to be cut; Generac, Etsy, PayPal, SolarEdge:
post-boom hangovers). The model's leverage features are all book-based and
debt/EBITDA goes blank when EBITDA turns negative; nothing it sees is older
than 52 weeks. The candidates are computed point-in-time on the champion's
panel base (S&P members each Friday; SF1 facts usable at filing date + 1 day,
as in features/sharadar_fundamentals.py) and probed the standing way (AGENTS.md
9h): mean weekly Spearman rank IC against label_4w, Newey-West t (lag 4 for the
overlapping label), pre-holdout only. Secondary: the same IC inside the
champion's cached top-15 picks each week — would the feature re-rank what the
model actually buys?

Pre-registered decline_features_probe_v1 (2026-09-04), keep rule fixed before
the numbers: universe |t| >= 2 on 2002-01 -> 2024-06-14 AND the same IC sign in
2002-2012 and 2013-2024. Keepers go to the sampled paired model exam (owner's
go), never straight into the champion: admitting a feature is a structural
re-selection trigger.

  .venv/bin/python ops/decline_features_probe.py --register   # ledger row, before any number
  .venv/bin/python ops/decline_features_probe.py              # ~3 min; reports/decline_features_probe.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.data.store import DataStore
from stocks_ml.features.sharadar_fundamentals import _asof
from stocks_ml.models.trials import record_trials

WORLD = Path("data/sharadar_world2000")
HOLD = Path("data/experiments/champion_2006_2024/holdings_4w_5y_s0.parquet")
REPORT = Path("reports/decline_features_probe.md")
NAME = "decline_features_probe_v1"
HOLDOUT = pd.Timestamp("2024-07-19")     # no label may end on or after this
START = pd.Timestamp("2002-01-01")       # SF1 coverage is 87-99% from 2002
SPLIT = pd.Timestamp("2013-01-01")       # the two halves of the sign check
NW_LAG = 4                               # 4-week label -> 4 overlapping weeks
T_BAR = 2.0
MIN_NAMES = 100                          # universe weeks need this many pairs
MIN_PICKS = 10                           # of the 15 cached picks

CANDIDATES = {
    "c_netdebt_mcap": "net debt / market cap (debt - cash over price x shares): market leverage, explodes as the equity collapses",
    "c_ebitda_debt": "EBITDA / debt, signed: stays defined under losses (debt/EBITDA is blank there); no debt -> +/-inf by the sign of EBITDA",
    "c_divyield": "trailing 12m dividends per share / price: a double-digit yield is a cut the market expects",
    "c_fcf_cash": "trailing FCF / cash: runway (negative = burning cash)",
    "c_rev_3y": "trailing revenue vs 3 years earlier: both tails (secular decline, post-boom spike)",
    "c_margin_3y": "EBITDA margin minus 3 years earlier",
    "c_rev_vs_sector": "revenue YoY minus the same-sector member median that week (the cap's sector map): shrinking while peers grow",
    "c_hi_3y": "price / 3-year high - 1",
    "c_ret_36m": "36-month price return",
}
# already in the model, same probe, for scale (stored representation: ranked, neutral-filled)
REFERENCE = ["f_short_dtc", "f_mom_4w", "f_log_mktcap", "f_leverage", "f_sf_de",
             "f_sf_debt_ebitda", "f_sf_fcf_yield", "f_sf_revenue_yoy", "f_hi_52w", "f_mom_52w"]

SPEC = (f"{len(CANDIDATES)} candidates ({', '.join(CANDIDATES)}) computed point-in-time on the "
        f"champion panel base; statistic = mean weekly Spearman IC vs label_4w over S&P members, "
        f"weeks {START.date()} -> labels ending before {HOLDOUT.date()} (pre-holdout only), "
        f"Newey-West t lag {NW_LAG}; secondary = same IC within the cached top-15 picks. KEEP iff "
        f"universe |t| >= {T_BAR} AND same IC sign in both halves (split {SPLIT.date()}). Keepers go "
        f"to a sampled paired model exam next (owner's go), not into the champion.")


def register():
    record_trials([{"kind": "preregistration", "name": NAME, "notes": SPEC}])
    print("registered", NAME)


def nw_t(x: pd.Series, lag: int = NW_LAG) -> float:
    """Newey-West t-stat of the mean of a weekly series (Bartlett kernel)."""
    x = x.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 10:
        return np.nan
    d = x - x.mean()
    var = d @ d / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (d[:-k] @ d[k:]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def three_year(art: pd.DataFrame) -> pd.DataFrame:
    """3-year changes on ART rows (one per quarterly report): 12 reports back,
    accepted only when the report periods are 33-39 months apart."""
    art = art.sort_values(["ticker", "reportperiod"]).copy()
    g = art.groupby("ticker")
    gap = g["reportperiod"].diff(12).dt.days
    ok = (gap > 1000) & (gap < 1200)
    prev_rev = g["revenue"].shift(12)
    art["rev_3y"] = (art["revenue"] / prev_rev - 1).where(ok & (prev_rev > 0))
    art["margin_3y"] = (art["ebitdamargin"] - g["ebitdamargin"].shift(12)).where(ok)
    return art


def year_over_year(arq: pd.DataFrame) -> pd.DataFrame:
    """revenue_yoy exactly as features/sharadar_fundamentals.py builds it."""
    arq = arq.sort_values(["ticker", "reportperiod"]).copy()
    g = arq.groupby("ticker")
    gap = g["reportperiod"].diff(4).dt.days
    ok = (gap > 330) & (gap < 400)
    prev = g["revenue"].shift(4)
    arq["revenue_yoy"] = (arq["revenue"] / prev - 1).where(ok & (prev > 0))
    return arq


def candidate_features(world: DataStore, base: pd.DataFrame) -> pd.DataFrame:
    """Raw candidates on base rows (date, ticker, sector); base.index is kept."""
    prices = world.read("prices").sort_values("date")
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    dates = pd.DatetimeIndex(sorted(base["date"].unique()))
    px_d = close.reindex(close.index.union(dates)).ffill().reindex(dates)
    hi3 = close.rolling(756, min_periods=250).max().reindex(close.index.union(dates)).ffill().reindex(dates)
    old = close.shift(756).reindex(close.index.union(dates)).ffill().reindex(dates)

    def lookup(wide):
        s = wide.stack(future_stack=True)
        s.index.names = ["date", "ticker"]
        return s.reindex(pd.MultiIndex.from_frame(base[["date", "ticker"]])).to_numpy()

    px = pd.Series(lookup(px_d), index=base.index).replace(0, np.nan)
    fund = world.read("fundamentals")
    arq = year_over_year(fund[fund.dimension == "ARQ"])
    art = three_year(fund[fund.dimension == "ART"])
    a = _asof(base, arq, ["debt", "cashneq", "sharesbas", "revenue_yoy"])
    t = _asof(base, art, ["ebitda", "fcf", "dps", "rev_3y", "margin_3y"])

    out = pd.DataFrame(index=base.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        mkt = px * a["sharesbas"]
        out["c_netdebt_mcap"] = (a["debt"] - a["cashneq"]) / mkt
        debt, ebitda = a["debt"].to_numpy(), t["ebitda"].to_numpy()
        out["c_ebitda_debt"] = np.where(debt > 0, ebitda / debt,
                                        np.where(ebitda > 0, np.inf,
                                                 np.where(ebitda < 0, -np.inf, np.nan)))
        out["c_divyield"] = t["dps"] / px
        out["c_fcf_cash"] = (t["fcf"] / a["cashneq"]).where(a["cashneq"] > 0)
        out["c_rev_3y"] = t["rev_3y"]
        out["c_margin_3y"] = t["margin_3y"]
        yoy = a["revenue_yoy"]
        med = yoy.groupby([base["date"], base["sector"]]).transform("median")
        out["c_rev_vs_sector"] = yoy - med
        out["c_hi_3y"] = px / pd.Series(lookup(hi3), index=base.index) - 1
        out["c_ret_36m"] = px / pd.Series(lookup(old), index=base.index) - 1
    return out.replace([np.inf, -np.inf], [1e300, -1e300])


def weekly_ic(df: pd.DataFrame, col: str, min_n: int) -> pd.Series:
    """Spearman(col, label_4w) per week over rows where both are defined."""
    d = df[["date", col, "label_4w"]].dropna()
    n = d.groupby("date").size()
    ics = d.groupby("date").apply(
        lambda g: g[col].corr(g["label_4w"], method="spearman") if len(g) >= min_n else np.nan,
        include_groups=False)
    return ics.reindex(n.index)


def summarize(ics: pd.Series) -> dict:
    ics = ics.dropna()
    a, b = ics[ics.index < SPLIT], ics[ics.index >= SPLIT]
    return {"ic": ics.mean(), "t": nw_t(ics), "ic_a": a.mean(), "ic_b": b.mean(),
            "weeks": len(ics), "same_sign": bool(np.sign(a.mean()) == np.sign(b.mean()))
            if len(a) and len(b) else False}


def run():
    world = DataStore(str(WORLD))
    cols = ["date", "ticker", "label_4w", "label_end_date_4w"] + REFERENCE
    pan = pd.read_parquet(WORLD / "panel_sf.parquet", columns=cols)
    pan = pan[(pan["date"] >= START) & (pan["label_end_date_4w"] < HOLDOUT)].reset_index(drop=True)
    mem = world.read("membership")
    smap = dict(mem.dropna(subset=["sector"]).drop_duplicates("ticker")[["ticker", "sector"]].values)
    pan["sector"] = pan["ticker"].map(smap)
    feats = candidate_features(world, pan[["date", "ticker", "sector"]])
    pan = pd.concat([pan, feats], axis=1)

    hold = pd.read_parquet(HOLD)
    hold["week"] = pd.to_datetime(hold["week"])
    picks = hold[["week", "top15"]].assign(ticker=hold["top15"].str.split(",")).explode("ticker")
    picks = picks.rename(columns={"week": "date"}).merge(
        pan[["date", "ticker", "label_4w"] + list(CANDIDATES) + REFERENCE], on=["date", "ticker"])

    rows = []
    for col in list(CANDIDATES) + REFERENCE:
        u = summarize(weekly_ic(pan, col, MIN_NAMES))
        p = summarize(weekly_ic(picks, col, MIN_PICKS))
        cov = pan[col].notna().mean()
        keep = (abs(u["t"]) >= T_BAR) and u["same_sign"]
        rows.append({"feature": col, "candidate": col in CANDIDATES, "coverage": cov, **u,
                     "picks_ic": p["ic"], "picks_t": p["t"], "keep": keep})
    res = pd.DataFrame(rows)

    def fmt(r):
        verdict = ("KEEP" if r.keep else "drop") if r.candidate else "in the model"
        return (f"| {r.feature} | {r.coverage:.0%} | {r.ic:+.4f} | {r.t:+.1f} | {r.ic_a:+.4f} | "
                f"{r.ic_b:+.4f} | {r.picks_ic:+.3f} | {r.picks_t:+.1f} | {verdict} |")

    head = ["| feature | coverage | IC | NW t | IC 2002-12 | IC 2013-24 | picks IC | picks t | verdict |",
            "|---|---|---|---|---|---|---|---|---|"]
    kept = res[res.candidate & res.keep].feature.tolist()
    n_weeks = pan["date"].nunique()
    lines = ["# Decline / distress feature probe", "",
             f"Pre-registered `{NAME}` (2026-09-04). {SPEC}", "",
             f"Universe: {len(pan):,} member-weeks over {n_weeks} weeks, {pan['date'].min().date()} -> "
             f"{pan['date'].max().date()} (last label ends before {HOLDOUT.date()}). Picks: the champion's "
             f"cached top-15 per week ({picks['date'].nunique()} weeks). IC = mean weekly Spearman vs "
             f"label_4w; NW t = Newey-West (lag {NW_LAG}). Scale (AGENTS.md): 0.01 is real, 0.02 is good.", "",
             "## Candidates", "", *head, *[fmt(r) for r in res[res.candidate].itertuples()], "",
             "## Reference: features already in the model", "", *head,
             *[fmt(r) for r in res[~res.candidate].itertuples()], "",
             "## Verdict", "",
             (f"Keep: {', '.join(kept)}." if kept else "Keep: none.") +
             " Next step for keepers is the sampled paired model exam (champion params, same weeks, "
             "top-6 4-week return and hit10 paired vs the champion's inputs, t > 2), on the owner's go. "
             "Nothing here changes the champion.", "",
             "## Definitions", "", *[f"- `{k}`: {v}" for k, v in CANDIDATES.items()], ""]
    REPORT.write_text("\n".join(lines))
    print("\n".join(lines))

    entries = [{"kind": "feature_probe", "name": f"{NAME}:{r.feature}",
                "config": {"probe": NAME, "candidate": bool(r.candidate)},
                "ic": r.ic, "nw_t": r.t, "ic_2002_2012": r.ic_a, "ic_2013_2024": r.ic_b,
                "picks_ic": r.picks_ic, "picks_t": r.picks_t, "coverage": r.coverage,
                "keep": bool(r.keep) if r.candidate else None,
                "notes": f"{REPORT}"} for r in res.itertuples()]
    entries.append({"kind": "feature_probe", "name": NAME,
                    "notes": f"result: keep {kept or 'none'} of {len(CANDIDATES)} candidates by the "
                             f"registered rule; {REPORT}"})
    record_trials(entries)
    return res


if __name__ == "__main__":
    register() if "--register" in sys.argv else run()
