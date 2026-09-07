"""The feature screen: a window-bounded stage of `stocks-ml select`.

Decides, from the selection window alone, which candidate columns the run's
model gets on top of the standing features. Two gates, both pre-registered
here and fixed before any number is computed on a new window (AGENTS.md 9h:
feature probe -> paired model exam -> population):

  probe   every ``x_`` candidate the panel carries (features/candidates.py,
          rebuilt raw from the store for the probe: unranked, NaN where
          undefined, so the statistic sees the feature as written) and every
          PENDING_ABLATION_FEATURES column present (as the panel carries it),
          on the window's weeks whose labels end inside the window. Dense
          features (defined and non-zero on >= SPARSE of member-weeks): mean
          weekly Spearman IC vs the run's label, Newey-West t (lag = weeks
          the label spans); keep iff |t| >= T_BAR and the same IC sign in
          both halves of the window (split at its midpoint). Sparse flags:
          the weekly mean label of flagged names (weeks with >= MIN_FLAGGED
          flagged), same bar and sign rule.
  exam    the keepers, as one bundle, in the model at the run's chosen
          (horizon, window): the holdings stage is run again WITH the bundle
          (identical weeks and seeds; extra_features), and the paired
          difference in the top-6 forward return (PRIMARY; top-3 and top-10
          reported) over the window's weeks gets a calendar-HAC t (Bartlett
          kernel on calendar distance, bandwidth = the label span, so
          overlapping labels do not inflate it) — reported, not the rule.
          ADMITTED iff the top-6 book's cost-adjusted compounded %/yr on
          those weeks (selection.compounded_pct: the book layer's own
          metric, phase-averaged) is higher WITH the bundle than without —
          the standard every other layer applies (v3.1, owner 2026-09-05:
          v3's bar of HAC t > T_BAR was a stricter standard than the
          incumbent's layers ever faced; changed after seeing t 1.61 on
          2006-2015, before any grading-year number). The whole bundle is
          admitted — no within-bundle ablation, no second look. No keepers,
          or not admitted: the run proceeds on the standing features.

Order inside the run: grid -> wsweep -> holdings -> screen -> cascade. The
engine decisions (horizon, window, book) are therefore made on the standing
features; the screen tests the candidates at that engine, and admitted
features enter the holdings the risk layers (floor, stop, cap) and the grade
are decided on — the same path a passing bundle takes to the champion.

Ideas are not bounded by the screen, only statistics: the current candidates
were written from the 2006-2024 record (candidates.py), so a nested grade that
admits them is reported as a separate "+ engineered features" line with that
asterisk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.features.candidates import candidate_cols, candidate_features
from stocks_ml.features.panel import PENDING_ABLATION_FEATURES

T_BAR = 2.0            # the probe's bar on a candidate's own NW t
ADMIT = "argmax"       # the exam's rule: top-6 compounded %/yr with > without (v3.1)
MIN_NAMES = 100        # member-weeks need this many (feature, label) pairs to count
SPARSE = 0.10          # defined-and-non-zero share below this -> flag rule
MIN_FLAGGED = 3        # flagged names a week needs to count
PRIMARY = "top6"
SECONDARY = ("top3", "top10")


def label_span_days(kweeks: int) -> int:
    """Calendar days a k-week label spans on the fill basis (label week + the
    next open): the purge the walk uses for that horizon."""
    return kweeks * 7 + 7


# ----------------------------------------------------------------------------- statistics
def nw_t(x: pd.Series, lag: int) -> float:
    """Newey-West t of the mean of a weekly series (Bartlett kernel, `lag` weeks)."""
    x = x.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 10:
        return np.nan
    d = x - x.mean()
    var = d @ d / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (d[:-k] @ d[k:]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def hac_t(d: pd.Series, weeks: pd.Series, bandwidth_days: int) -> float:
    """t of the mean of d with a Bartlett kernel on calendar distance: pairs
    of weeks >= bandwidth_days apart get zero weight, so on weeks spaced by
    the bandwidth it is the plain t (population ddof)."""
    x = d.to_numpy(dtype=float) - d.mean()
    days = pd.to_datetime(weeks).to_numpy(dtype="datetime64[D]").astype(np.int64)
    dist = np.abs(days[:, None] - days[None, :])
    w = np.clip(1 - dist / bandwidth_days, 0, None)
    var = (x[:, None] * x[None, :] * w).sum() / len(x) ** 2
    return float(d.mean() / np.sqrt(var)) if var > 0 else np.nan


def weekly_ic(df: pd.DataFrame, col: str, label: str, min_n: int = MIN_NAMES) -> pd.Series:
    """Spearman(col, label) per week over rows where both are defined."""
    d = df[["date", col, label]].dropna()
    n = d.groupby("date").size()
    ics = d.groupby("date").apply(
        lambda g: g[col].corr(g[label], method="spearman") if len(g) >= min_n else np.nan,
        include_groups=False)
    return ics.reindex(n.index)


def flagged_series(df: pd.DataFrame, col: str, label: str, min_flagged: int = MIN_FLAGGED) -> pd.Series:
    """Weekly mean label of names with col != 0 (weeks with >= min_flagged)."""
    d = df.loc[df[col].fillna(0) != 0, ["date", label]].dropna()
    g = d.groupby("date")[label]
    return g.mean().where(g.size() >= min_flagged)


def summarize(ics: pd.Series, split, lag: int) -> dict:
    ics = ics.dropna()
    a, b = ics[ics.index < split], ics[ics.index >= split]
    return {"ic": float(ics.mean()) if len(ics) else np.nan, "t": nw_t(ics, lag),
            "ic_a": float(a.mean()) if len(a) else np.nan, "ic_b": float(b.mean()) if len(b) else np.nan,
            "weeks": int(len(ics)),
            "same_sign": bool(np.sign(a.mean()) == np.sign(b.mean())) if len(a) and len(b) else False}


# ----------------------------------------------------------------------------- probe
def screen_candidates(panel: pd.DataFrame) -> list[str]:
    """Every ``x_`` column plus every pending ``f_`` feature the panel carries."""
    return sorted(candidate_cols(panel)) + sorted(c for c in PENDING_ABLATION_FEATURES if c in panel.columns)


def probe_window(panel: pd.DataFrame, lo, hi, kweeks: int) -> pd.DataFrame:
    """Panel rows of the weeks in [lo, hi] whose labels end inside the window."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    last = hi - pd.Timedelta(days=label_span_days(kweeks))
    return panel[(panel["date"] >= lo) & (panel["date"] <= last)].reset_index(drop=True)


def probe_frame(panel: pd.DataFrame, world, lo, hi, kweeks: int, label: str, log=None) -> pd.DataFrame:
    """The probe's rows: the window's panel rows (date, ticker, label, pending
    f_ features as carried) with the panel's ``x_`` candidates rebuilt raw
    from the store — the model gets them ranked, the statistic sees them as
    written."""
    rows = probe_window(panel, lo, hi, kweeks)
    cols = ["date", "ticker", label] + [c for c in PENDING_ABLATION_FEATURES if c in rows.columns]
    frame = rows[cols]
    present = candidate_cols(panel)
    if present:
        raw = candidate_features(world, rows[["date", "ticker"]], log=log)
        frame = pd.concat([frame, raw[[c for c in raw.columns if c in present]]], axis=1)
    return frame


def probe(panel: pd.DataFrame, lo, hi, kweeks: int, label: str, candidates=None) -> pd.DataFrame:
    """One row per candidate: coverage, the statistic, its NW t, the era
    split and the keep verdict under the registered rule."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    pan = probe_window(panel, lo, hi, kweeks)
    split = lo + (hi - lo) / 2
    rows = []
    for col in (screen_candidates(pan) if candidates is None else list(candidates)):
        cov = float(pan[col].notna().mean())
        active = float((pan[col].fillna(0) != 0).mean())
        sparse = active < SPARSE
        u = summarize(flagged_series(pan, col, label) if sparse else weekly_ic(pan, col, label), split, kweeks)
        keep = bool(np.isfinite(u["t"]) and abs(u["t"]) >= T_BAR and u["same_sign"])
        rows.append({"feature": col, "coverage": cov, "active": active, "sparse": sparse, **u, "keep": keep})
    return pd.DataFrame(rows, columns=["feature", "coverage", "active", "sparse", "ic", "t", "ic_a", "ic_b",
                                       "weeks", "same_sign", "keep"])


def keepers(res: pd.DataFrame) -> list[str]:
    return [] if res.empty else res.loc[res["keep"], "feature"].tolist()


# ----------------------------------------------------------------------------- exam
def paired_rows(without: pd.DataFrame, with_: pd.DataFrame, lo, hi, kweeks: int) -> pd.DataFrame:
    """The two holdings frames joined on the window's weeks whose forward
    returns end inside the window: week, spy, rand_mean, top{k}_without/with."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    last = hi - pd.Timedelta(days=label_span_days(kweeks))
    cols = ["week", "spy", "rand_mean", "top3", "top6", "top10"]
    a = without[(without.week >= lo) & (without.week <= last)][cols]
    b = with_[(with_.week >= lo) & (with_.week <= last)][cols[:1] + cols[3:]]
    df = a.merge(b, on="week", suffixes=("_without", "_with")).sort_values("week").reset_index(drop=True)
    return df


def paired(df: pd.DataFrame, stat: str, bandwidth_days: int) -> dict:
    d = df[f"{stat}_with"] - df[f"{stat}_without"]
    return {"stat": stat, "without": float(df[f"{stat}_without"].mean()), "with": float(df[f"{stat}_with"].mean()),
            "diff": float(d.mean()), "t": hac_t(d, df["week"], bandwidth_days) if len(d) > 2 else np.nan,
            "n": int(len(d)), "wins": float((d > 0).mean()) if len(d) else np.nan}


def compounded(df: pd.DataFrame, col: str, kweeks: int) -> float:
    """The book layer's metric on the exam rows: cost-adjusted compounded %/yr
    of `col` held kweeks, averaged over the kweeks phases."""
    from stocks_ml.selection import compounded_pct      # selection imports this module
    return compounded_pct(df, col, kweeks, df["week"].min(), df["week"].max())


def exam(df: pd.DataFrame, kweeks: int) -> dict:
    """Paired statistics of with vs without on the exam rows and the verdict:
    admitted iff the primary book compounds faster WITH the bundle (ADMIT)."""
    bw = kweeks * 7
    stats = {s: paired(df, s, bw) for s in (PRIMARY, *SECONDARY)}
    cmp = {arm: compounded(df, f"{PRIMARY}_{arm}", kweeks) for arm in ("without", "with")}
    return {"stats": stats, "primary": PRIMARY, "rule": ADMIT, "compounded_pct": cmp,
            "hac_bandwidth_days": bw, "passed": bool(cmp["with"] > cmp["without"])}


# ----------------------------------------------------------------------------- report
def _pct(v):
    return f"{v:+.2%}"


def report_lines(name, window, kweeks, label, res, keep, ex, df) -> list[str]:
    lo, hi = window
    lines = [f"# Feature screen: {name}", "",
             f"Selection window {pd.Timestamp(lo).date()} -> {pd.Timestamp(hi).date()}; label `{label}`; "
             f"probe weeks end inside the window (last rank date {pd.Timestamp(hi).date()} minus "
             f"{label_span_days(kweeks)} days). Rule (registered, feature_screen.py): dense features keep iff "
             f"|NW t| >= {T_BAR} (lag {kweeks}) and the same IC sign in both halves of the window; sparse flags "
             f"(active < {SPARSE:.0%}) use the weekly mean label of flagged names. Keepers are examined as one "
             f"bundle at the run's engine; ADMITTED iff the top-6 book's cost-adjusted compounded %/yr on the exam "
             f"weeks is higher with the bundle than without (the book layer's metric; v3.1). The paired "
             f"calendar-HAC t is reported.", "",
             "## Probe", "",
             "| feature | defined / active | stat | value | NW t | first half | second half | weeks | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in res.itertuples():
        lines.append(f"| {r.feature} | {r.coverage:.0%} / {r.active:.0%} | {'flag mean' if r.sparse else 'IC'} | "
                     f"{r.ic:+.4f} | {r.t:+.1f} | {r.ic_a:+.4f} | {r.ic_b:+.4f} | {r.weeks} | "
                     f"{'KEEP' if r.keep else 'drop'} |")
    lines += ["", f"Keepers: {', '.join(keep) if keep else 'none'}.", ""]
    if ex is not None:
        lines += ["## Exam", "",
                  f"{len(df)} weeks {df.week.min().date()} -> {df.week.max().date()}; SPY mean {_pct(df.spy.mean())}, "
                  f"member mean {_pct(df.rand_mean.mean())}; forward returns on the fill basis, raw. Calendar-HAC t, "
                  f"{ex['hac_bandwidth_days']}-day bandwidth.", "",
                  "| statistic | without | with | diff (with - without) | HAC t | weeks with > without |",
                  "|---|---|---|---|---|---|"]
        for s, r in ex["stats"].items():
            lines.append(f"| {s}{' (primary)' if s == PRIMARY else ''} | {_pct(r['without'])} | {_pct(r['with'])} | "
                         f"{_pct(r['diff'])} | {r['t']:+.2f} | {r['wins']:.0%} |")
        c = ex["compounded_pct"]
        lines += ["", f"Top-6 cost-adjusted compounded %/yr on these weeks: without {c['without']:.2f}, "
                  f"with {c['with']:.2f} (paired HAC t {ex['stats'][PRIMARY]['t']:+.2f}).", "",
                  "## Verdict", "",
                  (f"ADMITTED: the top-6 book compounds faster with the bundle ({c['with']:.2f} vs "
                   f"{c['without']:.2f} %/yr); {keep} enters this run." if ex["passed"] else
                   f"NOT ADMITTED: the top-6 book does not compound faster with the bundle ({c['with']:.2f} vs "
                   f"{c['without']:.2f} %/yr); the run proceeds on the standing features."), ""]
    else:
        lines += ["## Verdict", "", "No keeper cleared the probe; nothing is admitted.", ""]
    return lines


def write_screen(out: Path, name, window, kweeks, label, res, keep, ex, df, report_path=None) -> dict:
    """screen.json (machine) and the markdown report; returns the summary."""
    lo, hi = window
    summary = {"name": name, "window": [str(pd.Timestamp(lo).date()), str(pd.Timestamp(hi).date())],
               "label": label, "kweeks": kweeks, "rule": {"t_bar": T_BAR, "admit": ADMIT, "sparse": SPARSE,
                                                          "min_flagged": MIN_FLAGGED, "min_names": MIN_NAMES},
               "candidates": res["feature"].tolist(), "keepers": keep,
               "probe": res.to_dict("records"),
               "exam": None if ex is None else {**ex, "weeks": int(len(df))},
               "admitted": keep if (ex is not None and ex["passed"]) else []}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "screen.json").write_text(json.dumps(summary, indent=2, default=_json))
    lines = report_lines(name, window, kweeks, label, res, keep, ex, df)
    (Path(out) / "screen.md").write_text("\n".join(lines))
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text("\n".join(lines))
    return summary


def _json(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    raise TypeError(type(v))


def load_screen(out) -> dict | None:
    p = Path(out) / "screen.json"
    return json.loads(p.read_text()) if p.exists() else None
