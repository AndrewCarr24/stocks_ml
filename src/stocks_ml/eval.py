"""`stocks-ml eval`: the one look at a walk — the grade, its confidence, the
falsification test, the leak audit, the charts.

A walk is a directory with two segments written by `stocks-ml train`:

    <walk>/select/preds.parquet   2006-2015, where the strategy layers are decided
    <walk>/extend/preds.parquet   2016-2024 (pre-holdout), read once

The walk's own settings come from the procedure's code on its select segment
(procedure.decide, cached as <walk>/procedure.json — the spec is not
written; adoption is `stocks-ml procedure`). Then, at K copies:

    table          $100, %/yr, Sharpe, drawdown on 2006-2015 / 2016-2024 /
                   2006-2024 beside the S&P 500, paired weekly t
    incumbent      the same for --incumbent (a walk directory), each model at
                   its own procedure-decided settings
    falsification  paired weekly excess vs the incumbent on 2016-2024:
                   t < -2 rejects the walk (pre-registered)
    leak audit     stocks_ml.leak_audit on each segment (identity gate)
    confidence     95% intervals: seed noise (resampled copies), history
                   noise (block bootstrap of the weeks), both nested
    charts         growth of $100 vs the S&P 500 on 2016-2024 (out of sample only)

Outputs: <walk>/eval.json, reports/<label>_eval.md, reports/<label>_vs_sp500_
2016_2024.png, one ledger row. Default --walk is the spec's walk (the
champion; label "champion"). Nothing at or past the holdout is read.

    stocks-ml eval                                  # the champion
    stocks-ml eval --walk data/experiments/<challenger> --incumbent <champion walk>
    stocks-ml eval --ci-draws 0 --no-charts         # the cheap part only
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.backtest import (SPANS, copies_in, load_preds, paired_t, row_vs_spy, settings_label,
                                table_md, walk_records, weekly_returns)
from stocks_ml.selection import HOLDOUT_START, K_COPIES

SPEC_PATH = Path("models/champion_spec.json")
STORE = "data/sharadar_world2000_nominal_dl"
CI_SEED_DRAWS = 200
CI_HISTORY_DRAWS = 4000
CI_NESTED_PER_SEED = 20
CI_BLOCK_WEEKS = 8
CI_RNG_SEED = 20260911
FALSIFY_T = -2.0
LABELS_4W_SHORT = {"label_4w": "4w label", "label_4w_sector": "sector-relative 4w label",
                   "label_4w_sector_log": "sector-relative 4w log label",
                   "label_4w_sector_clip": "sector-relative 4w label, clipped",
                   "label_4w_sector_rank": "sector-relative 4w rank label",
                   "label_4w_rank": "4w rank label", "label_13w_sector_rank": "sector-relative 13w rank label",
                   "label_blend_rank": "4w+13w blended rank label",
                   "label_4w_sector11_rank": "Sharadar-sector-relative 4w rank label"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def spec_walk(spec_path: Path = SPEC_PATH) -> Path:
    """The champion's walk directory, from the spec's procedure record."""
    spec = json.loads(Path(spec_path).read_text())
    return Path(spec["procedure"]["preds"]["path"]).parent.parent


def segments(walk: Path) -> list[Path]:
    paths = [Path(walk) / "select" / "preds.parquet", Path(walk) / "extend" / "preds.parquet"]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"walk {walk} lacks {missing} (stocks-ml train writes them)")
    return paths


def walk_settings(walk: Path, store: str = STORE, k: int | None = None, log=log) -> dict:
    """The strategy layers the procedure's code decides on the walk's select
    segment — cached as <walk>/procedure.json, keyed by the file's sha256."""
    from stocks_ml.procedure import decide
    walk = Path(walk)
    sel_path = walk / "select" / "preds.parquet"
    sha = hashlib.sha256(sel_path.read_bytes()).hexdigest()
    cache = walk / "procedure.json"
    if cache.exists():
        proc = json.loads(cache.read_text())
        if proc["preds"]["sha256"] == sha and (k is None or proc["k_copies"] == k) \
                and "vol_cut" in proc["decision"]:            # a cache from before the fifth layer is stale
            return proc
    proc = decide(sel_path, store, k, log=log)
    cache.write_text(json.dumps(proc, indent=1, default=str))
    return proc


def settings_of(proc: dict) -> dict:
    d = proc["decision"]
    return dict(book=d["book_size"], floor=d["floor"], stop=d["stop_loss"], cap=d["sector_cap"],
                vol_cut=d.get("vol_cut"))


def who(rec: dict, st: dict) -> str:
    """One line naming the model and its settings, for titles."""
    r = rec.get("recipe", {})
    return (f"{LABELS_4W_SHORT.get(r.get('label'), r.get('label'))} / {r.get('train_years')}y / "
            f"top-{st['book']} / {st['floor']} / cap {st['cap'] or 'none'}")


def check_complete(ctx, preds: pd.DataFrame, k: int, lo, hi) -> None:
    want = [t for t in ctx.weeks if lo <= t < hi]
    have = set(preds.week.unique())
    if any(t not in have for t in want):
        raise RuntimeError(f"the walk has {len(have)} of {len(want)} rank weeks of "
                           f"{lo.date()} -> {hi.date()}: finish it (stocks-ml train)")
    if copies_in(preds) < k:
        raise RuntimeError(f"the walk holds {copies_in(preds)} copies, K={k} asked")


def _ci_stats(r: np.ndarray, spy: np.ndarray):
    """terminal $100, CAGR %, excess CAGR vs SPY % — compounded, same weeks."""
    yrs = len(r) / 52.18
    w, s = np.prod(1 + r), np.prod(1 + spy)
    return 100 * w, 100 * (w ** (1 / yrs) - 1), 100 * ((w / s) ** (1 / yrs) - 1)


def confidence(sel, ctx, preds, st: dict, k: int, spans: dict = SPANS,
               seed_draws: int = CI_SEED_DRAWS, log=log) -> dict:
    """95% intervals: seed noise (K-copy ensembles drawn by resampling the
    copies with replacement, each re-simulated), history noise (circular
    block bootstrap of the weeks, strategy and SPY resampled on the same
    weeks) and the two nested."""
    rng = np.random.default_rng(CI_RNG_SEED)
    point = weekly_returns(sel, ctx, preds, range(1, k + 1), st)
    spy_all = ctx.wret["SPY"].reindex(point.index)
    draws, t0 = [], time.time()
    for b in range(seed_draws):
        copies = list(rng.integers(1, k + 1, size=k))
        draws.append(weekly_returns(sel, ctx, preds, copies, st).reindex(point.index))
        if (b + 1) % 25 == 0:
            log(f"  ci: {b + 1}/{seed_draws} seed draws, {(time.time() - t0) / (b + 1):.1f} s/draw")
    A_all = pd.concat(draws, axis=1)

    def block_idx(n, B):
        nb = int(np.ceil(n / CI_BLOCK_WEEKS))
        starts = rng.integers(0, n, size=(B, nb))
        idx = (starts[:, :, None] + np.arange(CI_BLOCK_WEEKS)[None, None, :]).reshape(B, -1) % n
        return idx[:, :n]

    out = {}
    names = ("terminal_100", "cagr_pct", "excess_cagr_vs_sp500_pct")
    for w, (lo, hi) in spans.items():
        m = (point.index >= lo) & (point.index < hi)
        pt, spy, A = point[m].to_numpy(), spy_all[m].to_numpy(), A_all[m].to_numpy()
        ok = np.isfinite(pt) & np.isfinite(spy) & np.isfinite(A).all(axis=1)
        pt, spy, A = pt[ok], spy[ok], A[ok]
        n = len(pt)
        p = _ci_stats(pt, spy)
        seed = np.array([_ci_stats(A[:, j], spy) for j in range(A.shape[1])])
        hist = np.array([_ci_stats(pt[i], spy[i]) for i in block_idx(n, CI_HISTORY_DRAWS)])
        idx2 = block_idx(n, CI_NESTED_PER_SEED * A.shape[1])
        nest = np.array([_ci_stats(A[i, j % A.shape[1]], spy[i]) for j, i in enumerate(idx2)])
        ex = np.log1p(pt) - np.log1p(spy)
        q = lambda a, kk: [round(float(v), 2) for v in np.percentile(a[:, kk], [2.5, 97.5])]
        out[w] = {"weeks": int(n),
                  **{nm: {"point": round(float(p[kk]), 2), "seed_95": q(seed, kk),
                          "history_95": q(hist, kk), "nested_95": q(nest, kk)}
                     for kk, nm in enumerate(names)},
                  "p_excess_positive_nested": round(float((nest[:, 2] > 0).mean()), 3),
                  "paired_weekly_log_excess_t_vs_sp500": round(float(ex.mean() / ex.std(ddof=1) * np.sqrt(n)), 2)}
    out["method"] = {"seed_draws": seed_draws, "history_draws": CI_HISTORY_DRAWS,
                     "nested_per_seed": CI_NESTED_PER_SEED, "block_weeks": CI_BLOCK_WEEKS,
                     "rng_seed": CI_RNG_SEED}
    return out


def falsification(pkg: pd.Series, inc: pd.Series, spans: dict = SPANS) -> dict:
    """Paired weekly excess of the walk over the incumbent per span; the
    pre-registered rule reads 2016-2024."""
    d = (pkg - inc.reindex(pkg.index)).dropna()
    out = {w: {"weeks": int(((d.index >= a) & (d.index < b)).sum()),
               "mean_weekly_excess_pct": round(float(d[(d.index >= a) & (d.index < b)].mean() * 100), 4),
               "t": round(paired_t(d[(d.index >= a) & (d.index < b)]), 2)}
           for w, (a, b) in spans.items()}
    t = out["2016-2024"]["t"]
    out["rule"] = f"paired weekly excess vs the incumbent on 2016-2024, t < {FALSIFY_T} rejects"
    out["verdict"] = "REJECTED" if t < FALSIFY_T else "not rejected"
    return out


def chart(df: pd.DataFrame, title: str, footer: str, lo: str, out: Path, k: int) -> None:
    """Growth of $100, the walk vs the S&P 500 from `lo`: NAV on a log axis,
    the ratio to the S&P, both drawdowns."""
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, AQUA, SURF, INK, INK2 = "#2a78d6", "#1baf7a", "#fcfcfb", "#0b0b0b", "#52514e"
    w = df[df.index >= lo]
    end = w.index[-1].strftime("%Y-%m")
    navs = {c: (1 + w[c].fillna(0)).cumprod() * 100 for c in ("walk", "sp500")}
    names = {"walk": title, "sp500": "sp500"}
    colors = {"walk": BLUE, "sp500": AQUA}
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11.5, 9), dpi=150, sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1.2, 1.0], "hspace": 0.12})
    fig.patch.set_facecolor(SURF)
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(SURF)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(True, axis="y", color="#e8e7e3", lw=0.8)
    for c, nav in navs.items():
        ax1.plot(nav.index, nav.values, color=colors[c], lw=2, label=names[c])
    ends = sorted(((np.log10(n.iloc[-1]), c) for c, n in navs.items()), reverse=True)
    ys = [y for y, _ in ends]
    for i in range(1, len(ys)):
        ys[i] = min(ys[i], ys[i - 1] - 0.07)
    for y, (_, c) in zip(ys, ends):
        ax1.annotate(f" ${navs[c].iloc[-1]:,.0f}", (navs[c].index[-1], 10 ** y),
                     color=colors[c], fontsize=9, fontweight="bold", va="center")
    ax1.set_yscale("log")
    lo_y = min(n.min() for n in navs.values()) * 0.9
    hi_y = max(n.max() for n in navs.values()) * 1.1
    ax1.set_ylim(lo_y, hi_y)
    ax1.set_yticks([t for t in (25, 50, 75, 100, 150, 200, 400, 800, 1600, 3200) if lo_y <= t <= hi_y])
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    ax1.set_title(f"growth of $100, {lo[:7]} -> {end} (pre-holdout, K={k}, next-open fills, 5 bp a side)",
                  color=INK, fontsize=12, loc="left", pad=10)
    idx = w.index
    ax1.set_xlim(idx[0], idx[-1] + (idx[-1] - idx[0]) * 0.07)
    ratio = navs["walk"] / navs["sp500"]
    ax2.plot(ratio.index, ratio.values, color=BLUE, lw=2)
    ax2.axhline(1.0, color=INK2, lw=1, ls=":")
    ax2.set_ylabel("walk / sp500", color=INK2, fontsize=9)
    ax2.annotate(f" {ratio.iloc[-1]:.2f}x", (ratio.index[-1], ratio.iloc[-1]),
                 color=BLUE, fontsize=9, fontweight="bold", va="center")
    worst = {}
    for c, nav in navs.items():
        dd = nav / nav.cummax() - 1
        ax3.fill_between(dd.index, dd.values, 0, color=colors[c], alpha=0.25, lw=0)
        ax3.plot(dd.index, dd.values, color=colors[c], lw=1)
        worst[c] = dd.min()
    ax3.text(0.99, 0.06, "   ".join(f"worst {c} {v:.0%}" for c, v in worst.items()),
             transform=ax3.transAxes, color=INK2, fontsize=8, ha="right", va="bottom")
    ax3.set_ylim(min(worst.values()) * 1.15, 0.02)
    ax3.set_ylabel("drawdown", color=INK2, fontsize=9)
    ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.text(0.01, 0.005, footer, color=INK2, fontsize=7.5)
    fig.savefig(out, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def report_md(res: dict) -> list[str]:
    """The markdown the eval prints and writes: settings line, the table,
    falsification, leak audit, the confidence table."""
    from stocks_ml.leak_audit import leak_line
    label, st, k = res["label"], res["settings"], res["k"]
    md = [f"# {label}: the one look",
          "",
          f"Walk `{res['walk']}` ({res['who']}) at K={k}, settings {settings_label(st)} decided by "
          f"`stocks-ml procedure`'s code on its 2006-2015 segment. $100 at each span's start; "
          f"pre-tax (Roth); pre-holdout only. Generated {res['evaluated_at']} by `stocks-ml eval`.",
          ""]
    md += table_md(res["table"])
    if res.get("falsification"):
        f = res["falsification"]
        md += ["", f"Falsification ({f['rule']}): 2016-2024 t {f['2016-2024']['t']:+.2f} on "
                   f"{f['2016-2024']['weeks']} weeks -> **{f['verdict']}**. Edge over the incumbent, %/yr: "
                   f"2006-2015 {res['edge_over_incumbent_cagr_pp']['2006-2015']:+.2f} (selection window), "
                   f"2016-2024 {res['edge_over_incumbent_cagr_pp']['2016-2024']:+.2f} (the one look)."]
    md += ["", "Leak audit: " + leak_line(res["leak_audit"])]
    ci = res.get("confidence")
    if ci:
        md += ["", "| window | metric | point | seed-only 95% | history-only 95% | nested 95% |",
               "|---|---|---|---|---|---|"]
        for w in ("2016-2024", "2006-2024", "2006-2015"):
            for nm, lab in (("excess_cagr_vs_sp500_pct", "excess CAGR vs sp500 %/yr"),
                            ("cagr_pct", "CAGR %/yr"), ("terminal_100", "terminal $100")):
                c = ci[w][nm]
                fmt = (lambda v: f"${v:,.0f}") if nm == "terminal_100" else (lambda v: f"{v:+.1f}")
                md.append(f"| {w} | {lab} | {fmt(c['point'])} | {fmt(c['seed_95'][0])} … {fmt(c['seed_95'][1])} | "
                          f"{fmt(c['history_95'][0])} … {fmt(c['history_95'][1])} | "
                          f"{fmt(c['nested_95'][0])} … {fmt(c['nested_95'][1])} |")
            md.append(f"| {w} | P(excess > 0), nested | {ci[w]['p_excess_positive_nested']:.2f} | | | |")
        md += ["", "Seed noise: the K copies resampled with replacement and re-simulated "
                   f"({ci['method']['seed_draws']} draws). History noise: a circular block bootstrap of the "
                   f"weeks, {ci['method']['block_weeks']}-week blocks, strategy and S&P on the same weeks "
                   f"({ci['method']['history_draws']} draws). Nested: both."]
    if res.get("charts"):
        md += [""] + [f"![{label} vs sp500]({Path(p).name})" for p in res["charts"]]
    return md


def run(walk: Path | None = None, incumbent: Path | None = None, store: str = STORE,
        k: int = K_COPIES, ci_draws: int = CI_SEED_DRAWS, charts: bool = True,
        spec_path: Path = SPEC_PATH, log=log) -> dict:
    from stocks_ml.leak_audit import audit_segments
    from stocks_ml.models.trials import record_trials
    from stocks_ml.train import context
    champion_walk = spec_walk(spec_path)
    walk = Path(walk) if walk else champion_walk
    is_champion = walk.resolve() == champion_walk.resolve()
    label = "champion" if is_champion else walk.name
    paths = segments(walk)
    rec = walk_records(paths)[0]
    proc = walk_settings(walk, store, k, log=log)
    st = settings_of(proc)
    if is_champion:
        spec_st = settings_of(json.loads(Path(spec_path).read_text())["procedure"])
        if spec_st != st:
            raise SystemExit(f"the spec's settings {spec_st} differ from the procedure's on its own walk "
                             f"{st}: run stocks-ml procedure")
    sel, ctx, _ = context(store)
    lo, hi = SPANS["2006-2024"]
    preds = load_preds(paths)
    check_complete(ctx, preds, k, lo, hi)
    log(f"eval: {label} = {walk} ({who(rec, st)}), {preds.week.nunique()} rank weeks at K={k}")
    pkg = weekly_returns(sel, ctx, preds, range(1, k + 1), st)
    spy = ctx.wret["SPY"]
    rows = {label: row_vs_spy(sel, pkg, spy)}
    res = {"label": label, "walk": str(walk), "record": rec, "who": who(rec, st), "k": k,
           "settings": st, "procedure": {kk: proc[kk] for kk in ("decision", "evidence", "preds", "decided_at")},
           "rank_weeks": int(preds.week.nunique()), "evaluated_at": str(pd.Timestamp.now().floor("s"))}
    if incumbent:
        inc_walk = Path(incumbent)
        inc_paths = segments(inc_walk)
        inc_proc = walk_settings(inc_walk, store, k, log=log)
        inc_st = settings_of(inc_proc)
        inc_preds = load_preds(inc_paths)
        check_complete(ctx, inc_preds, k, lo, hi)
        inc = weekly_returns(sel, ctx, inc_preds, range(1, k + 1), inc_st)
        # Each model is graded at its own procedure-decided settings only. A
        # row of the walk at the incumbent's settings was dropped 2026-09-14
        # (owner: a challenger at another model's settings is misleading).
        rows[f"incumbent ({inc_walk.name}, {settings_label(inc_st)})"] = row_vs_spy(sel, inc, spy)
        fals = falsification(pkg, inc)
        inc_row = rows[f"incumbent ({inc_walk.name}, {settings_label(inc_st)})"]
        res["incumbent"] = {"walk": str(inc_walk), "settings": inc_st,
                            "record": walk_records(inc_paths)[0]}
        res["falsification"] = fals
        res["edge_over_incumbent_cagr_pp"] = {w: round(rows[label][w]["cagr_pct"] - inc_row[w]["cagr_pct"], 2)
                                             for w in SPANS}
        log(f"eval: falsification t {fals['2016-2024']['t']:+.2f} on 2016-2024 -> {fals['verdict']}; "
            f"edge over the incumbent %/yr {res['edge_over_incumbent_cagr_pp']}")
    rows["sp500"] = {w: sel.metrics(spy.reindex(pkg.index), a, b) for w, (a, b) in SPANS.items()}
    res["table"] = rows
    log("\n".join(table_md(rows)))
    res["leak_audit"] = audit_segments(store, [str(p) for p in paths], ctx)
    log(f"eval: leak audit {res['leak_audit']['VERDICT']} (worst retention {res['leak_audit']['worst_retention']})")
    if ci_draws:
        res["confidence"] = confidence(sel, ctx, preds, st, k, seed_draws=ci_draws, log=log)
    df = pd.DataFrame({"walk": pkg, "sp500": spy.reindex(pkg.index)})
    df = df[(df.index >= lo) & (df.index < HOLDOUT_START)]
    df.index.name = "week"
    df.to_parquet(walk / "eval_weekly.parquet")
    if charts:
        footer = (f"walk {paths[0]} (sha256 {proc['preds']['sha256'][:12]}), settings decided by the procedure's "
                  f"code {proc['decided_at']}; the label and window were chosen on 2006-2015, 2016-2024 read once")
        res["charts"] = []
        for start in ("2016-01-01",):        # the out-of-sample years only: a chart that
            out = Path("reports") / f"{label}_vs_sp500_{start[:4]}_2024.png"   # includes 2006-2015 misleads
            chart(df, f"{label}: {res['who']}", footer, start, out, k)
            res["charts"].append(str(out))
            log(f"chart -> {out}")
    (walk / "eval.json").write_text(json.dumps(res, indent=1, default=str))
    md = report_md(res)
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / f"{label}_eval.md").write_text("\n".join(md) + "\n")
    record_trials([{"kind": "eval", "name": f"eval_{label}_{walk.name}_k{k}",
                    "pre_holdout_sharpe": rows[label]["2006-2024"]["sharpe"],
                    "notes": json.dumps({"walk": str(walk), "settings": st,
                                         "2016_2024": rows[label]["2016-2024"],
                                         "pre_holdout": rows[label]["2006-2024"],
                                         "falsification_t": (res.get("falsification") or {}).get("2016-2024", {}).get("t"),
                                         "verdict": (res.get("falsification") or {}).get("verdict"),
                                         "leak": res["leak_audit"]["VERDICT"]})}])
    log(f"eval -> {walk / 'eval.json'}, reports/{label}_eval.md")
    return res
