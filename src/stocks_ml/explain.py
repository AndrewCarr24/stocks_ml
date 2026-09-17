"""`stocks-ml explain`: what the champion's fits lean on — Shapley values.

For each year of a span, one copy of the champion's recipe is refit at the
year's first rank week exactly as the walk fits it (same training window,
purge, seed, early stopping), and every member scored that week gets its
exact TreeSHAP contributions from the fitted booster (XGBoost's
pred_contribs: the Shapley values of a tree ensemble, no approximation, no
extra package). Per feature and year: the mean |SHAP| over the scored
members (how much the feature moves scores that week) and the rank
correlation between the feature's value and its contribution (the sign:
does a high value push the score up or down). The plot shows the top
features by mean |SHAP| across the years, the year-to-year spread, and the
sign; the table beside it carries every feature.

    stocks-ml explain                       # 2007-2024, copy 1, reports/champion_shap.{png,md}
    stocks-ml explain --years 2016-2024 --copies 1,2

Reads the selection window and the one-look years only; never the holdout.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SPEC_PATH = Path("models/champion_spec.json")
OUT_PNG = Path("reports/champion_shap.png")
OUT_MD = Path("reports/champion_shap.md")
OUT_TOP3_PNG = Path("reports/champion_shap_top3.png")
OUT_PICKS_PNG = Path("reports/champion_shap_picks.png")
TOP3_N = 3
PICKS_SAMPLE = 6          # random picks decomposed in the per-name figure
TOP = 25
WORDS = {"f_vol_12w": "12-week volatility", "f_vol_4w": "4-week volatility", "f_idio_vol_60d": "idiosyncratic volatility (60d)",
         "f_downside_dev": "downside deviation", "f_beta_60d": "beta (60d)", "f_vol_chg_12w": "volatility change (12w)",
         "f_beta_chg_12w": "beta change (12w)", "f_mom_1w": "1-week return", "f_mom_4w": "4-week return", "f_mom_12w": "12-week return",
         "f_mom_26w": "26-week return", "f_mom_52w": "52-week return", "f_mom_12w_skip1w": "12-week return, skip 1w",
         "f_mom_52w_skip4w": "52-week return, skip 4w", "f_mom_interm": "intermediate momentum", "f_mom_accel_4w": "momentum acceleration",
         "f_mom_sharpe_12w": "12-week return / volatility", "f_mom_consist_12w": "momentum consistency (12w)",
         "f_hi_52w": "distance from 52-week high", "f_lo_52w": "distance from 52-week low", "f_log_mktcap": "market cap",
         "f_dollar_vol": "dollar volume", "f_abn_volume": "abnormal volume", "f_amihud_4w": "illiquidity (4w)", "f_amihud_12w": "illiquidity (12w)",
         "f_overnight_4w": "overnight return share (4w)", "f_intraday_4w": "intraday return share (4w)",
         "f_overnight_12w": "overnight return share (12w)", "f_intraday_12w": "intraday return share (12w)",
         "f_rev_resid_mkt_1w": "1-week residual reversal", "f_sf_earnings_yield": "earnings yield", "f_sf_book_to_market": "book / market",
         "f_sf_fcf_yield": "free-cash-flow yield", "f_sf_sales_to_price": "sales / price", "f_sf_debt_ebitda": "debt / EBITDA",
         "f_sf_current_ratio": "current ratio", "f_sf_de": "debt / equity", "f_sf_roe": "return on equity", "f_sf_gross_prof": "gross profitability",
         "f_sf_ebitda_margin": "EBITDA margin", "f_sf_netinc_yoy": "net income growth", "f_sf_revenue_yoy": "revenue growth",
         "f_sf_issuance": "share issuance", "f_sf_neg_ebitda": "negative EBITDA (flag)", "f_sfi_net_13w": "insider net buying (13w)",
         "f_sfi_buyers_13w": "insider buyers (13w)", "f_insider_net_13w": "insider net buying, Form 4 (13w)",
         "f_insider_buyers_13w": "insider buyers, Form 4 (13w)", "f_evt_insider_buy_2w": "insider buy in the last 2w (flag)",
         "f_short_ratio": "short interest ratio", "f_short_dtc": "short interest, days to cover", "f_short_chg_8w": "short interest change (8w)",
         "f_evt_8k_7d": "8-K in the last week (flag)", "f_evt_earnings_8k_7d": "earnings 8-K in the last week (flag)",
         "f_days_since_earnings_8k": "days since earnings", "f_evt_filed_5d": "filing in the last 5 days (flag)",
         "f_days_since_filing": "days since filing", "f_pead": "post-earnings drift", "f_sue": "earnings surprise", "f_nincr": "earnings increases streak",
         "f_net_issuance": "net issuance", "f_mkt_mom_4w": "market 4-week return", "f_mkt_mom_26w": "market 26-week return",
         "f_mkt_vol_4w": "market volatility (4w)", "f_mkt_dispersion": "cross-sectional dispersion", "f_macro_T10Y2Y": "yield curve (10y-2y)",
         "f_macro_FEDFUNDS": "fed funds rate", "f_macro_T10Y2Y_chg": "yield curve change", "f_macro_FEDFUNDS_chg": "fed funds change",
         "f_mom_4w_sect": "4-week return vs sector", "f_mom_12w_sect": "12-week return vs sector", "f_month": "month", "f_woq": "week of quarter",
         "f_earnings_yield": "earnings yield (EDGAR)", "f_book_to_market": "book / market (EDGAR)", "f_sales_to_price": "sales / price (EDGAR)",
         "f_cf_to_price": "cash flow / price (EDGAR)", "f_roe": "ROE (EDGAR)", "f_gross_profitability": "gross profitability (EDGAR)",
         "f_ocf_to_assets": "operating cash flow / assets", "f_asset_growth": "asset growth", "f_leverage": "leverage"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_at(sel, ctx, t, copy: int, label: str, train_years: int, features=(), params=None):
    """The champion's copy fitted at rank week t exactly as train.copy_preds
    fits it; returns (fitted WeekBootstrapEstimator, the scored rows, fcols)."""
    from sklearn.base import clone

    from stocks_ml.features.panel import feature_cols
    from stocks_ml.models.replication import WeekBootstrapEstimator
    from stocks_ml.models.walk import MIN_TRAIN_ROWS, MIN_TRAIN_WEEKS
    from stocks_ml.models.xgb import TimeTailEarlyStopXGB, dated_features
    purge = sel.label_purge(label, "4w")
    est = WeekBootstrapEstimator(TimeTailEarlyStopXGB(**{**sel.MODEL_PARAMS, **(params or {})}, **sel.fixed(purge)),
                                 bootstrap_seed=copy)
    pan = ctx.pan
    fcols = feature_cols(pan) + [c for c in features if c not in feature_cols(pan)]
    labeled = pan[pan[label].notna()]
    train_end = t - pd.Timedelta(days=purge)
    train = labeled[labeled["date"].between(train_end - pd.DateOffset(years=train_years), train_end)]
    if len(train) < MIN_TRAIN_ROWS or train["date"].nunique() < MIN_TRAIN_WEEKS:
        raise RuntimeError(f"too little training data at {t.date()}")
    model = clone(est).fit(dated_features(train, fcols), train[label])
    rows = pan[pan["date"] == t]
    return model, rows, fcols


def shap_at(model, rows: pd.DataFrame, fcols: list[str]) -> pd.DataFrame:
    """Exact TreeSHAP contributions (XGBoost pred_contribs) for the scored
    rows: one column per feature, the last is the bias; indexed by ticker."""
    import xgboost as xgb
    booster = model.model_.get_booster()
    X = rows[fcols]
    contrib = booster.predict(xgb.DMatrix(X, feature_names=list(fcols)), pred_contribs=True)
    return pd.DataFrame(contrib[:, :-1], columns=fcols, index=rows["ticker"].values)


def year_table(shap: pd.DataFrame, rows: pd.DataFrame, fcols: list[str]) -> pd.DataFrame:
    """Per feature: mean |SHAP| over the scored members and the rank
    correlation between the feature's value and its contribution (sign)."""
    import warnings

    from scipy.stats import spearmanr
    X = rows.set_index("ticker")[fcols].reindex(shap.index)
    out = {"mean_abs_shap": shap.abs().mean()}
    sign = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # a constant contribution column has no sign
        for c in fcols:
            x, s = X[c].values, shap[c].values
            ok = np.isfinite(x) & np.isfinite(s)
            sign[c] = (spearmanr(x[ok], s[ok]).correlation
                       if ok.sum() > 20 and len(np.unique(x[ok])) > 5 and len(np.unique(s[ok])) > 1 else np.nan)
    out["sign_corr"] = pd.Series(sign)
    return pd.DataFrame(out)


def explain(store: str, years, copies=(1,), spec_path: Path = SPEC_PATH, log=log) -> dict:
    """Mean |SHAP| and sign per feature for each year's first rank week, for
    the spec's recipe; returns {"by_year": DataFrame (feature x year),
    "sign": DataFrame, "recipe": dict, "weeks": [...]}."""
    from stocks_ml.selection import HOLDOUT_START
    from stocks_ml.train import context
    spec = json.loads(Path(spec_path).read_text())
    recipe = {"label": spec["horizon"]["label"], "train_years": int(spec["training_window_years"]),
              "features": list(spec.get("features") or []),
              "params": {k: v for k, v in spec["model"]["params"].items()}}
    sel, ctx, _ = context(store)
    from stocks_ml.selection import MODEL_PARAMS
    overrides = {k: v for k, v in recipe["params"].items() if str(v) != str(MODEL_PARAMS.get(k))}
    by_year, sign, weeks, picks = {}, {}, [], []
    for y in years:
        cands = [t for t in ctx.weeks if t >= pd.Timestamp(f"{y}-01-01") and t < HOLDOUT_START]
        if not cands:
            continue
        t = cands[0]
        tabs, shaps, scores = [], [], []
        for c in copies:
            model, rows, fcols = fit_at(sel, ctx, t, c, recipe["label"], recipe["train_years"],
                                        recipe["features"], overrides)
            sh = shap_at(model, rows, fcols)
            tabs.append(year_table(sh, rows, fcols)); shaps.append(sh)
            scores.append(pd.Series(model.predict(rows[fcols]), index=rows["ticker"].values))
        tab = pd.concat(tabs).groupby(level=0).mean()
        by_year[y] = tab["mean_abs_shap"]; sign[y] = tab["sign_corr"]; weeks.append(str(t.date()))
        # the top-TOP3_N by the copies' mean score (members only, as the live job ranks): their contributions
        score = pd.concat(scores, axis=1).mean(axis=1)
        members = [x for x in ctx.members.get(t, []) if x in score.index]
        top3 = score.loc[members].sort_values(ascending=False).head(TOP3_N)
        sh_mean = pd.concat(shaps).groupby(level=0).mean()
        for rank, (tk, sc) in enumerate(top3.items(), 1):
            row = sh_mean.loc[tk].copy(); row.name = tk
            picks.append({"year": y, "week": str(t.date()), "rank": rank, "ticker": tk, "score": float(sc), "shap": row})
        top = tab["mean_abs_shap"].sort_values(ascending=False).head(5)
        log(f"{y} ({t.date()}, {len(rows)} members, {len(copies)} copies): top-3 {', '.join(top3.index)}; " +
            ", ".join(f"{WORDS.get(k, k)} {v:.4f}" for k, v in top.items()))
    return {"by_year": pd.DataFrame(by_year), "sign": pd.DataFrame(sign), "recipe": recipe, "weeks": weeks,
            "picks": picks}


def render(res: dict, png: Path = OUT_PNG, md: Path = OUT_MD, top: int = TOP) -> pd.DataFrame:
    """The plot and the table. Returns the summary frame (feature rows)."""
    by, sg = res["by_year"], res["sign"]
    summ = pd.DataFrame({"mean_abs_shap": by.mean(axis=1), "sd_across_years": by.std(axis=1),
                         "sign_corr": sg.mean(axis=1), "share_of_total": by.mean(axis=1) / by.mean(axis=1).sum()})
    summ = summ.sort_values("mean_abs_shap", ascending=False)
    summ["name"] = [WORDS.get(c, c) for c in summ.index]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt is not None:
        head = summ.head(top).iloc[::-1]
        colors = ["#2a6f97" if s >= 0 else "#c9553d" for s in head["sign_corr"]]
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(head) + 1.6))
        ax.barh(head["name"], head["mean_abs_shap"], xerr=head["sd_across_years"], color=colors,
                error_kw={"ecolor": "#666", "elinewidth": 0.8, "capsize": 2})
        ax.set_xlabel("mean |Shapley value| over the members scored that week, averaged over years (score units)")
        r = res["recipe"]
        ax.set_title(f"What the champion leans on — {r['label']}, {r['train_years']}-year window; "
                     f"{len(by.columns)} yearly fits {min(by.columns)}–{max(by.columns)}, {len(res['weeks'])} weeks\n"
                     f"blue: a higher value raises the score; red: lowers it; bars: year-to-year spread", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=150)
        plt.close(fig)
    lines = [f"# Champion feature importance (Shapley values)", "",
             f"Recipe {res['recipe']}; one yearly fit at each year's first rank week ({res['weeks'][0]} → {res['weeks'][-1]}), "
             f"exact TreeSHAP on every member scored that week. mean |SHAP| in score units; share = of the total across features; "
             f"sign = rank correlation between the feature's value and its contribution (positive: higher raises the score).", "",
             "| rank | feature | mean abs SHAP | share | sd across years | sign |", "|---|---|---|---|---|---|"]
    for i, (c, r) in enumerate(summ.iterrows(), 1):
        lines.append(f"| {i} | {r['name']} (`{c}`) | {r['mean_abs_shap']:.4f} | {r['share_of_total']:.1%} | {r['sd_across_years']:.4f} | {r['sign_corr']:+.2f} |")
    md.write_text("\n".join(lines) + "\n")
    return summ


def render_top3(res: dict, png: Path = OUT_TOP3_PNG, md: Path = OUT_MD, top: int = 20) -> pd.DataFrame:
    """What pushed the model's top-3 into the top-3: the mean signed
    contribution per feature over every pick (a contribution is the feature's
    push on that name's score relative to the week's average member), the
    plot for the largest, and per week the names with their three biggest
    drivers appended to the table."""
    picks = res.get("picks") or []
    if not picks:
        return pd.DataFrame()
    S = pd.DataFrame([p["shap"] for p in picks])
    mean_signed = S.mean().sort_values(key=lambda v: v.abs(), ascending=False)
    share_pos = (S > 0).mean().reindex(mean_signed.index)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    if plt is not None:
        head = mean_signed.head(top).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(head) + 1.8))
        ax.barh([WORDS.get(c, c) for c in head.index], head.values,
                color=["#2a6f97" if v >= 0 else "#c9553d" for v in head.values])
        for i, (c, v) in enumerate(head.items()):
            ax.text(v + (0.0004 if v >= 0 else -0.0004), i, f"{share_pos[c]:.0%} of picks", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7, color="#444")
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlabel("mean Shapley contribution to the top-3 picks' scores (score units; label: share of picks pushed up by it)")
        r = res["recipe"]
        ax.set_title(f"What puts a name in the champion's top-3 — {len(picks)} picks, the first rank week of each year "
                     f"{res['weeks'][0][:4]}–{res['weeks'][-1][:4]}\n{r['label']}, {r['train_years']}-year window; "
                     f"a contribution is the feature's push relative to that week's average member", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        plt.close(fig)
    lines = ["", "## The top-3 picks and what drove them", "",
             "Each week's three highest-scored members; the three largest contributions to each score "
             "(score units; positive pushed the name up).", "",
             "| week | rank | ticker | score | biggest drivers |", "|---|---|---|---|---|"]
    for p in picks:
        sh = p["shap"].sort_values(key=lambda v: v.abs(), ascending=False).head(3)
        lines.append(f"| {p['week']} | {p['rank']} | {p['ticker']} | {p['score']:+.3f} | "
                     + "; ".join(f"{WORDS.get(k, k)} {v:+.3f}" for k, v in sh.items()) + " |")
    lines += ["", "Mean signed contribution over all picks, largest first: "
              + ", ".join(f"{WORDS.get(k, k)} {v:+.4f}" for k, v in mean_signed.head(12).items())]
    with md.open("a") as f:
        f.write("\n".join(lines) + "\n")
    return pd.DataFrame({"mean_signed_shap": mean_signed, "share_of_picks_pushed_up": share_pos})


def render_picks(res: dict, png: Path = OUT_PICKS_PNG, n: int = PICKS_SAMPLE, seed: int = 0, top: int = 12) -> list:
    """Shapley decompositions of a random sample of top-3 picks: one panel
    per name with its largest contributions (signed, largest first) and
    the rest summed — the additive story of that one score. Returns the
    sampled picks."""
    picks = res.get("picks") or []
    if not picks:
        return []
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(picks), size=min(n, len(picks)), replace=False))
    sample = [picks[i] for i in idx]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return sample
    cols = 2
    rows_n = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(13, 3.6 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, sample):
        sh = p["shap"].sort_values(key=lambda v: v.abs(), ascending=False)
        head, rest = sh.head(top), sh.iloc[top:].sum()
        parts = pd.concat([head, pd.Series({f"{len(sh) - top} other features": rest})]).iloc[::-1]
        ax.barh([WORDS.get(c, c) for c in parts.index], parts.values,
                color=["#2a6f97" if v >= 0 else "#c9553d" for v in parts.values])
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_title(f"{p['ticker']}  ·  {p['week']}  ·  rank {p['rank']}  ·  score {p['score']:+.3f}\n"
                     f"features sum to {sh.sum():+.3f} above the week's average member", fontsize=9)
        ax.tick_params(axis="y", labelsize=8); ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[len(sample):]:
        ax.set_visible(False)
    fig.suptitle(f"Shapley decompositions of {len(sample)} random top-3 picks (seed {seed}): each bar is that feature's "
                 f"push on the name's score, in score units", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return sample


def run(store: str, years, copies=(1,), log=log) -> pd.DataFrame:
    res = explain(store, years, copies, log=log)
    summ = render(res)
    t3 = render_top3(res)
    if len(t3):
        log(f"top-3 drivers -> {OUT_TOP3_PNG}: " + ", ".join(f"{WORDS.get(k, k)} {v:+.4f}" for k, v in t3['mean_signed_shap'].head(6).items()))
        sample = render_picks(res)
        log(f"per-pick decompositions -> {OUT_PICKS_PNG}: " + ", ".join(f"{p['ticker']} {p['week']}" for p in sample))
    log(f"explain -> {OUT_PNG}, {OUT_MD}; top: " + ", ".join(f"{r['name']} {r['share_of_total']:.0%}" for _, r in summ.head(8).iterrows()))
    return summ
