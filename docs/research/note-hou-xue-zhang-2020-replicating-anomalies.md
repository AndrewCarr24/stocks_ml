# Hou, Xue & Zhang (2020) — "Replicating Anomalies"

PDF: `papers/hou-xue-zhang-2020-replicating-anomalies.pdf` (NBER WP 23394)

## What the paper shows

Replication of ~450 published cross-sectional anomalies under one protocol:
NYSE breakpoints and **value-weighted** returns, so microcaps (60% of names,
~3% of market cap) can no longer drive the results. Under |t| ≥ 1.96, ~65% of
anomalies fail; under the Harvey–Liu–Zhu t ≥ 2.78 hurdle, ~82% fail. What
survives the screen: **momentum** (including industry and intermediate-horizon
variants and earnings momentum / PEAD-family), **investment** (asset growth,
issuance), and **profitability** (ROE, gross profitability). What largely dies
once microcaps are gone: most **trading-frictions** anomalies — liquidity levels,
idiosyncratic volatility, turnover, many distress measures.

## What it implies for stocks_ml

This is the paper whose experimental design most resembles our universe: an
S&P 500-only panel is, in effect, a permanently microcap-screened, mega-cap-tilted
sample. Their survivor list is therefore the right *filter* on the
Green–Hand–Zhang shopping list (see that note).

- **The two lists agree on what we're missing.** GHZ survivors ∩ HXZ survivors ∩
  not-in-`panel.py` = long-horizon momentum, net share issuance, SUE/earnings
  momentum. Those are already recommendations #1 and #2 in
  `recommendations-2026-08.md` — HXZ upgrades them from "one paper says" to
  "the two harshest replication studies both say."
- **Our fundamentals block is HXZ-approved.** `f_gross_profitability` (verified
  GP/assets from XBRL `gross_profit`/`assets`), `f_roe`, `f_asset_growth`,
  `f_ocf_to_assets` all sit in their surviving investment/profitability
  categories. Keep, and prioritize their `label_4w` ablation.
- **Expect little from the liquidity family as *alpha*.** `f_amihud_4w/12w`,
  `f_dollar_vol`, `f_abn_volume`, `f_idio_vol_60d` belong to the category HXZ
  kill in value-weighted large-cap samples. That does *not* mean deleting them:
  in a nonlinear model they can earn their keep as conditioning/risk variables
  (GKX find volatility matters for *interaction* effects). It does mean: never
  spend an ablation slot trying to expand this family, and don't be surprised if
  a feature-importance audit shows them contributing little on their own.
- **Their protocol is a warning about our own reporting.** Equal-weighted top-8
  dollars (our headline) is the small-N cousin of the equal-weighting HXZ
  criticize: a couple of extreme names can carry a backtest. We already learned
  this the hard way (tie-guard postmortem, $139 vs $252 anchor spread). The
  IC-first, dollars-second evaluation order in the project is the correct HXZ-
  compatible stance; this paper is the citation for keeping it.
- **Effect-size calibration:** surviving anomalies in their value-weighted large-cap
  world run ~20–50 bps/month top-minus-bottom *decile* — with ~500 names, our
  top-8 sits inside the top decile, and weekly re-ranking captures a fraction of
  that. Consistent with champion IC ≈ 0.02 being near the honest ceiling.

## Concrete candidates

1. No new features from this paper itself — its role is to *veto*: before any
   future family ablation, check the anomaly's HXZ replication verdict first.
2. Add the HXZ verdict as a column in any future feature-shopping doc alongside
   GHZ's survivor flag.
