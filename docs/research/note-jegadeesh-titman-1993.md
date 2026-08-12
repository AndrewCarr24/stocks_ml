# Jegadeesh & Titman (1993) — "Returns to Buying Winners and Selling Losers"

PDF: `papers/jegadeesh-titman-1993-momentum.pdf` (JSTOR scan mirrored at U. Houston)

## What the paper shows

The original momentum paper: rank stocks on past J-month returns (J = 3,6,9,12),
hold K months (K = 3,6,9,12); all 16 strategies earn positive abnormal returns
(~1%/month for 6/6). Two structural details matter more than the headline:
(1) they **skip a week** between formation and holding to avoid contamination by
short-term reversal and bid-ask bounce; (2) the profits build over months 2–12
after formation and **partially reverse** thereafter (and momentum loses money in
January). Formation returns at 3–12 months predict; the most recent days/weeks
anti-predict.

## What it implies for stocks_ml

- **Skip-adjusted momentum, not just longer momentum.** The J-month/skip-a-week
  construction is the reason the standard factor is "12-1" not "12-0". Our
  `f_mom_4w` and `f_mom_12w` are *unskipped* — each includes the most recent week,
  which the paper (and our own `f_rev_resid_mkt_1w` reversal feature) says carries
  the opposite signal. The model can partially untangle this because the reversal
  feature is present, but rank-space mixing is lossy: skipped variants
  (`close(t−1w)/close(t−12w)`) hand the model the clean decomposition directly.
  This complements the Novy-Marx 2012 additions — together they define the
  momentum block: 4w (reversal-dominated), 12w-skip-1w, 26w, 52w-skip-4w,
  intermediate 52w→26w.
- **Horizon fit favors our label, mildly.** Momentum profits accrue from roughly
  month 2 onward — a forward-5-day label repeatedly re-samples that accrual, so
  weekly rebalancing is fine *provided the strategy's holding overlap is high*
  (top-8 membership should be sticky week to week for momentum-driven picks;
  turnover cost is the risk, and the simulator is already cost-aware).
- **The January effect is a calendar interaction we can already express.**
  Momentum's January losses (losers rebound) interact with our existing `f_month`
  calendar feature — a tree model can learn "damp momentum in January" only if
  both features are present. They are; no action beyond noting the mechanism when
  reading feature importances.
- **The post-12-month reversal is a warning for the monthly pipeline.** With
  `label_4w` and slower cadence, over-weighting very long lookbacks risks drifting
  into the reversal region (>12 months). Cap new momentum lookbacks at 52 weeks.
- **Sub-period stability is the paper's own robustness standard** — momentum
  worked in every 5-year sub-period they test. That maps onto our missing-data
  policy #6 (prefer fold-stable gains); a momentum-feature ablation should be
  positive in most of the 4 folds, not rescued by one.

## Concrete candidates

1. Fold skip-week construction into the new momentum features (see Novy-Marx 2012
   note): use `close(t−5d)` numerators for the 12w/26w lookbacks.
2. Watch weekly top-8 turnover in the backtest report when momentum features land;
   if turnover jumps without IC gain, the cost model is eating the accrual.
