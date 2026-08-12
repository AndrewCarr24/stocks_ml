# Short interest as a cross-sectional predictor — Boehmer, Jones & Zhang (2008) and the short-interest literature

PDF: `papers/boehmer-jones-zhang-2008-which-shorts-informed.pdf` ("Which Shorts Are
Informed?", JF 2008; author copy via Notre Dame conference site)

## What the papers show

BJZ 2008 use proprietary NYSE order-level **shorting flow** (not the biweekly
short-interest stock): heavily-shorted stocks underperform lightly-shorted ones by
~1.16%/month over the following 20 days, the information sits with institutional
non-program shorts, and — unusually for this literature — the result is
**value-weighted and survives in large caps**. Since their daily flow data is not
free, the operative companion results for our FINRA **short-interest levels** are:
Asquith–Pathak–Ritter 2005 (high short-interest-ratio underperformance is mostly
an equal-weighted/small-cap result and weak value-weighted), Boehmer–Huszár–Jordan
2010 (the *long* side — very low short interest predicts positive returns, and the
signal is informative among larger, liquid names), and Hong et al. 2015
(**days-to-cover** — short interest scaled by volume — beats the shares-outstanding
ratio as the cross-sectional variable, because it measures crowded-exit risk).

## What it implies for stocks_ml

- **What we already have is the literature's preferred pair.** `features/insiders.py::
  short_features` builds `f_short_ratio` (SI/shares outstanding, EDGAR-sourced
  shares) and `f_short_dtc` (SI/20d average volume = days-to-cover), both joined on
  FINRA **publication_date** (settlement + ~14 days), so the PIT lag is honest.
  The controlled A/B in project history #4 (+0.002 IC, consistent across folds)
  is, at our universe and horizon, in line with what the value-weighted
  literature would predict: real but small. Nobody should expect this family to
  double.
- **The missing transformation is the change, not the level.** BJZ's flow result
  says *new* shorting activity is where the information is; the free-data proxy
  for flow is the **change in short interest** between consecutive FINRA
  publications. Levels capture crowdedness (slow, works via DTC); changes capture
  information arrival (faster, closer to BJZ's mechanism). Candidates:
  `f_short_chg_8w` (change in short ratio over ~4 publication cycles) and an
  abnormal-level variant (`f_short_ratio` minus its own trailing 52-week mean),
  both computable from the parquet we already store.
- **Timing realism caps expectations for changes.** With settlement-plus-14-days
  publication, a "change" is ~3–5 weeks stale by the time we can trade it; BJZ's
  20-day horizon is mostly gone. That argues for testing changes on the weekly
  label but *expecting* the level/DTC features to remain the workhorses — the
  ablation is cheap either way since no new ingestion is needed.
- **Both tails matter in large caps.** BHJ 2010's long-side result fits our
  long-only `topk_spy` strategy better than the classic short-side story: for us
  the useful signal is "low/falling short interest supports a long candidate,"
  which rank-space features deliver automatically (the model sees the full
  cross-section, not a high-SI screen).
- **History window constraint stands.** FINRA history ≈ 2018+; the 2015–2018 folds
  see neutral-filled shorts (missing-data policy #3) — the Optuna gaming incident
  came from exactly this family's NaN structure, so any new short feature must
  re-verify the 488-week eligibility gate.

## Concrete candidates

1. `f_short_chg_8w` + `f_short_ratio_abn52w` from existing FINRA parquet;
   one family-level ablation, weekly label, t ≥ 3 rule.
2. No new data source; do not chase daily shorting-flow proxies (borrow fees,
   FTDs) — paid or unreliable free sources, against project data policy.
