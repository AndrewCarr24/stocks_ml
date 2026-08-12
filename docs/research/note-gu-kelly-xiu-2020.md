# Gu, Kelly & Xiu (2020) — "Empirical Asset Pricing via Machine Learning"

PDF: `papers/gu-kelly-xiu-2020-empirical-asset-pricing-ml.pdf` (author copy, dachxiu.chicagobooth.edu)

## What the paper shows

Horse race of OLS, elastic net, PCR/PLS, random forests, GBTs, and neural nets on
~30k US stocks, 1957–2016, 94 characteristics + 8 macro series, predicting monthly
returns. Trees/NNs win; the gain over penalized linear models comes almost entirely
from **interactions** (characteristic × characteristic and characteristic × macro
state). Monthly OOS R² is tiny (~0.4%) yet economically large in long-short
portfolios. Predictability is concentrated in **small, illiquid stocks**; the most
important predictors are **recent price trends** (momentum at several horizons,
short-term reversal), then liquidity and volatility; fundamentals matter less at
short horizons.

## What it implies for stocks_ml

- **Our numbers are the right order of magnitude.** GKX's large-cap results are far
  weaker than their headline all-cap results. A weekly rank IC of ~0.02 on an
  S&P 500-only universe is consistent with their large-cap slice, not evidence we
  are leaving something big on the table. This calibrates expectations for every
  future ablation.
- **XGB ≈ ElasticNet is diagnostic, not a fluke.** GKX say tree gains come from
  interactions. Our champion XGB (IC 0.0198) barely beats ENet (0.0190) — matching
  project history #5 ("the signal is largely linear in rank space"). Two plausible
  causes worth separating: (a) the 2-year `cv_train_years` window is too short to
  learn stable interactions (GKX use 18+-year expanding windows); (b) our admitted
  macro set (T10Y2Y, FEDFUNDS only) removes the macro-interaction channel that
  drives much of their nonlinear gain. (a) is testable *within* the fixed fold
  design: train-window length is a model hyperparameter, not a fold boundary —
  a 4-year-window candidate in the tournament would answer it.
- **Their top feature block is price trends at multiple horizons.** Our price-trend
  family stops at `f_mom_12w` (~3 months). GKX's important set includes 6- and
  12-month momentum — a concrete gap (see the Novy-Marx 2012 and Jegadeesh–Titman
  notes; all three papers point at the same missing features).
- **Validation discipline matches ours.** They select hyperparameters on a
  disjoint validation sample, never the test years — same spirit as our purged CV +
  untouchable holdout. No change needed; useful as an external justification for
  iron rule #2.
- **What not to copy:** their NN gains rely on microcaps and a 60-year panel. Model
  zoo expansion (their NN3/NN4) is precisely the "new algorithms" direction that
  has repeatedly nulled for us; the paper's transferable lesson is the *feature*
  ranking, not the architecture ranking.

## Concrete candidates

1. Add 26-week and 52-week (skip most recent 4 weeks) momentum features.
2. Tournament candidate with a longer training window (same folds, same purge).
3. Low priority: an interaction-aware ablation (e.g. XGB depth sweep) only *after*
   longer windows, to test whether interactions were data-starved.
