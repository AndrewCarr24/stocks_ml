# Kelly, Malamud & Zhou — "The Virtue of Complexity in Return Prediction"

PDF: `papers/kelly-malamud-zhou-virtue-of-complexity.pdf` (NBER WP 30217)

## What the paper shows

Theory + experiments arguing that out-of-sample performance of return prediction
can *increase* in model complexity without bound, provided shrinkage is heavy:
take a small set of raw signals, expand them into thousands of random nonlinear
features (random Fourier features), fit ridge regression with P ≫ T, and OOS Sharpe
keeps improving past the interpolation boundary ("double descent" for markets).
The demonstration is mostly **market timing** with ~15 aggregate predictors and
tiny T (12 months of training data in the headline exhibit). Key mechanism: with
heavy ridge shrinkage, a misspecified overparameterized model behaves like a
kernel forecaster and its variance is controlled; complexity buys back bias.

## What it implies for stocks_ml

- **The transferable claim is about shrinkage, not magic.** Our setting (weekly
  cross-sectional *ranking* of ~500 names, T ≈ 100 weeks × 500 stocks per training
  window) is much better-conditioned than their T=12 timing exercise, so the
  dramatic wedge they show should not be expected here. What does carry over:
  when comparing model families, the regularization path matters more than the
  family. This is consistent with ENet nearly tying tuned XGB in our tournament.
- **A cheap, well-defined new candidate family.** A ridge on random Fourier
  features of the existing rank-normalized `f_*` columns (rank space is already
  bounded in (-1,1], ideal for RFF) is ~30 lines in `models/candidates.py` using
  sklearn's `RBFSampler` + `Ridge`. It is the paper's exact recipe transplanted to
  the cross-section, and unlike past model-zoo additions it introduces a genuinely
  different inductive bias (smooth nonlinear surface, massive shrinkage) rather
  than another boosted tree. Project history says new algorithms mostly null —
  this is the one paper that gives a principled reason to try exactly one more.
- **Do not relax the eligibility gate for it.** An overparameterized ridge can go
  near-constant in quiet weeks; iron rule #3 (rankable predictions every week,
  488-week coverage) already screens the failure mode. If RFF-ridge collapses in
  any fold, it loses — that's the gate working, not a reason to tweak z-scaling.
- **Complexity's virtue is measured after honest trial counting.** Each RFF width /
  ridge grid point is a trial; the pipeline league's Sharpe-deflation count must
  include the sweep (ties into the Bailey–López de Prado and Harvey–Liu–Zhu
  notes). The paper's own critics (the "too good to be true" replications) show
  the gains shrink under careful accounting — treat any big win as suspect per
  house rule (IC > 0.05 ⇒ suspect a bug).

## Concrete candidates

1. `RFFRidge` candidate: `RBFSampler(n_components≈4000)` → `Ridge(alpha≫1)`,
   tuned only on the pre-holdout folds, entered in the champion tournament.
2. If it survives, a second variant trained on `label_4w` for the monthly-cadence
   pipeline (their theory favors longer-horizon, smoother targets).
