# Timmermann (2006) — "Forecast Combinations" (Handbook of Economic Forecasting, ch. 4)

PDF: `papers/timmermann-2006-forecast-combinations.pdf` (UCSD working-paper version,
Nov 2004; note the PDF's text layer uses a shifted font encoding — it renders fine
but copy-paste is garbled)

## What the paper shows

The survey behind the "forecast combination puzzle": combinations of forecasts
reliably beat the ex-ante best individual model, and **simple equal weights
routinely beat estimated 'optimal' weights** because weight estimation error
swamps the gains, especially with short samples and unstable relative performance.
Best practices that survive the survey: equal or rank-based weights, trimming the
worst models before averaging, shrinkage of estimated weights toward equality,
and combining models with genuinely *different* information sets or functional
forms (diversification gain comes from low error correlation, not model count).

## What it implies for stocks_ml

- **It sharpens why our past prediction-ensembles nulled.** Project history #5
  records ensembles as a null. Timmermann predicts exactly this when the combined
  models share one information set and one functional family (tuned XGB, LGBM,
  CatBoost on identical rank features are near-clones; error correlation ≈ 1, no
  diversification gain). The paper's condition for combination gains — different
  information or different biases — points at XGB + **ElasticNet** (different
  functional form, nearly equal standalone IC 0.0198 vs 0.0190) and at
  cross-*horizon* combination (weekly model + `label_4w` monthly model), not at
  more boosted trees.
- **Equal weights, in rank space, decided before looking.** If a combination
  candidate enters the tournament, the paper says: average the two models'
  within-week prediction *ranks* with fixed 50/50 weights — do not fit weights on
  CV (that's another trial and, per the puzzle, likely worse). This is a zero-new-
  hyperparameter candidate, so it costs one entry in the trials ledger.
- **The staggered-refit ensemble is already Timmermann-endorsed.** Averaging
  models refit at staggered anchors (commit 796ae7d) is combination across
  *estimation windows* — his §5 instability argument is precisely why it makes
  predictions independent of the refit anchor. The paper supplies the citation and
  the refinement: trimming (drop the worst-performing anchor member each week)
  usually helps when one member degenerates — which is our documented failure mode
  (degenerate refits with ~no trees). The tie guard treats the symptom at the
  strategy layer; trimmed combination would treat it at the prediction layer.
- **Stability, not mean IC, is the honest yardstick for combinations.** The
  survey's gains show up mostly as variance reduction of forecast errors. For us:
  judge a combination candidate on fold-IC dispersion and holdout drawdown, and
  expect little mean-IC gain. That expectation should be written down *before*
  the run, or the deflated-Sharpe accounting quietly turns into cherry-picking.

## Concrete candidates

1. `RankBlend(xgb, enet)` tournament candidate: fixed 50/50 average of weekly
   prediction ranks; eligibility gate applies as usual.
2. Trimmed variant of the staggered-refit ensemble: per week, drop members whose
   recent-tail predictions are near-constant (reuses the tie-guard's degeneracy
   signal) before averaging.
