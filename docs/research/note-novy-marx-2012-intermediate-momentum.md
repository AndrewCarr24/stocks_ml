# Novy-Marx (2012) — "Is Momentum Really Momentum?"

PDF: `papers/novy-marx-2012-intermediate-momentum.pdf` (JFE 103, CiteSeerX copy)

## What the paper shows

Decomposing the classic 12-month momentum signal: returns from months **t−12 to
t−7** ("intermediate horizon") drive nearly all of momentum's predictive power;
returns from t−6 to t−2 ("recent horizon") add little and are more crash-prone.
Intermediate momentum is *stronger in large caps* than recent momentum, survives
among the largest quintile, and the "echo" (old winners keep winning even after
flat recent performance) holds across time and internationally in his sample.
(Later literature debates the echo's robustness — e.g. Goyal–Wahal find it weaker
elsewhere — so treat the decomposition, not the exact ranking, as the takeaway.)

## What it implies for stocks_ml

- **This is the clearest feature gap on the entire reading list.** Our price-trend
  family tops out at `f_mom_12w` — twelve *weeks*, i.e. inside Novy-Marx's "recent"
  window. The panel contains no 6-month momentum, no 12-month momentum, and no
  intermediate (t−12m…t−7m) return at all. Everything the momentum literature
  agrees is the strongest, most large-cap-robust horizon is absent from the model.
  Given project history #5 ("new information helps, new algorithms don't"), this
  is the cheapest plausible IC improvement available: a few `pct_change` lines in
  `panel.py::price_features`, rank-normalized like the others, full price history
  already on disk.
- **Recommended additions** (all point-in-time by construction from close prices):
  - `f_mom_26w` — 6-month momentum;
  - `f_mom_52w_skip4w` — close(t−4w)/close(t−52w)−1, the classic 12-1;
  - `f_mom_interm` — close(t−26w)/close(t−52w)−1, the paper's intermediate signal.
  Let the model weigh them; the paper's decomposition says `f_mom_interm` should
  dominate `f_mom_4w` in importance if our universe behaves like his large-cap slice.
- **Crash behavior matters for a top-8 account.** Recent-horizon momentum is the
  crash-prone component (2009-style reversals); intermediate momentum is milder.
  With `topk_spy` concentrating in 8 names, tilting the model's trend information
  toward the intermediate horizon is also a tail-risk argument, not just an IC one —
  relevant to the June-2026 semiconductor-crash week in the tie-guard postmortem.
- **Interaction with the 52-week-high feature.** `f_hi_52w` (price/52-week max) is
  correlated with intermediate momentum but not identical (George–Hwang vs
  Novy-Marx). Keep both; the ablation should report their joint and marginal
  contributions so we don't double-count one mechanism.
- **Sector-relative variants**: `sector_relative_momentum` currently transforms
  only `f_mom_4w`/`f_mom_12w`. If the new horizons earn their place, extend the
  same same-week sector-median demeaning to them — momentum in large caps is
  substantially industry momentum.

## Concrete candidates

1. Add the three lookbacks above to `panel.py`, ablate on identical folds per
   missing-data policy #4 (they are full-history, so no coverage caveats).
2. If adopted, extend sector-relative demeaning and rerun the champion tournament.
