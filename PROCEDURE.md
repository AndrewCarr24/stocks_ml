# Procedure card

Blueprint of the production procedure (generated 2026-09-01 by
`stocks-ml procedure-card` from models/champion_spec.json — edit the spec,
not this file). Rationale and history: AGENTS.md.

## Current champion

| Component | Spec |
|---|---|
| Model | simple-DT: depth-3 gbtree, untuned by design (config search measured as noise) |
| Prediction target | label_4w: stock's 4-week return minus that week's median member's (35-day purge) |
| Training | weekly refit on trailing 5 years; early stop on validation rank correlation |
| Ensemble | K=4 copies (random_state + whole-week bootstrap), predictions averaged |
| Book | top-6, equal weight, 4 staggered sleeves rotating weekly, 4-week holds; weekly re-leveling; max 2/sector (blocked slots to next-ranked other-sector name); no stop (audited: adds nothing over the ballast) |
| Ballast | 70% book / 30% ballast: ballast in SPY, shifted to IEF one-third per breached trailing MA (30/40/52w) |
| Honest expectation | edge is era-concentrated (strong 2013-2020, index-matching in whipsaw/megacap regimes); pre-holdout numbers carry design-iteration shine; deployment sizing should assume SPY-like outcomes in adverse regimes |

## Cadences

| Activity | When | Human involvement |
|---|---|---|
| Refit + rotate one sleeve | weekly (Friday decision, Monday open trade, 5 bps) | none |
| Re-level weights / check stops / ballast state | weekly | none |
| Full re-selection (any layer) | **never on a calendar** — structural triggers only: new data source / feature family passes the gate; pre-registered kill-criterion breached; owner directive | pre-registered, owner-approved |

## Selection procedure (how each component is chosen)

Mechanical cascade, run in this order on the selection window only; validated by the nested test (nested2_v1, 2026-09-01: select on 2006-2015, grade 2016-2024). Model config is NOT searched (simple-DT fixed; tuning measured as noise). Run programmatically: `stocks-ml select --sel-start A --sel-end B [--eval-start C --eval-end D]` (stages cached & resumable under data/experiments/).

| step | menu | decided by |
|---|---|---|
| horizon | 1w vs 4w | cost-adjusted compounded return of the top-6 book (full population of the selection window) |
| training window | 1/2/3/4/5 years | top-6 edge vs random basket, paired on identical sampled weeks; all windows fully formed |
| book size | top-3 / top-6 / top-10 | cost-adjusted compounded return |
| stagger | fixed on | mechanism (removes rotation-date luck); not searched |
| floor | none / half-gate / 80-20 / 70-30 / 60-40 trend-ballast | Sharpe ratio |
| stop-loss | off / -25% | Sharpe ratio (adopt only if higher) |
| sector cap | off / 2-of-book | Sharpe ratio (adopt only if higher) |

Metric convention: engine choices (horizon, window, book) by cost-adjusted compounded return; risk layers (floor, stop, cap) by Sharpe — owner-ratified 2026-09-01, 'for now'; earlier campaigns ranked by pre-tax earnings. Measured selection inflation of this
procedure: +3.8%/yr on dollars, ~0 on Sharpe (champion vs honest-procedure config, 2016-2024).

## Standing rules

- 2024-07-19+ is holdout: UNSPENT — single-use exam, owner-gated.
- Champion changes are owner-approved and recorded here + in the ledger;
  doubts become pre-registered falsification tests, never quiet overrides.
- Every evaluated config enters models/trials_ledger.json.
- All pre-holdout numbers carry design-iteration shine; treat accordingly.
