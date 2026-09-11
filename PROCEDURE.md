# Procedure card

Blueprint of the production procedure (generated 2026-09-11 by
`stocks-ml procedure-card` from models/champion_spec.json — edit the spec,
not this file). Rationale and history: AGENTS.md.

## Current champion

| Component | Spec |
|---|---|
| Model | simple-DT: depth-3 gbtree, untuned by design (config search measured as noise) |
| Prediction target | label_4w: stock's 4-week return minus that week's median member's (35-day purge) |
| Training | weekly refit on trailing 5 years; early stop on validation rank correlation |
| Features | the panel's f_ columns (features/panel.py, Sharadar f_sf_*/f_sfi_*); no screened bundle (the split-leak bundle retired 2026-09-11, features/bundle.py) |
| Price basis / labels | level features on the nominal basis; a delisting's label grades to its final print (last_print) |
| Ensemble | K=16 copies (random_state + whole-week bootstrap), predictions averaged |
| Book | top-6, equal weight, 4 staggered sleeves rotating weekly, 4-week holds; weekly re-leveling; max 2/sector (blocked slots to next-ranked other-sector name); no stop (audited: adds nothing over the ballast) |
| Ballast | 70% book / 30% ballast: ballast in SPY, shifted to IEF one-third per breached trailing MA (30/40/52w) |
| Honest expectation | a few %/yr over SPY in expectation (2016-2024 excess CAGR +4.2%/yr, 95% nested CI -3.5..+16.3; 2006-2024 +2.8%/yr, -2.4..+10.6), a band that includes no edge, 40-60% drawdowns along the way; edge era-concentrated (2013-2020; index-matching in whipsaw/megacap regimes); pre-holdout numbers carry design-iteration shine; sizing should assume SPY-like outcomes in adverse regimes |

## Cadences

| Activity | When | Human involvement |
|---|---|---|
| Refit + rotate one sleeve | weekly (Friday decision, Monday open trade, 5 bps) | none |
| Re-level weights / check stops / ballast state | weekly | none |
| Full re-selection (any layer) | **never on a calendar** — structural triggers only: new data source / feature family passes the gate; pre-registered kill-criterion breached; owner directive | pre-registered, owner-approved |

## Selection procedure (how each component is chosen)

Mechanical cascade, run in this order on the selection window only; validated by the nested test (nested3_v1, 2026-09-04: select on every week of 2006-2015, grade 2016-2024; nested3_v2 re-ran it with the feature screen under rule v3.1, 2026-09-05; nested2_v1 was its sampled predecessor). Model config is NOT searched (simple-DT fixed; tuning measured as noise). Run programmatically: `stocks-ml select --sel-start A --sel-end B [--eval-start C --eval-end D] [--screen]` (stages cached & resumable under data/experiments/).

| step | menu | decided by |
|---|---|---|
| horizon | 1w vs 4w | cost-adjusted compounded return of the top-6 book (full population of the selection window) |
| training window | 1/2/3/4/5 years | top-6 edge vs random basket, paired on the weeks every window's population walk has (every week of the selection window since v3); all windows fully formed |
| feature screen (--screen) | a candidate bundle over the nominal raw inputs (ops/nominal_program.py screen: generated formulas, or the x_ candidates of features/candidates.py) | screened on 2006-2015 alone; admitted iff the top-book cost-adjusted compounded return is higher with it than without at the cascade's own book (the paired HAC t reported, not the rule); every candidate passes ops/leak_audit.py per segment. No bundle is admitted as of 2026-09-11 (nominal screen 5.93 vs clean 8.14 %/yr) |
| book size | top-3 / top-6 / top-10 | cost-adjusted compounded return |
| stagger | fixed on | mechanism (removes rotation-date luck); not searched |
| floor | none / half-gate / 80-20 / 70-30 / 60-40 trend-ballast | Sharpe ratio |
| stop-loss | off / -25% | Sharpe ratio (adopt only if higher) |
| sector cap | off / 2-of-book | Sharpe ratio (adopt only if higher) |

No layer reads a sample (v3, 2026-09-04), and no layer reads a rank week whose forward label ends after the window (`selection.label_end`: 35 days for 4w, 14 for 1w; 2026-09-05) — at a window ending at the holdout's edge those weeks would be graded on holdout prices. Selection on 2006-2015 only; 2016-01 -> 2024-07-12 is the one out-of-sample look, read once per registered candidate; the holdout never feeds selection. Every candidate passes ops/leak_audit.py on both segments before it is believed (the split leak, 2026-09-09). A bundle the screen admits is graded as a separate line beside the clean line, never in place of it; none is admitted as of 2026-09-11.

Metric convention: engine choices (horizon, window, book) by cost-adjusted compounded return; risk layers (floor, stop, cap) by Sharpe — owner-ratified 2026-09-01, 'for now'; earlier campaigns ranked by pre-tax earnings. Measured selection inflation of this
procedure: champion vs the nested honest-procedure pick on 2016-2024 (contaminated basis, 2026-09-04/05): clean $590 vs v3's $521, +1.9%/yr on dollars, ~0 on Sharpe. On the clean nominal line the cascade's 2006-2015 argmax differs from the standing settings on book and floor (strategy.settings_note); a nested re-measurement on the nominal world is a registered research item.

## Standing rules

- 2024-07-19+ is holdout: UNSPENT — single-use exam, owner-gated.
- Champion changes are owner-approved and recorded here + in the ledger;
  doubts become pre-registered falsification tests, never quiet overrides.
- Every evaluated config enters models/trials_ledger.json.
- All pre-holdout numbers carry design-iteration shine; treat accordingly.
