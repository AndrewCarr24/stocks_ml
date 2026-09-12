# Improving the clean line — registration

Written 2026-09-11, before any number, on the owner's go ("Go", 2026-09-11,
to Stages A and B of the staged proposal). Motivation, in the owner's words:
"We had spent so much time optimizing with a contaminated feature set.
Explore avenues to improve model on this new 'uncontaminated' set ... feature
engineering with OpenFE, more hyperparm tuning (just the most important
settings and be mindful of compute time), investment strategy, and anything
else we may not be optimizing. Remember to always avoid leakage! 2016-2024
should give me a clean estimate of earnings, sharpe, etc."

The incumbent is the champion as deployed since fcc56a7 (2026-09-11): 64
base features, nominal basis, last-print labels, K=16 copies of
`selection.MODEL_PARAMS` on a 5-year window, top-6 / cap 2 / no stop /
70-30 trend floor. Its clean-line record (dl world, windows ending
2024-07-19 exclusive): 2006-2015 $229 (SR 0.46, DD 59%) | 2016-2024 $449
(+19.1%/yr, SR 0.80, DD 42%) | pre-holdout $1,028 (+13.4%/yr, SR 0.62,
DD 59%) vs SPY $197 / $316 / $621.

## Rules (binding on every stage)

1. **World.** Every walk runs on the delisting-honest research world
   (`data/sharadar_world2000_nominal_dl`: nominal basis, `last_print`
   labels, live's traded-within-7-days universe). Same world for challenger
   and incumbent; same weeks; same copies.
2. **Selection window.** Every decision — hyperparameters, window, label,
   feature admission, book, floor, stop, cap — is taken on 2006-01-01 ..
   2015-12-31 only, by the pre-registered metric's argmax. Nothing after
   2015-12-31 is read until Stage E.
3. **The one look.** 2016-01-01 .. 2024-07-18 is read exactly once, in
   Stage E, for the frozen package, at K=16. No re-selection follows from
   it. If the package fails the falsification test, the incumbent stands and
   the program ends; there is no second package.
4. **Holdout.** 2024-07-19 onward is not read by any stage. It is graded
   once, on the owner's separate go, never for re-selection.
5. **Leakage.** Every candidate walk that reaches Stage C or later gets the
   leak audit (`ops/leak_audit.py`) per segment, worst-case gate, selection
   window mandatory, before its numbers are believed. New feature families
   (Stage D) are point-in-time by construction (functions of the panel's own
   past rows) and are audited the same way. Labels are never features; the
   driver asserts no `label_*`, `fwd_ret*`, `x_*`, `g_*` column enters the
   model matrix (`feature_cols` admits `f_` columns only; the label variants
   add `label_*` columns).
6. **Path noise.** One terminal figure carries ~±30% path noise on a 6-name
   book. Every comparison is reported as the K=4 copy spread plus the paired
   weekly t against the incumbent on the same weeks; t stats are reported,
   not gated (owner's rule), except the falsification test in rule 3.
7. **Samples.** Stage B runs on every 4th week (a spaced sample) at K=4: it
   ranks candidates for Stage C only. No frozen decision reads a sample;
   every frozen decision (Stages C, D, E) reads every week of 2006-2015.
8. **Same bar.** Challenger and incumbent are compared on identical weeks,
   copies, world, costs and grade window. No arithmetic-vs-compounded
   mixing.
9. **Gos.** Each stage past B needs its own go with the previous stage's
   numbers in hand. Nothing here changes the deployed spec; adoption is a
   separate owner decision after Stage E.

## Metrics

- **Stage B sample metric** (ranking only): mean over sample weeks of the
  top-6 pick's 4-week forward return minus the universe mean that week
  (`slice_row`'s `top6 - rand_mean`, the same forward-return frame the
  cascade reads), per copy and averaged over copies 1-4; paired weekly t vs
  the incumbent's copies 1-4 on the same weeks.
- **Selection metric** (Stages C, D — frozen decisions): the cascade's book
  metric, cost-adjusted compounded %/yr of the top-6 book held 4 weeks
  (`selection.compounded_pct`), over every week of 2006-2015, K=4 copies
  1-4, reported with the four single-copy values and the paired weekly t
  of top-6 excess vs the incumbent's copies 1-4.
- **Strategy layers** (Stage A, and again on the frozen model in E): book
  by compounded %/yr; floor, stop, cap by Sharpe of the simulated weekly
  series over 2006-2015 (`ops.k16_program.cascade_at`, unchanged), plus the
  full book × floor Sharpe grid for information.
- **Stage E** (the one look): the 8-field table (pre/post-tax × pre-holdout/
  2016-2024 × terminal & CAGR / SR & DD) with the sp500 row, at K=16, on the
  dl world, windows ending 2024-07-19 exclusive; K=4 draws; 95% CIs on the
  excess CAGR as in reports/clean_line_confidence.md.

## Falsification test (pre-registered)

The frozen package's paired weekly excess over the incumbent (both at K=16,
both at their own cascade settings, both on the dl world, 2016-01-01 ..
2024-07-18) with t < −2 rejects the package. Any other outcome is reported
as measured; adoption is the owner's call with the 2016-2024 numbers and CIs
in hand, and the measured selection inflation from 2006-2015 stated
alongside.

## Stages

### A — strategy layers on the incumbent (free; no new fits)

Re-run the cascade on the incumbent's saved K=16 dl predictions over
2006-2015, and report the full book × floor Sharpe grid (books 3/6/10;
floors none/halfgate/80-20/70-30/60-40), stop and cap at the winning book
and floor. This records what the strategy layers would choose on the clean
line today, including the already-known 60-40 vs 70-30 question. It does
not change the deployed spec.

### B — model sweep at sample scale (~2.5 h; ranking only)

Weeks: every 4th rank week of 2006-2015 (~130 weeks). Copies: 1-4, the same
`WeekBootstrapEstimator` seeds the incumbent uses. Baseline: the incumbent's
own copies 1-4 from the saved dl preds on the same weeks (free). Fourteen
single-knob variants of `MODEL_PARAMS` / `fixed` / window / label, each
otherwise identical to the incumbent:

| id | knob | value (incumbent) |
|---|---|---|
| depth2 | max_depth | 2 (3) |
| depth4 | max_depth | 4 (3) |
| depth6 | max_depth | 6 (3) |
| lr01 | learning_rate | 0.01 (0.02) |
| lr05 | learning_rate | 0.05 (0.02) |
| mcw5 | min_child_weight | 5 (20) |
| mcw100 | min_child_weight | 100 (20) |
| col50 | colsample_bytree | 0.5 (0.8) |
| col30 | colsample_bytree | 0.3 (0.8) |
| fixed300 | n_estimators 300, no early stop | (1500 with early stop on the time tail) |
| win3 | training window | 3 years (5) |
| win8 | training window | 8 years (5) |
| lab_gauss | label | per-week rank-Gaussian of fwd_ret_4w (median-demeaned fwd_ret_4w) |
| lab_sector | label | fwd_ret_4w minus the week's sector median (median-demeaned) |

Output: one table, rows sorted by the sample metric, with the incumbent's
row, the per-copy values and the paired t. Stage B decides nothing; it
picks the ≤4 single knobs (plus 1-2 combinations of the top knobs) that go
to Stage C. The pick rule is mechanical: the four highest sample metrics.

### C — frozen model selection (needs a go; ~4 h)

The Stage B picks and combinations at K=4 on every week of 2006-2015; the
selection metric's argmax among {incumbent, candidates} is the frozen model.
Leak audit on each candidate walk. If the incumbent is the argmax, the
frozen model is the incumbent.

### D — feature families (needs a go; ~1.5 h)

With the frozen model: (i) sector-relative versions of the 64 base features
(feature minus the week's sector median; point-in-time by construction),
(ii) feature-rank momentum (change in a feature's cross-sectional rank over
4/13/26 weeks). Screened on 2006-2015 with a window-bounded select stage,
dedup at |corr| ≤ 0.7, admitted only if the selection metric at K=4 on every
week of 2006-2015 beats the frozen model without them (same bar). Reported
as a separate "+ engineered features" line with the asterisk.

### E — the package (needs a go; ~5 h)

The frozen model (+ admitted features) at K=16 on every week of 2006-2015,
the cascade re-decided, then the single 2016-2024 look, the leak audit, the
falsification test, CIs, the 8-field table with the sp500 row, and the
record in models/trials_ledger.json. The deployed spec changes only on the
owner's go after this table.

## Compute plan

Stage A minutes; B ~2.5 h (14 configs × ~130 weeks × 4 copies); C ~4 h;
D ~1.5 h; E ~5 h. All on the 14-core machine; walks checkpoint per config
and resume by week under a spec guard (a resume under another recipe
refuses).
