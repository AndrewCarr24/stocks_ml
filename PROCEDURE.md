# Procedure card

Blueprint of the production procedure (generated 2026-09-24 by
`stocks-ml procedure-card` from models/champion_spec.json — edit the spec,
not this file). Rationale and history: AGENTS.md.

## Current champion

| Component | Spec |
|---|---|
| Model | XGBoost: gradient-boosted trees of depth 3, learning rate 0.02, up to 1,500 rounds with early stopping on a time-ordered tail (weekly rank correlation), 16 bootstrap-seeded copies averaged; parameters fixed, never tuned (tuning measured as noise) |
| Prediction target | label_4w: the stock's 4-week return minus that week's median member's (35-day purge) |
| Training | weekly refit on trailing 8 years; early stop on validation rank correlation |
| Features | the panel's f_ columns with the survivor-keyed ones withheld (the 2026-09-21 correction: the S&P store's EDGAR facts, 8-K metadata and Form 4 filings had been pulled for today's members only, so a blank meant "gone by 2026" — 23 columns dropped: the EDGAR ratios, filing timing, PEAD, 8-K events, short ratio, SEC Form 4 insider flow, plus the two mixed-basis columns), Sharadar fundamentals and insiders (f_sf_*/f_sfi_*) kept, and in their place the corrected columns x_dollar_vol, x_short_dtc, x_evt_filed_5d, x_days_since_filing, x_pead, x_evt_8k_7d, x_evt_earnings_8k_7d, x_days_since_earnings_8k, x_mkt_dispersion: dollar volume and days-to-cover on one basis, the filing/PEAD and 8-K features recomputed from SEC tables that cover every name ever in the index; no screened bundle |
| Price basis / labels | level features on the nominal basis; a delisting's label grades to its final print (last_print) |
| Ensemble | K=16 copies (random_state + whole-week bootstrap), predictions averaged |
| Book | top-3, equal weight, 4 staggered sleeves rotating weekly, 4-week holds; weekly re-leveling; max 2/sector (blocked slots to next-ranked other-sector name); no stop; no volatility cut |
| Ballast | 60% book / 40% ballast: ballast in SPY, shifted to IEF one-third per breached trailing MA (30/40/52w) |
| Decided by | `stocks-ml procedure` on data/experiments/week_label_weekly/select/preds.parquet (K=16, 522 rank weeks of 2006-01-01 -> 2015-12-31, world data/sharadar_world2000_nominal_dl), 2026-09-24 19:41:17: book 3 / floor 60/40 / stop None / cap 2; model fields from the walk's own record: label_4w / 8-year window — tests hold the spec's model and strategy fields and the live job to this record |
| Honest expectation | no demonstrated edge over the S&P 500 out of sample: 2016-2024 $267, +12.1%/yr against the index's +14.3%/yr (excess -2.0%/yr, 95% nested CI -14.3..+13.7, P(excess>0) 0.42, paired weekly t -0.29); better than the champion it replaced on the same years (+9.6%/yr; +2.5%/yr, falsification t +0.13, not rejected) but still behind the index. 2006-2024 reads +3.4%/yr over the index (-6.8..+14.8, 0.73), carried by the selection window (+8.2%/yr on 2006-2015). A factor decomposition of the previous clean champion found the picks to be volatile, high-beta 12-month losers with no alpha after those exposures out of sample; expect this one to share that character until shown otherwise. Drawdowns of 46-50% are on the record, deeper than the index's 32% on the same years. Every pre-holdout number carries design-iteration shine: the label won a 35-way re-selection on 2006-2015 (a tie there) and a head-to-head on 2016-2019, a window this label family had been read on before |

## Cadences

| Activity | When | Human involvement |
|---|---|---|
| Refit + rotate one sleeve | weekly (Friday decision, Monday open trade, 5 bps) | none |
| Re-level weights / check stops / ballast state | weekly | none |
| Full re-selection (any layer) | **never on a calendar** — structural triggers only: new data source / feature family passes the gate; pre-registered kill-criterion breached; owner directive | pre-registered, owner-approved |

## Selection procedure (how each component is chosen)

Mechanical cascade, run in this order on the selection window only; validated by the nested test (nested3_v1, 2026-09-04: select on every week of 2006-2015, grade 2016-2024; nested3_v2 re-ran it with the feature screen under rule v3.1, 2026-09-05; nested2_v1 was its sampled predecessor). Model config is NOT searched (the XGBoost parameters fixed; tuning measured as noise). Run as code (the 2026-09-13 simplification, tag pre-simplify holds the old cascade tool): `stocks-ml train` walks a candidate model on 2006-2015; `stocks-ml challenge` (and its prototype `challenge-fast`) scores it by the MODEL SCORE — the mean over the top-3, top-6 and top-10 books of the cost-adjusted compounded %/yr on 2006-2015 (selection.decide_book, the numbers `stocks-ml backtest` prints; owner's rule 2026-09-14, replacing the single top-6 book) — and the argmax across candidate walks, incumbent included, names the model; `stocks-ml eval` reads 2016-2024 once with the falsification test and the leak audit; the strategy layers (book, floor, stop, cap) that reach this spec are written by `stocks-ml procedure --preds <K=16 walk of the selection window>` (stocks_ml.procedure), the only path into strategy.* and ballast.mix. Since 2026-09-12 the procedure also writes the model fields (horizon.label, training_window_years) from the walk's own record (<walk>/spec.json recipe), so the spec's model is the walk that was graded, never typed; the label and window themselves were chosen by the clean-improvement program's stage C (reports/clean_improvement_registration.md), the row below.

| step | menu | decided by |
|---|---|---|
| model: label x training window x fit | candidate recipes (label, training window, features, params) walked on 2006-2015 by `stocks-ml challenge`: a sample screen (every 4th week, K=4, ranks only) advancing the top three (two until 2026-09-24), the finalists every week at K=16 against the incumbent's two seed sets, a win needing a gap beyond twice the combined seed noise and a paired weekly t of 2, else a tie that keeps the incumbent, then the head-to-head on 2016-2019 (`--adjudicate`); `stocks-ml challenge-fast` prototypes only and never promotes. History: the clean program's stage C (2026-09-11) ranked 15 variants on a 131-week sample, then walked the four highest and their pairings on every week at K=4 | the model score: the mean over the top-3/6/10 books of the cost-adjusted compounded %/yr on the common ranked weeks (owner's rule 2026-09-14; the ls_w8 adoption of 2026-09-12 used the top-6 book alone: argmax ls_w8 +7.57 vs the incumbent +5.44, paired t +0.95 reported); the winner's strategy layers then decided by stocks-ml procedure on its K=16 walk |
| features | the panel's f_ columns; any extra panel columns are named in `features` and walked as a challenger model (asterisk when the ideas were written from the whole record) | the model metric above, on 2006-2015 alone, with > without at the cascade's own book; every candidate passes the leak audit per segment. None admitted as of 2026-09-11 (nominal screen 5.93 vs clean 8.14 %/yr) |
| book size | top-3 / top-6 / top-10 | cost-adjusted compounded return |
| stagger | fixed on | mechanism (removes rotation-date luck); not searched |
| floor | none / half-gate / 80-20 / 70-30 / 60-40 trend-ballast | Sharpe ratio |
| stop-loss | off / -25% | Sharpe ratio (adopt only if higher) |
| sector cap | off / 2-of-book | Sharpe ratio (adopt only if higher) |

No layer reads a sample (v3, 2026-09-04), and no layer reads a rank week whose forward label ends after the window (`selection.label_end`: 35 days for 4w, 14 for 1w; 2026-09-05) — at a window ending at the holdout's edge those weeks would be graded on holdout prices. Selection on 2006-2015 only; 2016-01 -> 2024-07-12 is the one out-of-sample look, read once per registered candidate; the holdout never feeds selection. Every candidate passes the leak audit (`stocks-ml eval`, leak_audit.py) on both segments before it is believed (the split leak, 2026-09-09). A bundle the screen admits is graded as a separate line beside the clean line, never in place of it; none is admitted as of 2026-09-11.

Metric convention: engine choices (horizon, window, book) by cost-adjusted compounded return; risk layers (floor, stop, cap) by Sharpe — owner-ratified 2026-09-01, 'for now'; earlier campaigns ranked by pre-tax earnings. Measured selection inflation of this
procedure: champion vs the nested honest-procedure pick on 2016-2024 (contaminated basis, 2026-09-04/05): clean $590 vs v3's $521, +1.9%/yr on dollars, ~0 on Sharpe. The deployed strategy layers are the cascade's own 2006-2015 argmax on the clean line since 2026-09-11 (strategy.settings_note); a nested re-measurement on the nominal world is a registered research item. Stage E (2026-09-12): the package chosen on 2006-2015 alone beat the incumbent on 2016-2024 by +6.4%/yr at the incumbent's settings ($555 vs $420) and +8.6%/yr at its own ($660); one read, as registered.

## Standing rules

- 2024-07-19+ is holdout: UNSPENT — single-use exam, owner-gated.
- Champion changes are owner-approved and recorded here + in the ledger;
  doubts become pre-registered falsification tests, never quiet overrides.
- Every evaluated config enters models/trials_ledger.json.
- All pre-holdout numbers carry design-iteration shine; treat accordingly.
