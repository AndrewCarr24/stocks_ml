# Procedure card

Blueprint of the production procedure (generated 2026-09-07 by
`stocks-ml procedure-card` from models/champion_spec.json — edit the spec,
not this file). Rationale and history: AGENTS.md.

## Current champion

| Component | Spec |
|---|---|
| Model | simple-DT: depth-3 gbtree, untuned by design (config search measured as noise) |
| Prediction target | label_4w: stock's 4-week return minus that week's median member's (35-day purge) |
| Training | weekly refit on trailing 5 years; early stop on validation rank correlation |
| Features | the panel's f_ columns plus the generated bundle of 40 (features/bundle.py, selected on 2006-2015 alone, no asterisk): g_00 = `(r_close*r_sf_bvps)`, g_01 = `(f_sf_sales_to_price/r_sf_bvps)`, g_02 = `(r_close/r_sf_roe)`, g_03 = `max(f_dollar_vol,r_sf_bvps)`, g_04 = `(f_sf_sales_to_price-r_sf_bvps)`, g_05 = `(r_8k_2_02/r_sf_bvps)`, g_06 = `(f_sf_debt_ebitda+r_close)`, g_07 = `max(r_close,r_sf_bvps)`, g_08 = `(f_log_mktcap+r_8k_2_03)`, g_09 = `(f_sf_gross_prof/r_close)`, g_10 = `(r_sf_bvps*r_sf_grossmargin)`, g_11 = `(f_idio_vol_60d*r_close)`, g_12 = `(r_sf_bvps+r_sf_roe_yoy)`, g_13 = `(r_sf_bvps/r_sf_de)`, g_14 = `(r_8k_2_02/r_close)`, g_15 = `(f_log_mktcap+r_8k_1_01)`, g_16 = `(f_leverage-f_log_mktcap)`, g_17 = `(f_log_mktcap-r_sf_dps_yoy)`, g_18 = `(r_sf_bvps/r_sf_revenue)`, g_19 = `(r_close/r_sf_shareswa)`, g_20 = `(f_sf_sales_to_price/r_sf_eps)`, g_21 = `min(f_month,r_sf_bvps)`, g_22 = `(f_sfi_buyers_13w-r_sf_bvps)`, g_23 = `min(f_sf_fcf_yield,f_sf_gross_prof)`, g_24 = `(f_sf_roe/r_close)`, g_25 = `(f_sf_fcf_yield/r_sf_bvps)`, g_26 = `(f_log_mktcap-r_sf_fcf_yoy)`, g_27 = `(f_days_since_earnings_8k/r_sf_bvps)`, g_28 = `min(f_log_mktcap,r_sf_bvps)`, g_29 = `(r_sf_bvps-r_sf_divyield_yoy)`, g_30 = `(f_log_mktcap+r_sf_netinc_yoy)`, g_31 = `(f_book_to_market/r_sf_bvps)`, g_32 = `(f_log_mktcap+r_sf_epsusd)`, g_33 = `(f_sf_current_ratio*r_sf_bvps)`, g_34 = `(f_cf_to_price/r_8k_7_01)`, g_35 = `(f_sfi_buyers_13w+r_close)`, g_36 = `min(f_days_since_filing,r_sf_bvps)`, g_37 = `(f_log_mktcap-r_8k_2_02)`, g_38 = `(r_8k_9_01/r_sf_debt)`, g_39 = `min(f_days_since_earnings_8k,r_sf_bvps)` |
| Ensemble | K=16 copies (random_state + whole-week bootstrap), predictions averaged |
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

Mechanical cascade, run in this order on the selection window only; validated by the nested test (nested3_v1, 2026-09-04: select on every week of 2006-2015, grade 2016-2024; nested3_v2 re-ran it with the feature screen under rule v3.1, 2026-09-05; nested2_v1 was its sampled predecessor). Model config is NOT searched (simple-DT fixed; tuning measured as noise). Run programmatically: `stocks-ml select --sel-start A --sel-end B [--eval-start C --eval-end D] [--screen]` (stages cached & resumable under data/experiments/).

| step | menu | decided by |
|---|---|---|
| horizon | 1w vs 4w | cost-adjusted compounded return of the top-6 book (full population of the selection window) |
| training window | 1/2/3/4/5 years | top-6 edge vs random basket, paired on the weeks every window's population walk has (every week of the selection window since v3); all windows fully formed |
| feature screen (--screen) | the x_ candidates (features/candidates.py) as one bundle | probe: Newey-West t of 2 or more in size and the same IC sign in both halves of the window; the keepers ride a second population walk and enter iff the top-6 book's cost-adjusted compounded return is higher with them than without (v3.1, 2026-09-05; the paired HAC t is reported, not the rule). The champion carries the bundle nested3_v2 admitted (`features`, adopted 2026-09-05); a future screen's bundle enters the champion only on the owner's go |
| book size | top-3 / top-6 / top-10 | cost-adjusted compounded return |
| stagger | fixed on | mechanism (removes rotation-date luck); not searched |
| floor | none / half-gate / 80-20 / 70-30 / 60-40 trend-ballast | Sharpe ratio |
| stop-loss | off / -25% | Sharpe ratio (adopt only if higher) |
| sector cap | off / 2-of-book | Sharpe ratio (adopt only if higher) |

No layer reads a sample (v3, 2026-09-04), and no layer reads a rank week whose forward label ends after the window (`selection.label_end`: 35 days for 4w, 14 for 1w; 2026-09-05) — at a window ending at the holdout's edge those weeks would be graded on holdout prices. A bundle the screen admits is graded as a separate "+ engineered features" line with an asterisk (the candidate ideas were written from the whole 2006-2024 record), beside the clean line, never in place of it; the owner adopted nested3_v2's bundle into the champion on 2026-09-05 (features_note), so the champion's own record now carries the asterisk.

Metric convention: engine choices (horizon, window, book) by cost-adjusted compounded return; risk layers (floor, stop, cap) by Sharpe — owner-ratified 2026-09-01, 'for now'; earlier campaigns ranked by pre-tax earnings. Measured selection inflation of this
procedure: champion vs the nested honest-procedure pick on 2016-2024, as deployed: clean $590 (+23.2%/yr) vs v3's $521 (+21.2%/yr), +1.9%/yr on dollars, ~0 on Sharpe (0.95 vs 1.00); with the bundle the nested cascade's own pick out-earns the champion's settings ($1,379 vs $772), so the champion's book/ballast carry no inflation there — the asterisk on the bundle's ideas is the caveat that remains. History: +3.4%/yr vs v2's $463 as deployed, +3.1 on the close basis with the fixed join, +3.8 as first graded.

## Standing rules

- 2024-07-19+ is holdout: UNSPENT — single-use exam, owner-gated.
- Champion changes are owner-approved and recorded here + in the ledger;
  doubts become pre-registered falsification tests, never quiet overrides.
- Every evaluated config enters models/trials_ledger.json.
- All pre-holdout numbers carry design-iteration shine; treat accordingly.
