# Tails exam, third sample: 116 random in-between weeks

Pre-registered `tails_exam_v3` (2026-09-04). third sample for the tails bundle ['f_earn_window', 'f_price_level', 'f_sf_cash_runway', 'f_8k_officer_26w', 'f_earn_react_mean4', 'f_8k_distress_26w', 'f_sf_accrual', 'f_overnight_12w_exam']: 116 weeks drawn uniformly without replacement (seed 3) from the panel Fridays in [2006-01-01, 2024-06-14] not in tails_exam_v1/v2; identical design (selection.ensemble_preds K=4, champion params, 5y window, label_4w purge 35, WITH vs WITHOUT on identical weeks/seeds; raw 4w returns, no costs). Labels overlap, so every t is a calendar HAC t (Bartlett kernel, 28-day bandwidth; equals the plain t on the spaced 232). REPLICATION = new weeks alone (reported); PRIMARY = pooled v1+v2+v3 (~348 weeks): PASS iff pooled top-6 paired t > 2.0 AND the new weeks' top-6 diff > 0. Third look at one hypothesis, pooled with the earlier samples, by the owner's rule. PASS -> bundle admitted, adoption via the standing structural re-selection on the owner's go; FAIL -> closed for good.

## Replication (the 116 new weeks only): 114 weeks 2006-01-13 -> 2024-03-22

SPY 4w mean -0.17%, member mean -0.05%.

| statistic | without | with | diff (with - without) | HAC t | weeks with > without |
|---|---|---|---|---|---|
| top6 (primary) | +0.54% | +1.88% | +1.34% | +2.35 | 57% |
| top10 | +0.04% | +1.18% | +1.14% | +2.79 | 65% |
| hit10 | 0.495 | 0.504 | 0.010 | +0.52 | 40% |

Top-6 by era: 2006-12 diff +2.41%, t +2.38 (n 43); 2013-24 diff +0.68%, t +1.03 (n 71). Plain-t 95% CI on the top-6 diff +0.22% .. +2.45% (narrower than the HAC interval).

## Primary (pooled v1 + v2 + v3): 346 weeks 2006-01-06 -> 2024-06-14

SPY 4w mean +0.43%, member mean +0.50%.

| statistic | without | with | diff (with - without) | HAC t | weeks with > without |
|---|---|---|---|---|---|
| top6 (primary) | +1.06% | +1.97% | +0.91% | +2.75 | 53% |
| top10 | +0.75% | +1.35% | +0.61% | +2.63 | 57% |
| hit10 | 0.504 | 0.516 | 0.012 | +1.16 | 39% |

Top-6 by era: 2006-12 diff +1.38%, t +2.12 (n 130); 2013-24 diff +0.63%, t +1.78 (n 216). Plain-t 95% CI on the top-6 diff +0.30% .. +1.53% (narrower than the HAC interval).

## Verdict

PASS: pooled top-6 paired HAC t +2.75 > 2.0 and the new weeks' diff +1.34% > 0. The bundle is admitted; adoption follows the standing structural re-selection (full walk with the bundle, champion spec) on the owner's go.
