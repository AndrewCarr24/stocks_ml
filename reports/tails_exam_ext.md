# Tails exam, extended: the eight probe keepers on the full 4-weekly sample

Pre-registered `tails_exam_v2` (2026-09-04). extension of tails_exam_v1: the ~120 selection.sample_weeks(2006-01-01 -> 2024-06-14, seed 11, >=28d apart) weeks not in v1 (disjoint, >=28d from every v1 week), identical design (bundle ['f_earn_window', 'f_price_level', 'f_sf_cash_runway', 'f_8k_officer_26w', 'f_earn_react_mean4', 'f_8k_distress_26w', 'f_sf_accrual', 'f_overnight_12w_exam'], selection.ensemble_preds K=4, champion params, 5y window, label_4w purge 35, WITH vs WITHOUT on identical weeks/seeds); statistics: REPLICATION = new weeks alone (diff, t, reported); PRIMARY = pooled v1 + new (~234 weeks), PASS iff top-6 paired t > 2.0; raw 4w open-to-open returns, no costs. The pooled t is not independent of v1 (extension decided after seeing v1); the replication is. Pass -> within-bundle ablation, then population confirmation before adoption; fail -> bundle closed for good.

## Replication (new weeks only): 118 weeks 2006-02-03 -> 2024-06-14

SPY 4w mean +1.15%, member mean +1.30%.

| statistic | without | with | diff (with - without) | t | weeks with > without |
|---|---|---|---|---|---|
| top6 (primary) | +2.11% | +3.01% | +0.90% | +1.59 | 52% |
| top10 | +1.91% | +2.43% | +0.52% | +1.30 | 53% |
| hit10 | 0.518 | 0.520 | 0.003 | +0.14 | 37% |

Top-6 by era: 2006-12 diff +0.72%, t +0.72 (n 46); 2013-24 diff +1.02%, t +1.51 (n 72). 95% CI on the top-6 diff -0.21% .. +2.01%.

## Primary (pooled v1 + new): 232 weeks 2006-01-06 -> 2024-06-14

SPY 4w mean +0.72%, member mean +0.77%.

| statistic | without | with | diff (with - without) | t | weeks with > without |
|---|---|---|---|---|---|
| top6 (primary) | +1.31% | +2.02% | +0.71% | +1.88 | 51% |
| top10 | +1.09% | +1.44% | +0.34% | +1.30 | 53% |
| hit10 | 0.509 | 0.522 | 0.013 | +1.13 | 39% |

Top-6 by era: 2006-12 diff +0.87%, t +1.15 (n 87); 2013-24 diff +0.61%, t +1.54 (n 145). 95% CI on the top-6 diff -0.03% .. +1.44%.

## Verdict

FAIL: pooled top-6 paired t +1.88 does not clear 2.0; the replication alone shows +0.90% (t +1.59). The bundle is closed. The champion is unchanged.
