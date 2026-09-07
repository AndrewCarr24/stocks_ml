# Tails exam: the eight probe keepers in the champion's inputs

Pre-registered `tails_exam_v1` (2026-09-04). add ['f_earn_window', 'f_price_level', 'f_sf_cash_runway', 'f_8k_officer_26w', 'f_earn_react_mean4', 'f_8k_distress_26w', 'f_sf_accrual', 'f_overnight_12w_exam'] (the tails_features_probe_v1 keepers; ranked per week, neutral-filled) as one bundle to the champion inputs; ~115 sampled weeks 2006-01-01 -> 2024-06-14 (leverage_exam_v1's weeks: selection.sample_weeks seed 11, >=28d apart, thinned seed 7); selection.ensemble_preds K=4, champion params, 5y window, label_4w purge 35, fit fresh per week WITH and WITHOUT the bundle on identical weeks/seeds; statistics = paired diff (with - without) in top-6 4w return (PRIMARY: the champion holds six), top-10 4w return and hit10 (secondary, reported); PASS iff the primary's paired t > 2.0; raw returns, no costs. Sample pass -> within-bundle ablation, then population confirmation before adoption; fail -> the bundle is closed.

114 weeks 2006-01-06 -> 2024-05-10; SPY 4w mean +0.27%, member mean +0.21%. Returns are 4-week, open-to-open, raw.

| statistic | without | with | diff (with - without) | t | weeks with > without |
|---|---|---|---|---|---|
| top6 (primary) | +0.49% | +0.99% | +0.51% | +1.03 | 51% |
| top10 | +0.25% | +0.41% | +0.15% | +0.45 | 54% |
| hit10 | 0.500 | 0.525 | 0.025 | +1.62 | 40% |

Primary by era: 2006-12 diff +1.03%, t +0.90 (n 41); 2013-24 diff +0.21%, t +0.50 (n 73)

## Verdict

FAIL: the primary top-6 paired diff does not clear t > 2.0. The bundle adds nothing the trees can use at a 4-week horizon; it is closed. The champion is unchanged.
