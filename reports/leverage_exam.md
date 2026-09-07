# Leverage exam: EBITDA/debt + FCF/cash in the champion's inputs

Pre-registered `leverage_exam_v1` (2026-09-04). add ['f_sf_ebitda_debt', 'f_sf_fcf_cash'] (EBITDA/debt signed, FCF/cash; ranked per week, neutral-filled) to the champion inputs; 115 sampled weeks 2006-01-01 -> 2024-06-14 (selection.sample_weeks seed 11, >=28d apart, thinned seed 7); selection.ensemble_preds K=4, champion params, 5y window, label_4w purge 35, fit fresh per week WITH and WITHOUT the pair on identical weeks/seeds; statistics = paired diff (with - without) in top-6 4w return (PRIMARY: the champion holds six), top-10 4w return and hit10 (secondary, reported); PASS iff the primary's paired t > 2.0; raw returns, no costs. Sample pass -> population confirmation before adoption; fail -> closed.

115 weeks 2006-01-06 -> 2024-05-10; SPY 4w mean +0.29%, member mean +0.22%. Returns are 4-week, open-to-open, raw.

| statistic | without | with | diff (with - without) | t | weeks with > without |
|---|---|---|---|---|---|
| top6 (primary) | +0.43% | +0.57% | +0.14% | +0.30 | 50% |
| top10 | +0.25% | +0.47% | +0.23% | +0.63 | 51% |
| hit10 | 0.502 | 0.507 | 0.005 | +0.42 | 37% |

## Verdict

FAIL: the primary top-6 paired diff does not clear t > 2.0. The pair adds nothing the trees can use at a 4-week horizon; the question is closed. The champion is unchanged.
