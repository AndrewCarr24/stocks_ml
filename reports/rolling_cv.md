# Two-year rolling cross-validation

Implemented 2026-07-21. The final two-year holdout remains untouched. Each fold
fits one frozen model on the preceding two calendar years, separated from its
validation block by the existing 10-calendar-day purge.

| fold | training boundary | validation dates | scored weeks |
|---|---|---|---:|
| 1 | 2013-02-24 → 2015-02-24 | 2015-03-06 → 2017-06-30 | 122 |
| 2 | 2015-06-27 → 2017-06-27 | 2017-07-07 → 2019-11-01 | 122 |
| 3 | 2017-10-29 → 2019-10-29 | 2019-11-08 → 2022-03-04 | 122 |
| 4 | 2020-03-01 → 2022-03-01 | 2022-03-11 → 2024-07-05 | 122 |

Boundary dates are calendar cutoffs; training rows are the weekly panel dates
falling inside each boundary. Validation totals 488 weeks. The calendar-year
holdout starts 2024-07-19; labels whose five-day exit crosses that boundary are
embargoed from tuning and selection.

## Residual-return lag ablation

> Historical result, superseded by the 2026-07-21 integrity audit. The table
> used the former 489-week boundary and feature matrix; production was retuned
> on the corrected 488-week calendar.

Frozen incumbent parameters, identical folds and scoring calendar:

| model | feature set | mean IC | fold ICs | coverage |
|---|---|---:|---|---:|
| XGBoost | no new lags | 0.014508 | 0.005036, 0.010710, 0.034155, 0.008207 | 489/489 |
| XGBoost | lag 4 only | 0.019711 | 0.003957, 0.010184, 0.035813, 0.029019 | 489/489 |
| ElasticNet | no new lags | 0.022357 | 0.016860, 0.024898, 0.028836, 0.018878 | 489/489 |
| ElasticNet | lag 4 only | 0.023130 | 0.017842, 0.026848, 0.028823, 0.019050 | 489/489 |

`f_resid_ret_lag4w` is admitted: it improves both models' mean IC, all four
ElasticNet folds, and two XGBoost folds materially. Lags 1–3 remain generated
for research and no-lookahead testing but are excluded from production model
matrices.

## XGBoost stopping protocol

- `n_estimators=5000` is a safety ceiling, not a tuned tree count.
- Early-stopping patience is fixed at 75 rounds.
- The most recent 10% of **complete weekly dates** in each fold's training
  window is the inner validation set.
- A separate 10-calendar-day inner purge separates that set from inner training.
- Early stopping minimizes negative mean weekly Spearman IC on that inner set,
  matching the outer selection metric rather than using regression RMSE.
- The outer fold validation set is never used for early stopping.
- Optuna selects configurations by complete-calendar outer-CV rank IC only.