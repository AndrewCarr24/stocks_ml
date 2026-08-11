# Point-in-time feature-family ablation

> **Superseded 2026-07-21:** these historical ablations used the former
> 489-week boundary and included non-vintage FRED and non-effective-dated sector
> inputs. Current production excludes those families and uses a 488-week,
> label-end-embargoed calendar. Do not compare these ICs with the current champion.

Run on 2026-07-21 using the same pre-holdout purged four-fold CV splits, rows,
and 489-week scoring calendar for every variant. The rolling two-year holdout
was not read. Hyperparameters were frozen at the existing CV-selected values.

## XGBoost

| features | mean IC | fold ICs | coverage |
|---|---:|---|---:|
| Existing baseline | 0.007244 | -0.012349, 0.010018, 0.024206, 0.007264 | 489/489 |
| + residual reversal | 0.007425 | -0.014199, 0.004354, 0.034638, 0.005085 | 489/489 |
| + Amihud | 0.004506 | -0.011376, 0.004538, 0.015521, 0.009469 | 489/489 |
| + SEC 8-K | 0.009098 | -0.011536, 0.000327, 0.038077, 0.009693 | 489/489 |
| + all three families | 0.003794 | -0.013539, 0.006425, 0.018992, 0.003439 | 489/489 |

Columns were kept in canonical panel order for every XGBoost variant. This is
necessary because seeded column subsampling addresses columns by position; an
earlier diagnostic run that appended each family at the end was discarded.

## ElasticNet

| features | mean IC | fold ICs | coverage |
|---|---:|---|---:|
| Existing baseline | 0.018558 | 0.004878, 0.015163, 0.033743, 0.020560 | 489/489 |
| + residual reversal | 0.018937 | 0.003254, 0.010849, 0.039530, 0.022243 | 489/489 |
| + Amihud | 0.018714 | 0.006046, 0.015002, 0.033998, 0.019912 | 489/489 |
| + SEC 8-K | 0.018808 | 0.005118, 0.015312, 0.034201, 0.020715 | 489/489 |
| + all three families | 0.019104 | 0.003960, 0.010888, 0.040196, 0.021494 | 489/489 |

## Admission decision

- **Admit SEC 8-K metadata.** Its ElasticNet gain is small (+0.000244 mean IC)
  but positive in every fold, and ElasticNet is the stronger model on the
  rebuilt panel under frozen parameters.
- **Reject residual reversal for production.** Its gain is concentrated in the
  third fold and it reduces IC in three of four XGBoost folds, as well as the
  first two ElasticNet folds.
- **Reject Amihud for production.** XGBoost improves in three folds, but its
  ElasticNet effect is mixed and the all-family interaction loses the gains.
  That is not stable enough across models and combinations for admission.
- Retain rejected columns in the panel for diagnostics and future research, but
  exclude them from model matrices through `REJECTED_MODEL_FEATURES`.

The rebuilt-panel XGBoost baseline no longer reproduces the prior 0.022283 IC,
while ElasticNet is nearly unchanged. This was not caused by SEC feature
coverage or a source-data refresh. As part of the feature work, return
calculations were changed from pandas' legacy implicit forward fill to explicit
`fill_method=None`. The strict calculation removes 27 stock-week rows whose
one-week momentum crossed a missing price and also changes other return-derived
feature cells around price gaps. The old result therefore used synthetic stale
prices across gaps and is not the benchmark to preserve. This calculation
correction should have been isolated and reported separately from feature
admission.

A subsequent audit found that `make_labels` originally centered returns across
every stored price series before the point-in-time membership merge. That let
departed tickers—including HOT's 122 stale zero-volume records after its
2016-09-22 exit—slightly affect targets. Labels are now recentered after the
membership merge. HOT changed the raw-universe median in 25 post-exit weeks;
the broader raw-universe/member-universe mismatch affected 1,105 weeks. The
tables above are the rerun results using corrected member-only labels.