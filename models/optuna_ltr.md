# ltr Optuna tuning

TPE search, 60 trials, seed 0. Selection uses only mean weekly rank IC on the pre-holdout purged walk-forward CV folds. The holdout is never read by tuning.

| metric | value |
|---|---|
| best CV mean IC | 0.0089 |
| trials | 60 |
| eligible | 59 |
| ineligible | 1 |
| failed | 0 |
| pruned | 0 |

**Verdict: selected by CV.**

Best config: `{'eval_metric': 'ndcg@32', 'max_depth': 2, 'learning_rate': 0.29865965537108285, 'n_estimators': 2000, 'min_child_weight': 143, 'reg_alpha': 0.0, 'reg_lambda': 0.20258278225720347, 'subsample': 0.8013264228108883, 'colsample_bytree': 0.9203077400673174, 'n_jobs': -1, 'random_state': 0, 'eval_fraction': 0.1, 'early_stopping_rounds': 75, 'early_stop_purge_days': 10}`

## Top eligible trials

| trial | mean IC | fold ICs | coverage | best iterations |
|---:|---:|---|---:|---|
| 52 | 0.0089 | -0.0076, 0.0224, 0.0120, 0.0089 | 488/488 | 123, 18, 27, 69 |
| 59 | 0.0086 | -0.0108, 0.0219, 0.0187, 0.0046 | 488/488 | 93, 84, 49, 31 |
| 31 | 0.0078 | -0.0154, 0.0195, 0.0161, 0.0113 | 488/488 | 3, 205, 111, 28 |
| 57 | 0.0076 | -0.0055, 0.0162, 0.0120, 0.0078 | 488/488 | 43, 21, 6, 56 |
| 56 | 0.0076 | -0.0069, 0.0189, 0.0122, 0.0061 | 488/488 | 1, 38, 2, 46 |
| 25 | 0.0074 | -0.0108, 0.0172, 0.0217, 0.0017 | 488/488 | 58, 49, 135, 17 |
| 27 | 0.0074 | -0.0105, 0.0167, 0.0150, 0.0085 | 488/488 | 36, 75, 3, 48 |
| 24 | 0.0073 | -0.0055, 0.0130, 0.0142, 0.0076 | 488/488 | 100, 68, 1, 64 |
| 51 | 0.0067 | -0.0129, 0.0186, 0.0096, 0.0115 | 488/488 | 14, 38, 11, 129 |
| 13 | 0.0066 | -0.0104, 0.0158, 0.0097, 0.0113 | 488/488 | 43, 52, 38, 164 |

Complete trial details and the best-so-far curve are in `optuna_ltr_trials.json`.
