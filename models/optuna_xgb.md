# xgb Optuna tuning

TPE search, 100 trials, seed 0. Selection uses only mean weekly rank IC on the pre-holdout purged walk-forward CV folds. The holdout is never read by tuning.

| metric | value |
|---|---|
| best CV mean IC | 0.0198 |
| trials | 100 |
| eligible | 98 |
| ineligible | 2 |
| failed | 0 |
| pruned | 0 |

**Verdict: selected by CV.**

Best config: `{'max_depth': 2, 'learning_rate': 0.007005502056349284, 'n_estimators': 5000, 'min_child_weight': 20, 'reg_alpha': 0.0, 'reg_lambda': 0.23072779482344163, 'subsample': 0.820883186911213, 'colsample_bytree': 0.5000100879703847, 'n_jobs': -1, 'random_state': 0, 'eval_fraction': 0.1, 'early_stopping_rounds': 75, 'early_stop_purge_days': 10, 'early_stop_metric': 'weekly_spearman'}`

## Top eligible trials

| trial | mean IC | fold ICs | coverage | best iterations |
|---:|---:|---|---:|---|
| 44 | 0.0198 | 0.0061, 0.0206, 0.0180, 0.0343 | 488/488 | 17, 129, 6, 1 |
| 42 | 0.0194 | 0.0052, 0.0200, 0.0183, 0.0343 | 488/488 | 6, 129, 6, 1 |
| 81 | 0.0194 | 0.0063, 0.0185, 0.0184, 0.0343 | 488/488 | 6, 56, 5, 1 |
| 23 | 0.0190 | 0.0046, 0.0243, 0.0132, 0.0340 | 488/488 | 0, 7, 1, 1 |
| 71 | 0.0188 | 0.0091, 0.0142, 0.0176, 0.0343 | 488/488 | 25, 38, 5, 1 |
| 84 | 0.0188 | 0.0052, 0.0171, 0.0186, 0.0343 | 488/488 | 32, 38, 9, 1 |
| 95 | 0.0187 | 0.0041, 0.0230, 0.0131, 0.0343 | 488/488 | 12, 160, 1, 0 |
| 62 | 0.0185 | 0.0055, 0.0209, 0.0132, 0.0343 | 488/488 | 25, 127, 1, 1 |
| 88 | 0.0185 | 0.0053, 0.0211, 0.0131, 0.0343 | 488/488 | 70, 75, 1, 0 |
| 93 | 0.0184 | 0.0031, 0.0230, 0.0132, 0.0343 | 488/488 | 13, 88, 1, 1 |

Complete trial details and the best-so-far curve are in `optuna_xgb_trials.json`.
