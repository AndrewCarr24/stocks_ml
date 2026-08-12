# xgb4w Optuna tuning

TPE search, 60 trials, seed 0. Selection uses only mean weekly rank IC on the pre-holdout purged walk-forward CV folds. The holdout is never read by tuning.

| metric | value |
|---|---|
| best CV mean IC | 0.0378 |
| trials | 60 |
| eligible | 60 |
| ineligible | 0 |
| failed | 0 |
| pruned | 0 |

**Verdict: selected by CV.**

Best config: `{'max_depth': 2, 'learning_rate': 0.013222538567527737, 'n_estimators': 5000, 'min_child_weight': 32, 'reg_alpha': 0.0, 'reg_lambda': 0.07247453179660597, 'subsample': 0.7413529494550588, 'colsample_bytree': 0.28534867805107383, 'n_jobs': -1, 'random_state': 0, 'eval_fraction': 0.1, 'early_stopping_rounds': 75, 'early_stop_purge_days': 42, 'early_stop_metric': 'weekly_spearman'}`

## Top eligible trials

| trial | mean IC | fold ICs | coverage | best iterations |
|---:|---:|---|---:|---|
| 23 | 0.0378 | 0.0304, 0.0455, 0.0369, 0.0386 | 485/485 | 0, 27, 525, 1 |
| 8 | 0.0364 | 0.0210, 0.0469, 0.0507, 0.0272 | 485/485 | 75, 0, 99, 13 |
| 50 | 0.0363 | 0.0263, 0.0449, 0.0478, 0.0265 | 485/485 | 18, 5, 234, 1 |
| 43 | 0.0314 | 0.0181, 0.0406, 0.0376, 0.0294 | 485/485 | 21, 30, 51, 1 |
| 59 | 0.0308 | 0.0195, 0.0410, 0.0389, 0.0241 | 485/485 | 334, 10, 188, 3 |
| 51 | 0.0308 | 0.0277, 0.0501, 0.0490, -0.0036 | 485/485 | 18, 3, 783, 7 |
| 49 | 0.0305 | 0.0263, 0.0408, 0.0282, 0.0266 | 485/485 | 14, 7, 44, 0 |
| 1 | 0.0301 | 0.0131, 0.0313, 0.0480, 0.0283 | 485/485 | 577, 3, 523, 27 |
| 42 | 0.0287 | 0.0033, 0.0478, 0.0365, 0.0275 | 485/485 | 34, 0, 423, 0 |
| 31 | 0.0284 | 0.0051, 0.0474, 0.0336, 0.0275 | 485/485 | 35, 0, 137, 0 |

Complete trial details and the best-so-far curve are in `optuna_xgb4w_trials.json`.
