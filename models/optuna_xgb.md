# xgb Optuna tuning

TPE search, 100 trials, seed 0. Optimized on CV folds; the winner is adopted only if it also beats the random-search config on the untouched holdout.

| metric | value |
|---|---|
| best CV mean IC | 0.0281 |
| best config holdout IC | 0.0225 |
| incumbent (random-search) holdout IC | 0.0196 |
| trials | 100 |

**Verdict: adopted: holdout improved.**

Best config: `{'max_depth': 3, 'learning_rate': 0.011328839727060015, 'n_estimators': 2000, 'min_child_weight': 183, 'reg_alpha': 0.4664193884984293, 'reg_lambda': 0.13896188460332184, 'subsample': 0.8303030062643391, 'colsample_bytree': 0.6279236871733346, 'n_jobs': -1, 'random_state': 0, 'eval_fraction': 0.1, 'early_stopping_rounds': 20}`
