# lgbm Optuna tuning

TPE search, 100 trials, seed 0. Optimized on CV folds; the winner is adopted only if it also beats the random-search config on the untouched holdout.

| metric | value |
|---|---|
| best CV mean IC | 0.0268 |
| best config holdout IC | 0.0126 |
| incumbent (random-search) holdout IC | 0.0094 |
| trials | 100 |

**Verdict: adopted: holdout improved.**

Best config: `{'num_leaves': 48, 'learning_rate': 0.15595661430296504, 'n_estimators': 2000, 'min_child_samples': 109, 'reg_alpha': 0.0, 'reg_lambda': 7.823826459908271, 'subsample': 0.619918242051112, 'subsample_freq': 1, 'colsample_bytree': 0.8062460382271136, 'n_jobs': -1, 'random_state': 0, 'eval_fraction': 0.1, 'early_stopping_rounds': 20}`
