# catboost Optuna tuning

TPE search, 60 trials, seed 0. Optimized on CV folds; the winner is adopted only if it also beats the random-search config on the untouched holdout.

| metric | value |
|---|---|
| best CV mean IC | 0.0192 |
| best config holdout IC | -0.0044 |
| incumbent (random-search) holdout IC | 0.0194 |
| trials | 60 |

**Verdict: rejected: holdout did not improve — random-search config retained.**

Best config: `{'depth': 9, 'learning_rate': 0.10016388797655294, 'l2_leaf_reg': 16.540962773755655, 'iterations': 2000, 'random_state': 0, 'eval_fraction': 0.1, 'early_stopping_rounds': 20}`
