# ltr Optuna tuning

TPE search, 60 trials, seed 0. Selection uses only mean weekly NDCG@8 on the pre-holdout purged walk-forward CV folds. The holdout is never read by tuning.

| metric | value |
|---|---|
| best CV mean IC | 0.5172 |
| trials | 60 |
| eligible | 58 |
| ineligible | 2 |
| failed | 0 |
| pruned | 0 |

**Verdict: selected by CV.**

Best config: `{'eval_metric': 'ndcg@8', 'max_depth': 8, 'learning_rate': 0.0958942629466341, 'n_estimators': 2000, 'min_child_weight': 55, 'reg_alpha': 0.00010071571751694611, 'reg_lambda': 1.4916581699927463, 'subsample': 0.5861960206217256, 'colsample_bytree': 0.9230828285985039, 'n_jobs': -1, 'random_state': 0, 'eval_fraction': 0.1, 'early_stopping_rounds': 75, 'early_stop_purge_days': 10}`

## Top eligible trials

| trial | mean IC | fold ICs | coverage | best iterations |
|---:|---:|---|---:|---|
| 23 | 0.5172 | 0.5194, 0.5423, 0.4961, 0.5112 | 488/488 | 60, 60, 2, 29 |
| 2 | 0.5168 | 0.5044, 0.5221, 0.5304, 0.5104 | 488/488 | 78, 48, 38, 127 |
| 12 | 0.5165 | 0.5037, 0.5489, 0.5038, 0.5094 | 488/488 | 0, 80, 3, 3 |
| 46 | 0.5165 | 0.5229, 0.5337, 0.4987, 0.5106 | 488/488 | 11, 86, 135, 254 |
| 5 | 0.5162 | 0.5324, 0.5435, 0.4891, 0.5000 | 488/488 | 95, 58, 4, 8 |
| 17 | 0.5162 | 0.5176, 0.5276, 0.5082, 0.5115 | 488/488 | 32, 146, 23, 166 |
| 41 | 0.5160 | 0.5130, 0.5299, 0.5019, 0.5191 | 488/488 | 66, 93, 6, 22 |
| 45 | 0.5159 | 0.5147, 0.5389, 0.5006, 0.5092 | 488/488 | 63, 48, 19, 42 |
| 20 | 0.5142 | 0.5225, 0.5199, 0.4991, 0.5152 | 488/488 | 201, 66, 2, 236 |
| 9 | 0.5141 | 0.5015, 0.5425, 0.5014, 0.5111 | 488/488 | 1, 93, 81, 21 |

Complete trial details and the best-so-far curve are in `optuna_ltr_trials.json`.
