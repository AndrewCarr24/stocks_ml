# catboost hyperparameter tuning

Selection is by mean weekly rank IC on the pre-holdout purged walk-forward CV folds (plain CV selection) — the untouched holdout is never used for tuning and remains the honest test.
A config is only eligible for selection if every CV fold produced a non-NaN IC (mirrors champion.py's tournament eligibility gate — a NaN fold means degenerate/constant predictions in that fold, which would otherwise silently drop those weeks and inflate mean_ic).

Training window: 2005-01-07 → 2026-07-02 (455188 labeled rows).

| config | mean IC | fold ICs | test weeks | params |
|---|---|---|---|---|
| cfg2 **← selected** | 0.0155 | -0.1388, 0.0093, 0.0319, 0.0116 | 371 | {'depth': 8, 'learning_rate': 0.1, 'l2_leaf_reg': 10.0, 'iterations': 1500} |
| cfg4 | 0.0138 | -0.1388, 0.0111, 0.0198, 0.0167 | 371 | {'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 10.0, 'iterations': 1500} |
| cfg8 | 0.0111 | 0.0011, 0.0013, 0.0132, 0.0288 | 488 | {'depth': 4, 'learning_rate': 0.03, 'l2_leaf_reg': 3.0, 'iterations': 1500} |
| cfg5 | 0.0096 | -0.0141, 0.0101, 0.0232, 0.0192 | 488 | {'depth': 4, 'learning_rate': 0.1, 'l2_leaf_reg': 30.0, 'iterations': 1500} |
| cfg11 | 0.0081 | -0.0164, 0.0053, 0.0332, 0.0105 | 488 | {'depth': 4, 'learning_rate': 0.03, 'l2_leaf_reg': 30.0, 'iterations': 1500} |
| cfg10 | 0.0069 | -0.0117, -0.0006, 0.0284, 0.0114 | 488 | {'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 3.0, 'iterations': 1500} |
| cfg7 | 0.0061 | -0.0188, 0.0182, 0.0309, -0.0060 | 488 | {'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 10.0, 'iterations': 1500} |
| cfg3 | 0.0060 | -0.0115, 0.0075, 0.0147, 0.0133 | 488 | {'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 30.0, 'iterations': 1500} |
| cfg12 | 0.0059 | -0.0076, 0.0058, 0.0208, 0.0047 | 488 | {'depth': 4, 'learning_rate': 0.03, 'l2_leaf_reg': 10.0, 'iterations': 1500} |
| cfg6 | 0.0032 | -0.0115, 0.0020, 0.0221, 0.0001 | 488 | {'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 30.0, 'iterations': 1500} |
| cfg0 (production reference) | 0.0012 | -0.0182, 0.0090, 0.0200, -0.0060 | 488 | {'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 3.0, 'iterations': 1500} |
| cfg1 | 0.0012 | -0.0182, 0.0090, 0.0200, -0.0060 | 488 | {'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 3.0, 'iterations': 1500} |
| cfg9 | -0.0007 | -0.1789, -0.0079, 0.0292, -0.0162 | 371 | {'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 30.0, 'iterations': 1500} |