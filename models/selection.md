# Champion selection

| candidate | mean rank IC | fold ICs | test weeks |
|---|---|---|---|
| automl | 0.0261 | nan, nan, nan, 0.0261, nan | 122 |
| enet_tuned **← champion** | 0.0186 | 0.0162, 0.0034, 0.0146, 0.0261, 0.0326 | 611 |
| xgb_tuned | 0.0182 | 0.0139, 0.0049, 0.0045, 0.0383, 0.0295 | 611 |
| lgbm_tuned | 0.0169 | 0.0131, -0.0088, 0.0092, 0.0396, 0.0303 | 606 |
| ensemble | 0.0162 | 0.0152, -0.0009, 0.0058, 0.0284, 0.0323 | 611 |
| xgb | 0.0109 | 0.0102, -0.0058, 0.0205, 0.0267, 0.0028 | 611 |
| catboost_tuned | 0.0085 | 0.0016, -0.0442, 0.0131, 0.0271, -0.0056 | 494 |
| momentum | 0.0015 | 0.0098, 0.0068, -0.0035, -0.0051, -0.0009 | 611 |
| zero | nan | nan, nan, nan, nan, nan | 0 |

Baselines: zero, momentum. A champion must beat every baseline or selection falls back to momentum.
excluded (degenerate predictions in 4 folds): automl