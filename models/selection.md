# Champion selection

| candidate | mean rank IC | fold ICs | test weeks |
|---|---|---|---|
| xgb_tuned **← champion** | 0.0229 | 0.0060, 0.0188, 0.0340, 0.0331 | 490 |
| enet_tuned | 0.0178 | 0.0038, 0.0157, 0.0280, 0.0240 | 490 |
| ensemble | 0.0146 | -0.0015, 0.0102, 0.0259, 0.0238 | 490 |
| automl | 0.0131 | nan, 0.0131, nan, nan | 123 |
| xgb | 0.0128 | -0.0036, 0.0112, 0.0359, 0.0079 | 490 |
| lgbm_tuned | 0.0128 | -0.0070, 0.0029, 0.0383, 0.0163 | 485 |
| catboost_tuned | 0.0103 | -0.1388, 0.0053, 0.0171, 0.0147 | 372 |
| momentum | -0.0014 | 0.0076, -0.0033, -0.0072, -0.0028 | 490 |
| zero | nan | nan, nan, nan, nan | 0 |

Baselines: zero, momentum. A champion must beat every baseline or selection falls back to momentum.
excluded (degenerate predictions in 3 folds): automl