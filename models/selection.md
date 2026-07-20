# Champion selection

| candidate | mean rank IC | fold ICs | test weeks |
|---|---|---|---|
| xgb_tuned **← champion** | 0.0281 | -0.0993, 0.0418, 0.0078, 0.0369, 0.0414 | 369 |
| lgbm_tuned | 0.0268 | 0.0155, -0.0710, 0.0065, 0.0618, 0.0276 | 494 |
| automl | 0.0261 | nan, nan, nan, 0.0261, nan | 122 |
| ensemble | 0.0190 | 0.0177, 0.0033, 0.0082, 0.0335, 0.0323 | 611 |
| enet_tuned | 0.0184 | 0.0168, 0.0053, 0.0143, 0.0315, 0.0244 | 611 |
| catboost_tuned | 0.0153 | 0.0149, -0.1388, 0.0093, 0.0319, 0.0116 | 494 |
| xgb | 0.0127 | 0.0125, -0.0070, 0.0098, 0.0392, 0.0090 | 611 |
| momentum | 0.0015 | 0.0098, 0.0068, -0.0035, -0.0051, -0.0009 | 611 |
| zero | nan | nan, nan, nan, nan, nan | 0 |

Baselines: zero, momentum. A champion must beat every baseline or selection falls back to momentum.
excluded (degenerate predictions in 4 folds): automl