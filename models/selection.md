# Champion selection

| candidate | mean rank IC | fold ICs | test weeks |
|---|---|---|---|
| automl | 0.0261 | nan, nan, 0.0261, nan | 122 |
| xgb_tuned **← champion** | 0.0241 | 0.0058, 0.0093, 0.0428, 0.0384 | 488 |
| lgbm_tuned | 0.0216 | -0.0088, 0.0092, 0.0437, 0.0411 | 483 |
| ensemble | 0.0203 | -0.0025, 0.0094, 0.0351, 0.0391 | 488 |
| enet_tuned | 0.0192 | 0.0034, 0.0146, 0.0261, 0.0326 | 488 |
| catboost_tuned | 0.0155 | -0.1388, 0.0093, 0.0319, 0.0116 | 371 |
| xgb | 0.0127 | -0.0070, 0.0098, 0.0392, 0.0090 | 488 |
| momentum | -0.0007 | 0.0068, -0.0035, -0.0051, -0.0009 | 488 |
| zero | nan | nan, nan, nan, nan | 0 |

Baselines: zero, momentum. A champion must beat every baseline or selection falls back to momentum.
excluded (degenerate predictions in 3 folds): automl