# Champion selection

| candidate | mean rank IC | fold ICs | test weeks |
|---|---|---|---|
| automl | 0.0261 | nan, nan, nan, 0.0261, nan | 122 |
| xgb_tuned **← champion** | 0.0232 | 0.0199, 0.0058, 0.0093, 0.0428, 0.0384 | 611 |
| lgbm_tuned | 0.0198 | 0.0131, -0.0088, 0.0092, 0.0437, 0.0411 | 606 |
| ensemble | 0.0198 | 0.0180, -0.0025, 0.0094, 0.0351, 0.0391 | 611 |
| enet_tuned | 0.0186 | 0.0162, 0.0034, 0.0146, 0.0261, 0.0326 | 611 |
| catboost_tuned | 0.0153 | 0.0149, -0.1388, 0.0093, 0.0319, 0.0116 | 494 |
| xgb | 0.0127 | 0.0125, -0.0070, 0.0098, 0.0392, 0.0090 | 611 |
| momentum | 0.0015 | 0.0098, 0.0068, -0.0035, -0.0051, -0.0009 | 611 |
| zero | nan | nan, nan, nan, nan, nan | 0 |

Baselines: zero, momentum. A champion must beat every baseline or selection falls back to momentum.
excluded (degenerate predictions in 4 folds): automl