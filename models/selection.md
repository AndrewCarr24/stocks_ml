# Champion selection

| candidate | mean rank IC | fold ICs | test weeks |
|---|---|---|---|
| automl | 0.0261 | nan, nan, nan, 0.0261, nan | 122 |
| xgb_tuned **← champion** | 0.0232 | 0.0199, 0.0058, 0.0093, 0.0428, 0.0384 | 611 |
| xgb | 0.0127 | 0.0125, -0.0070, 0.0098, 0.0392, 0.0090 | 611 |
| momentum | 0.0015 | 0.0098, 0.0068, -0.0035, -0.0051, -0.0009 | 611 |
| zero | nan | nan, nan, nan, nan, nan | 0 |

Baselines: zero, momentum. A champion must beat every baseline or selection falls back to momentum.
excluded (degenerate predictions in 4 folds): automl