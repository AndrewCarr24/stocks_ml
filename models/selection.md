# Champion selection

| candidate | mean rank IC | fold ICs | test weeks |
|---|---|---|---|
| automl | 0.0258 | nan, nan, nan, 0.0258, nan | 122 |
| xgb_tuned **← champion** | 0.0237 | 0.0185, 0.0026, 0.0180, 0.0496, 0.0296 | 611 |
| xgb | 0.0135 | 0.0109, -0.0061, 0.0243, 0.0327, 0.0060 | 611 |
| momentum | 0.0017 | 0.0090, 0.0044, 0.0017, -0.0088, 0.0021 | 611 |
| zero | nan | nan, nan, nan, nan, nan | 0 |

Baselines: zero, momentum. A champion must beat every baseline or selection falls back to momentum.
excluded (degenerate predictions in 4 folds): automl