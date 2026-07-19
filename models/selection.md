# Champion selection

| candidate | mean rank IC | fold ICs | test weeks |
|---|---|---|---|
| xgb_tuned **← champion** | 0.0179 | 0.0158, 0.0054, 0.0074, 0.0429, 0.0182 | 611 |
| automl | 0.0149 | nan, 0.0051, 0.0136, 0.0260, nan | 366 |
| xgb | 0.0126 | -0.0037, -0.0045, 0.0209, 0.0303, 0.0200 | 611 |
| momentum | 0.0017 | 0.0094, 0.0041, 0.0018, -0.0089, 0.0019 | 611 |
| zero | nan | nan, nan, nan, nan, nan | 0 |

Baselines: zero, momentum. A champion must beat every baseline or selection falls back to momentum.
excluded (degenerate predictions in 2 folds): automl