# Champion selection

| candidate | mean rank IC | fold ICs | test weeks |
|---|---|---|---|
| lgbm_tuned **← champion** | 0.0169 | 0.0218, 0.0177, 0.0037, 0.0245 | 494/494 |
| enet_tuned | 0.0168 | 0.0132, 0.0220, 0.0182, 0.0136 | 494/494 |
| ensemble | 0.0165 | 0.0154, 0.0280, 0.0006, 0.0220 | 494/494 |
| catboost_tuned | 0.0125 | 0.0069, 0.0247, -0.0018, 0.0201 | 494/494 |
| automl | 0.0117 | 0.0058, nan, 0.0177, nan | 247/494 |
| xgb_tuned | 0.0111 | 0.0074, 0.0192, 0.0137, 0.0042 | 494/494 |
| xgb | 0.0034 | -0.0047, 0.0091, -0.0111, 0.0202 | 494/494 |
| momentum | -0.0020 | 0.0076, -0.0046, -0.0035, -0.0075 | 494/494 |
| zero | nan | nan, nan, nan, nan | 0/494 |

Baselines: zero, momentum. A champion must beat every baseline or selection falls back to momentum.
excluded (invalid predictions in 2 folds; coverage 247/494 weeks): automl