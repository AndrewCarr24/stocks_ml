# Champion selection

| candidate | mean rank IC | fold ICs | test weeks |
|---|---|---|---|
<<<<<<< HEAD
| xgb_tuned **← champion** | 0.0198 | 0.0061, 0.0206, 0.0180, 0.0343 | 488/488 |
| enet_tuned | 0.0190 | 0.0153, 0.0213, 0.0224, 0.0172 | 488/488 |
| momentum | -0.0008 | 0.0066, -0.0044, -0.0050, -0.0004 | 488/488 |
| zero | nan | nan, nan, nan, nan | 0/488 |
=======
| xgb_tuned **← champion** | 0.0229 | 0.0060, 0.0188, 0.0340, 0.0331 | 490 |
| enet_tuned | 0.0178 | 0.0038, 0.0157, 0.0280, 0.0240 | 490 |
| ensemble | 0.0146 | -0.0015, 0.0102, 0.0259, 0.0238 | 490 |
| automl | 0.0131 | nan, 0.0131, nan, nan | 123 |
| xgb | 0.0128 | -0.0036, 0.0112, 0.0359, 0.0079 | 490 |
| lgbm_tuned | 0.0128 | -0.0070, 0.0029, 0.0383, 0.0163 | 485 |
| catboost_tuned | 0.0103 | -0.1388, 0.0053, 0.0171, 0.0147 | 372 |
| momentum | -0.0014 | 0.0076, -0.0033, -0.0072, -0.0028 | 490 |
| zero | nan | nan, nan, nan, nan | 0 |
>>>>>>> 9e786135797bacbff3daa615bf8a5c4a39719c32

Baselines: zero, momentum. A champion must beat every baseline or selection falls back to momentum.