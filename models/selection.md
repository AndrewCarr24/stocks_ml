# Champion selection

| candidate | mean rank IC | fold ICs | test weeks |
|---|---|---|---|
| xgb_tuned **← champion** | 0.0198 | 0.0061, 0.0206, 0.0180, 0.0343 | 488/488 |
| enet_tuned | 0.0190 | 0.0153, 0.0213, 0.0224, 0.0172 | 488/488 |
| momentum | -0.0008 | 0.0066, -0.0044, -0.0050, -0.0004 | 488/488 |
| zero | nan | nan, nan, nan, nan | 0/488 |

Baselines: zero, momentum. A champion must beat every baseline or selection falls back to momentum.