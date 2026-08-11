# enet Optuna tuning

TPE search, 40 trials, seed 0. Selection uses only mean weekly rank IC on the pre-holdout purged walk-forward CV folds. The holdout is never read by tuning.

| metric | value |
|---|---|
| best CV mean IC | 0.0190 |
| trials | 40 |
| eligible | 31 |
| ineligible | 9 |
| failed | 0 |
| pruned | 0 |

**Verdict: selected by CV.**

Best config: `{'alpha': 0.0001324232749711622, 'l1_ratio': 0.9897535713619371}`

## Top eligible trials

| trial | mean IC | fold ICs | coverage | best iterations |
|---:|---:|---|---:|---|
| 12 | 0.0190 | 0.0153, 0.0213, 0.0224, 0.0172 | 488/488 | n/a |
| 31 | 0.0190 | 0.0156, 0.0205, 0.0226, 0.0175 | 488/488 | n/a |
| 0 | 0.0190 | 0.0150, 0.0215, 0.0223, 0.0171 | 488/488 | n/a |
| 39 | 0.0189 | 0.0157, 0.0198, 0.0224, 0.0177 | 488/488 | n/a |
| 23 | 0.0189 | 0.0157, 0.0198, 0.0224, 0.0177 | 488/488 | n/a |
| 22 | 0.0189 | 0.0157, 0.0197, 0.0223, 0.0177 | 488/488 | n/a |
| 24 | 0.0185 | 0.0153, 0.0190, 0.0218, 0.0178 | 488/488 | n/a |
| 28 | 0.0180 | 0.0130, 0.0217, 0.0206, 0.0168 | 488/488 | n/a |
| 38 | 0.0180 | 0.0075, 0.0284, 0.0169, 0.0192 | 488/488 | n/a |
| 32 | 0.0175 | 0.0131, 0.0176, 0.0209, 0.0183 | 488/488 | n/a |

Complete trial details and the best-so-far curve are in `optuna_enet_trials.json`.
