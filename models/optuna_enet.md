# enet Optuna tuning

TPE search, 40 trials, seed 0. Optimized on CV folds; the winner is adopted only if it also beats the random-search config on the untouched holdout.

| metric | value |
|---|---|
| best CV mean IC | 0.0184 |
| best config holdout IC | 0.0231 |
| incumbent (random-search) holdout IC | 0.0016 |
| trials | 40 |

**Verdict: adopted: holdout improved.**

Best config: `{'alpha': 4.089358338350741e-07, 'l1_ratio': 0.11188877949697237}`
