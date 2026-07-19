# enet hyperparameter tuning

Selection is by mean weekly rank IC on the pre-holdout purged walk-forward CV folds (plain CV selection) — the untouched holdout is never used for tuning and remains the honest test.
A config is only eligible for selection if every CV fold produced a non-NaN IC (mirrors champion.py's tournament eligibility gate — a NaN fold means degenerate/constant predictions in that fold, which would otherwise silently drop those weeks and inflate mean_ic).

Training window: 2005-01-07 → 2026-07-02 (455188 labeled rows).

| config | mean IC | fold ICs | test weeks | params |
|---|---|---|---|---|
| cfg1 **← selected** | 0.0186 | 0.0162, 0.0034, 0.0146, 0.0261, 0.0326 | 611 | {'alpha': 0.001, 'l1_ratio': 0.5} |
| cfg10 | 0.0184 | 0.0168, 0.0053, 0.0143, 0.0315, 0.0244 | 611 | {'alpha': 1e-06, 'l1_ratio': 0.1} |
| cfg2 | 0.0184 | 0.0167, 0.0053, 0.0143, 0.0314, 0.0244 | 611 | {'alpha': 1e-06, 'l1_ratio': 0.5} |
| cfg8 | 0.0184 | 0.0167, 0.0053, 0.0142, 0.0314, 0.0244 | 611 | {'alpha': 1e-06, 'l1_ratio': 0.9} |
| cfg7 | 0.0184 | 0.0167, 0.0053, 0.0142, 0.0314, 0.0244 | 611 | {'alpha': 1e-05, 'l1_ratio': 0.1} |
| cfg6 | 0.0182 | 0.0167, 0.0053, 0.0137, 0.0310, 0.0243 | 611 | {'alpha': 1e-05, 'l1_ratio': 0.5} |
| cfg5 | 0.0180 | 0.0165, 0.0053, 0.0133, 0.0307, 0.0241 | 611 | {'alpha': 1e-05, 'l1_ratio': 0.9} |
| cfg9 | 0.0179 | 0.0164, 0.0053, 0.0132, 0.0307, 0.0241 | 611 | {'alpha': 0.0001, 'l1_ratio': 0.1} |
| cfg0 (production reference) | 0.0154 | 0.0156, 0.0019, 0.0115, 0.0255, 0.0225 | 611 | {'alpha': 0.0001, 'l1_ratio': 0.5} |
| cfg3 | 0.0154 | 0.0156, 0.0019, 0.0115, 0.0255, 0.0225 | 611 | {'alpha': 0.0001, 'l1_ratio': 0.5} |
| cfg4 | 0.0146 | 0.0155, 0.0011, 0.0107, 0.0221, 0.0238 | 611 | {'alpha': 0.0001, 'l1_ratio': 0.9} |
| cfg12 | 0.0146 | 0.0154, 0.0011, 0.0109, 0.0213, 0.0242 | 611 | {'alpha': 0.001, 'l1_ratio': 0.1} |
| cfg11 (ineligible: degenerate fold) | 0.0117 | 0.0159, 0.0074, nan, nan, nan | 245 | {'alpha': 0.001, 'l1_ratio': 0.9} |