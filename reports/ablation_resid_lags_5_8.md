# Ablation: resid_lags_5_8

Paired weekly-ΔIC test on the fixed pre-holdout CV design (adoption hurdle t ≥ 3, per Harvey–Liu–Zhu).

| metric | value |
|---|---|
| features | 4: f_resid_ret_lag5w, f_resid_ret_lag6w, f_resid_ret_lag7w, f_resid_ret_lag8w |
| weeks compared | 492 |
| mean IC with / without | 0.0043 / 0.0104 |
| mean ΔIC (with − without) | -0.00604 |
| paired t | -0.93 |
| fold ICs with | 0.0058, 0.0070, -0.0003, 0.0048 |
| fold ICs without | 0.0059, -0.0021, 0.0125, 0.0252 |

**Verdict: do not adopt** (t < 3).
