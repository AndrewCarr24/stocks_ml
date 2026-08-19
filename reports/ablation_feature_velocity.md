# Ablation: feature_velocity

Paired weekly-ΔIC test on the fixed pre-holdout CV design (adoption hurdle t ≥ 3, per Harvey–Liu–Zhu).

| metric | value |
|---|---|
| features | 3: f_vol_chg_12w, f_beta_chg_12w, f_mom_accel_4w |
| weeks compared | 492 |
| mean IC with / without | 0.0064 / 0.0104 |
| mean ΔIC (with − without) | -0.00393 |
| paired t | -0.59 |
| fold ICs with | 0.0053, 0.0161, 0.0036, 0.0008 |
| fold ICs without | 0.0059, -0.0021, 0.0125, 0.0252 |

**Verdict: do not adopt** (t < 3).
