# Ablation: edgar_sue_bundle

Paired weekly-ΔIC test on the fixed pre-holdout CV design (adoption hurdle t ≥ 3, per Harvey–Liu–Zhu).

| metric | value |
|---|---|
| features | 3: f_sue, f_nincr, f_net_issuance |
| weeks compared | 488 |
| mean IC with / without | 0.0072 / 0.0148 |
| mean ΔIC (with − without) | -0.00758 |
| paired t | -1.47 |
| fold ICs with | -0.0000, 0.0171, 0.0108, 0.0008 |
| fold ICs without | 0.0036, 0.0143, 0.0138, 0.0273 |

**Verdict: do not adopt** (t < 3).
