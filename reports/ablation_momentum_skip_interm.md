# Ablation: momentum_skip_interm

Paired weekly-ΔIC test on the fixed pre-holdout CV design (adoption hurdle t ≥ 3, per Harvey–Liu–Zhu).

| metric | value |
|---|---|
| features | 3: f_mom_12w_skip1w, f_mom_52w_skip4w, f_mom_interm |
| weeks compared | 488 |
| mean IC with / without | 0.0095 / 0.0148 |
| mean ΔIC (with − without) | -0.00522 |
| paired t | -0.94 |
| fold ICs with | 0.0067, 0.0196, 0.0109, 0.0010 |
| fold ICs without | 0.0036, 0.0143, 0.0138, 0.0273 |

**Verdict: do not adopt** (t < 3).
