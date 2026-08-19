# Ablation: trend_quality

Paired weekly-ΔIC test on the fixed pre-holdout CV design (adoption hurdle t ≥ 3, per Harvey–Liu–Zhu).

| metric | value |
|---|---|
| features | 4: f_mom_sharpe_12w, f_mom_consist_12w, f_overnight_12w, f_intraday_12w |
| weeks compared | 492 |
| mean IC with / without | 0.0109 / 0.0104 |
| mean ΔIC (with − without) | +0.00059 |
| paired t | 0.11 |
| fold ICs with | 0.0080, 0.0203, 0.0120, 0.0035 |
| fold ICs without | 0.0059, -0.0021, 0.0125, 0.0252 |

**Verdict: do not adopt** (t < 3).
