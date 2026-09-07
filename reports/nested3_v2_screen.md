# Feature screen: nested3_v2

Selection window 2006-01-01 -> 2015-12-31; label `label_4w`; probe weeks end inside the window (last rank date 2015-12-31 minus 35 days). Rule (registered, feature_screen.py): dense features keep iff |NW t| >= 2.0 (lag 4) and the same IC sign in both halves of the window; sparse flags (active < 10%) use the weekly mean label of flagged names. Keepers are examined as one bundle at the run's engine; ADMITTED iff the top-6 book's cost-adjusted compounded %/yr on the exam weeks is higher with the bundle than without (the book layer's metric; v3.1). The paired calendar-HAC t is reported.

## Probe

| feature | defined / active | stat | value | NW t | first half | second half | weeks | verdict |
|---|---|---|---|---|---|---|---|---|
| x_8k_distress_26w | 100% / 12% | IC | +0.0106 | +2.9 | +0.0080 | +0.0133 | 516 | KEEP |
| x_8k_officer_26w | 100% / 47% | IC | +0.0077 | +1.8 | +0.0080 | +0.0075 | 516 | drop |
| x_8k_other_26w | 100% / 45% | IC | +0.0030 | +0.7 | +0.0022 | +0.0039 | 516 | drop |
| x_accrual | 99% / 99% | IC | -0.0105 | -1.6 | -0.0205 | -0.0002 | 516 | drop |
| x_beta_x_mkt4w | 99% / 99% | IC | +0.0118 | +0.7 | +0.0269 | -0.0038 | 516 | drop |
| x_bvps_chg_2q | 96% / 96% | IC | +0.0054 | +0.8 | +0.0074 | +0.0033 | 516 | drop |
| x_cash_runway | 99% / 99% | IC | +0.0178 | +2.8 | +0.0121 | +0.0237 | 516 | KEEP |
| x_div_cover | 77% / 77% | IC | +0.0059 | +0.5 | -0.0094 | +0.0216 | 516 | drop |
| x_dps_yoy | 73% / 63% | IC | +0.0181 | +2.2 | +0.0049 | +0.0315 | 516 | KEEP |
| x_earn_react_last | 66% / 66% | IC | +0.0074 | +1.3 | -0.0012 | +0.0161 | 516 | drop |
| x_earn_react_mean4 | 65% / 65% | IC | +0.0169 | +2.5 | +0.0096 | +0.0243 | 516 | KEEP |
| x_earn_window | 100% / 19% | IC | +0.0108 | +2.7 | +0.0144 | +0.0071 | 516 | KEEP |
| x_equity_yoy | 94% / 94% | IC | -0.0121 | -1.7 | -0.0031 | -0.0214 | 516 | drop |
| x_gm_chg | 95% / 84% | IC | -0.0040 | -0.7 | -0.0055 | -0.0025 | 516 | drop |
| x_ins_sell_26w | 99% / 56% | IC | +0.0083 | +1.6 | +0.0020 | +0.0147 | 516 | drop |
| x_jump_dn_4w | 100% / 100% | IC | +0.0169 | +1.0 | +0.0100 | +0.0241 | 516 | drop |
| x_jump_up_4w | 100% / 100% | IC | -0.0105 | -0.7 | -0.0111 | -0.0098 | 516 | drop |
| x_lev_x_peer_mom12 | 99% / 94% | IC | +0.0186 | +1.6 | +0.0080 | +0.0293 | 516 | drop |
| x_margin_trough | 98% / 77% | IC | +0.0146 | +1.8 | +0.0134 | +0.0157 | 516 | drop |
| x_margin_z_5y | 97% / 97% | IC | +0.0064 | +0.8 | +0.0001 | +0.0130 | 516 | drop |
| x_own_vs_peer_4w | 99% / 99% | IC | -0.0056 | -0.8 | -0.0051 | -0.0060 | 516 | drop |
| x_partial_z | 99% / 99% | IC | +0.0116 | +1.1 | +0.0095 | +0.0138 | 516 | drop |
| x_peer_mom_12w | 99% / 99% | IC | +0.0045 | +0.3 | -0.0030 | +0.0122 | 516 | drop |
| x_peer_mom_4w | 99% / 99% | IC | +0.0106 | +0.8 | +0.0164 | +0.0046 | 516 | drop |
| x_peer_rev_gap | 95% / 95% | IC | +0.0055 | +0.8 | +0.0020 | +0.0091 | 516 | drop |
| x_price_level | 100% / 100% | IC | -0.0280 | -3.7 | -0.0348 | -0.0211 | 516 | KEEP |
| x_rev_accel | 94% / 94% | IC | +0.0013 | +0.2 | -0.0107 | +0.0137 | 516 | drop |
| x_rev_accel_4q | 91% / 91% | IC | +0.0100 | +1.3 | +0.0032 | +0.0169 | 516 | drop |
| x_rev_streak | 95% / 29% | IC | -0.0135 | -1.6 | -0.0135 | -0.0136 | 516 | drop |
| x_seasonal_3y | 98% / 98% | IC | -0.0017 | -0.2 | -0.0063 | +0.0031 | 516 | drop |
| f_beta_chg_12w | 100% / 99% | IC | +0.0037 | +0.4 | +0.0144 | -0.0072 | 516 | drop |
| f_intraday_12w | 100% / 99% | IC | +0.0055 | +0.5 | +0.0004 | +0.0108 | 516 | drop |
| f_mom_12w_skip1w | 100% / 99% | IC | +0.0039 | +0.3 | -0.0053 | +0.0134 | 516 | drop |
| f_mom_52w_skip4w | 100% / 99% | IC | +0.0105 | +0.6 | -0.0165 | +0.0382 | 516 | drop |
| f_mom_accel_4w | 100% / 99% | IC | +0.0011 | +0.1 | -0.0045 | +0.0068 | 516 | drop |
| f_mom_consist_12w | 100% / 99% | IC | -0.0006 | -0.1 | -0.0104 | +0.0094 | 516 | drop |
| f_mom_interm | 100% / 99% | IC | +0.0162 | +1.2 | -0.0032 | +0.0361 | 516 | drop |
| f_mom_sharpe_12w | 100% / 99% | IC | +0.0003 | +0.0 | -0.0141 | +0.0151 | 516 | drop |
| f_net_issuance | 100% / 25% | IC | -0.0125 | -2.1 | +0.0052 | -0.0157 | 302 | drop |
| f_nincr | 100% / 27% | IC | -0.0013 | -0.2 | -0.0168 | +0.0039 | 341 | drop |
| f_overnight_12w | 100% / 99% | IC | -0.0038 | -0.5 | -0.0083 | +0.0008 | 516 | drop |
| f_resid_ret_lag5w | 100% / 99% | IC | +0.0022 | +0.4 | -0.0019 | +0.0063 | 516 | drop |
| f_resid_ret_lag6w | 100% / 99% | IC | +0.0049 | +0.8 | +0.0032 | +0.0067 | 516 | drop |
| f_resid_ret_lag7w | 100% / 99% | IC | -0.0003 | -0.0 | -0.0054 | +0.0050 | 516 | drop |
| f_resid_ret_lag8w | 100% / 99% | IC | -0.0010 | -0.2 | -0.0126 | +0.0109 | 516 | drop |
| f_short_chg_8w | 100% / 0% | flag mean | +nan | +nan | +nan | +nan | 0 | drop |
| f_sue | 100% / 21% | IC | +0.0021 | +0.5 | +0.0134 | +0.0010 | 278 | drop |
| f_vol_chg_12w | 100% / 99% | IC | -0.0241 | -2.9 | -0.0320 | -0.0159 | 516 | KEEP |

Keepers: x_8k_distress_26w, x_cash_runway, x_dps_yoy, x_earn_react_mean4, x_earn_window, x_price_level, f_vol_chg_12w.

## Exam

508 weeks 2006-01-06 -> 2015-11-20; SPY mean +0.61%, member mean +0.79%; forward returns on the fill basis, raw. Calendar-HAC t, 28-day bandwidth.

| statistic | without | with | diff (with - without) | HAC t | weeks with > without |
|---|---|---|---|---|---|
| top6 (primary) | +0.98% | +1.43% | +0.45% | +1.61 | 53% |
| top3 | +0.86% | +1.94% | +1.08% | +2.44 | 51% |
| top10 | +0.78% | +1.10% | +0.31% | +1.43 | 54% |

Top-6 cost-adjusted compounded %/yr on these weeks: without 4.03, with 10.16 (paired HAC t +1.61).

## Verdict

ADMITTED: the top-6 book compounds faster with the bundle (10.16 vs 4.03 %/yr); ['x_8k_distress_26w', 'x_cash_runway', 'x_dps_yoy', 'x_earn_react_mean4', 'x_earn_window', 'x_price_level', 'f_vol_chg_12w'] enters this run.
