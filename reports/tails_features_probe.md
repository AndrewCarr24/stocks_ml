# Tails feature probe: risk factors and growth the model misses

Pre-registered `tails_features_probe_v1` (2026-09-04). 30 candidates (c_8k_distress_26w, c_8k_officer_26w, c_8k_other_26w, c_ins_sell_26w, c_accrual, c_bvps_chg_2q, c_div_cover, c_dps_yoy, c_equity_yoy, c_rev_streak, c_cash_runway, c_partial_z, c_jump_dn_4w, c_jump_up_4w, c_price_level, c_rev_accel, c_rev_accel_4q, c_margin_z_5y, c_gm_chg, c_earn_react_last, c_earn_react_mean4, c_earn_window, c_seasonal_3y, c_peer_mom_4w, c_peer_mom_12w, c_own_vs_peer_4w, c_peer_rev_gap, c_lev_x_peer_mom12, c_beta_x_mkt4w, c_margin_trough) plus the 18 never-ablated PENDING_ABLATION_FEATURES as a separate block; universe = champion panel base, 2002-01-01 -> labels ending before 2024-07-19; IC = mean weekly Spearman vs label_4w, NW t lag 4; dense rule (defined & non-zero >= 10%): keep iff |t| >= 2.0 AND same IC sign in 2002-2012 and 2013-2024; sparse rule: weekly mean label_4w of flagged names (>= 3 flagged), keep iff |NW t| >= 2.0 AND same sign both eras. Keepers -> sampled paired model exam (owner's go); nothing changes the champion.

Universe: 578,762 member-weeks over 1172 weeks, 2002-01-04 -> 2024-06-14. Picks: the champion's cached top-15 per week (1164 weeks). Dense features: value = mean weekly Spearman IC vs label_4w (0.01 real, 0.02 good). Sparse flags: value = mean weekly label_4w (return minus the member median, 4 weeks) of flagged names. NW t = Newey-West, lag 4.

## Candidates

| feature | defined / active | stat | value | NW t | 2002-12 | 2013-24 | picks IC | picks t | verdict |
|---|---|---|---|---|---|---|---|---|---|
| c_8k_distress_26w | 100% / 10% | IC | +0.0068 | +2.6 | +0.0096 | +0.0047 | +0.009 | +0.7 | KEEP |
| c_8k_officer_26w | 100% / 46% | IC | +0.0079 | +2.8 | +0.0058 | +0.0095 | +0.031 | +2.6 | KEEP |
| c_8k_other_26w | 100% / 43% | IC | +0.0057 | +1.8 | +0.0062 | +0.0053 | +0.015 | +1.2 | drop |
| c_ins_sell_26w | 99% / 54% | IC | +0.0086 | +1.7 | +0.0071 | +0.0096 | +0.026 | +2.1 | drop |
| c_accrual | 98% / 98% | IC | -0.0099 | -2.2 | -0.0156 | -0.0043 | -0.016 | -1.5 | KEEP |
| c_bvps_chg_2q | 94% / 94% | IC | +0.0092 | +1.9 | +0.0091 | +0.0093 | +0.012 | +1.0 | drop |
| c_div_cover | 77% / 77% | IC | +0.0135 | +1.9 | -0.0018 | +0.0281 | +0.000 | +0.0 | drop |
| c_dps_yoy | 74% / 63% | IC | +0.0114 | +2.0 | +0.0092 | +0.0135 | +0.019 | +1.0 | drop |
| c_equity_yoy | 92% / 92% | IC | -0.0035 | -0.7 | -0.0048 | -0.0022 | -0.010 | -0.8 | drop |
| c_rev_streak | 95% / 28% | IC | -0.0095 | -1.6 | -0.0082 | -0.0108 | -0.020 | -1.6 | drop |
| c_cash_runway | 98% / 98% | IC | +0.0144 | +3.0 | +0.0095 | +0.0190 | +0.020 | +1.7 | KEEP |
| c_partial_z | 98% / 98% | IC | +0.0064 | +0.8 | -0.0033 | +0.0157 | +0.022 | +1.6 | drop |
| c_jump_dn_4w | 100% / 100% | IC | -0.0023 | -0.2 | +0.0008 | -0.0053 | -0.020 | -1.7 | drop |
| c_jump_up_4w | 100% / 100% | IC | -0.0052 | -0.5 | -0.0090 | -0.0014 | +0.007 | +0.6 | drop |
| c_price_level | 100% / 100% | IC | -0.0248 | -3.8 | -0.0424 | -0.0079 | -0.047 | -4.1 | KEEP |
| c_rev_accel | 94% / 94% | IC | +0.0061 | +1.4 | -0.0054 | +0.0171 | +0.017 | +1.6 | drop |
| c_rev_accel_4q | 90% / 90% | IC | +0.0079 | +1.4 | +0.0090 | +0.0068 | +0.018 | +1.5 | drop |
| c_margin_z_5y | 96% / 96% | IC | +0.0044 | +0.8 | +0.0096 | -0.0006 | +0.007 | +0.6 | drop |
| c_gm_chg | 94% / 84% | IC | -0.0004 | -0.1 | +0.0047 | -0.0053 | +0.011 | +0.9 | drop |
| c_earn_react_last | 65% / 65% | IC | +0.0024 | +0.6 | +0.0030 | +0.0020 | -0.005 | -0.3 | drop |
| c_earn_react_mean4 | 65% / 65% | IC | +0.0124 | +2.7 | +0.0112 | +0.0133 | +0.002 | +0.1 | KEEP |
| c_earn_window | 100% / 19% | IC | +0.0121 | +4.0 | +0.0159 | +0.0094 | +0.032 | +2.6 | KEEP |
| c_seasonal_3y | 98% / 98% | IC | -0.0039 | -0.7 | +0.0080 | -0.0153 | -0.003 | -0.3 | drop |
| c_peer_mom_4w | 99% / 99% | IC | -0.0143 | -1.6 | +0.0015 | -0.0294 | -0.015 | -1.2 | drop |
| c_peer_mom_12w | 99% / 99% | IC | -0.0139 | -1.4 | -0.0036 | -0.0237 | -0.012 | -1.0 | drop |
| c_own_vs_peer_4w | 99% / 99% | IC | -0.0085 | -2.0 | -0.0098 | -0.0073 | -0.012 | -1.2 | drop |
| c_peer_rev_gap | 95% / 95% | IC | +0.0052 | +1.2 | +0.0031 | +0.0071 | +0.021 | +1.8 | drop |
| c_lev_x_peer_mom12 | 98% / 95% | IC | -0.0051 | -0.6 | +0.0149 | -0.0242 | -0.011 | -0.8 | drop |
| c_beta_x_mkt4w | 99% / 99% | IC | -0.0046 | -0.4 | +0.0038 | -0.0127 | -0.014 | -1.1 | drop |
| c_margin_trough | 97% / 76% | IC | +0.0095 | +1.6 | +0.0140 | +0.0053 | +0.010 | +0.8 | drop |

## Pending-ablation panel features (never tested; same rule)

| feature | defined / active | stat | value | NW t | 2002-12 | 2013-24 | picks IC | picks t | verdict |
|---|---|---|---|---|---|---|---|---|---|
| f_beta_chg_12w | 100% / 99% | IC | -0.0046 | -0.6 | +0.0092 | -0.0179 | +0.008 | +0.7 | drop |
| f_intraday_12w | 100% / 99% | IC | -0.0041 | -0.5 | -0.0019 | -0.0062 | -0.014 | -1.1 | drop |
| f_mom_12w_skip1w | 100% / 99% | IC | -0.0096 | -1.1 | -0.0070 | -0.0121 | -0.011 | -0.9 | drop |
| f_mom_52w_skip4w | 100% / 99% | IC | +0.0054 | +0.5 | +0.0074 | +0.0035 | -0.012 | -0.8 | drop |
| f_mom_accel_4w | 100% / 99% | IC | -0.0100 | -1.6 | -0.0090 | -0.0109 | -0.005 | -0.5 | drop |
| f_mom_consist_12w | 100% / 99% | IC | -0.0038 | -0.5 | -0.0019 | -0.0056 | -0.024 | -2.2 | drop |
| f_mom_interm | 100% / 99% | IC | +0.0072 | +0.8 | +0.0141 | +0.0006 | +0.003 | +0.2 | drop |
| f_mom_sharpe_12w | 100% / 99% | IC | -0.0117 | -1.4 | -0.0134 | -0.0100 | -0.018 | -1.6 | drop |
| f_net_issuance | 100% / 38% | IC | -0.0089 | -1.7 | -0.0132 | -0.0079 | -0.013 | -1.0 | drop |
| f_nincr | 100% / 41% | IC | -0.0010 | -0.2 | -0.0046 | +0.0002 | +0.007 | +0.5 | drop |
| f_overnight_12w | 100% / 99% | IC | -0.0123 | -2.3 | -0.0098 | -0.0148 | +0.003 | +0.2 | KEEP |
| f_resid_ret_lag5w | 100% / 99% | IC | -0.0036 | -0.8 | -0.0059 | -0.0014 | -0.011 | -1.2 | drop |
| f_resid_ret_lag6w | 100% / 99% | IC | +0.0015 | +0.3 | -0.0009 | +0.0037 | +0.008 | +0.8 | drop |
| f_resid_ret_lag7w | 100% / 99% | IC | -0.0007 | -0.2 | -0.0046 | +0.0031 | -0.001 | -0.1 | drop |
| f_resid_ret_lag8w | 100% / 99% | IC | -0.0019 | -0.4 | -0.0032 | -0.0006 | +0.016 | +1.9 | drop |
| f_short_chg_8w | 100% / 22% | IC | +0.0034 | +0.7 | +nan | +0.0034 | -0.012 | -0.6 | drop |
| f_sue | 100% / 38% | IC | -0.0018 | -0.5 | -0.0011 | -0.0019 | +0.005 | +0.4 | drop |
| f_vol_chg_12w | 100% / 99% | IC | -0.0050 | -0.9 | -0.0131 | +0.0027 | +0.012 | +1.1 | drop |

## Reference: features already in the model

| feature | defined / active | stat | value | NW t | 2002-12 | 2013-24 | picks IC | picks t | verdict |
|---|---|---|---|---|---|---|---|---|---|
| f_short_dtc | 100% / 28% | IC | -0.0282 | -3.8 | +nan | -0.0282 | -0.063 | -3.3 | in the model |
| f_mom_4w | 100% / 99% | IC | -0.0152 | -2.1 | -0.0099 | -0.0203 | -0.020 | -1.7 | in the model |
| f_log_mktcap | 100% / 43% | IC | -0.0206 | -4.1 | -0.0334 | -0.0164 | -0.055 | -4.2 | in the model |
| f_pead | 100% / 47% | IC | -0.0004 | -0.1 | +0.0094 | -0.0036 | -0.025 | -1.9 | in the model |
| f_sf_fcf_yield | 100% / 98% | IC | +0.0151 | +2.3 | +0.0150 | +0.0151 | -0.012 | -1.1 | in the model |
| f_evt_earnings_8k_7d | 100% / 5% | IC | +0.0034 | +2.0 | +0.0020 | +0.0044 | -0.001 | -0.1 | in the model |
| f_days_since_earnings_8k | 100% / 65% | IC | -0.0001 | -0.0 | -0.0044 | +0.0030 | +0.006 | +0.6 | in the model |

## Verdict

Keep (candidates): c_8k_distress_26w, c_8k_officer_26w, c_accrual, c_cash_runway, c_price_level, c_earn_react_mean4, c_earn_window. Keep (pending block): f_overnight_12w. Keepers go to the sampled paired model exam (champion params, same weeks, top-6 4-week return primary, t > 2) on the owner's go. Nothing here changes the champion.

## Definitions

- `c_8k_distress_26w`: count of 8-Ks in the last 26 weeks with items 2.04 triggering event / 2.05 exit or disposal costs / 2.06 impairment / 3.01 delisting notice / 4.01 auditor change / 1.02 termination of a material agreement
- `c_8k_officer_26w`: count of 8-Ks in the last 26 weeks with item 5.02 (officer or director departure / appointment)
- `c_8k_other_26w`: count of 8-Ks in the last 26 weeks with item 8.01 'other events' (the PG&E flood of fire filings)
- `c_ins_sell_26w`: Form-4 open-market sales (code S) over the last 26 weeks, dollars / market cap (Mozilo, Ahearn, the homebuilders 2005-06)
- `c_accrual`: (net income - operating cash flow) / assets, trailing 12m: profits the cash never confirmed (GNRC, SEDG, BIG, the homebuilders)
- `c_bvps_chg_2q`: book value per share vs two quarters earlier: impairments and OCI losses hit the book before the P&L
- `c_div_cover`: trailing FCF / dividends paid (payers only): the dividend the cash flow cannot carry (LUMN, BIG, FCX, PCG)
- `c_dps_yoy`: trailing 12m dividends per share vs a year earlier (payers a year ago): a cut is the board conceding
- `c_equity_yoy`: book equity vs a year earlier (ARQ): write-downs and losses eating the equity
- `c_rev_streak`: consecutive quarters of year-over-year revenue decline (0-8)
- `c_cash_runway`: cash / trailing cash burn (years) when FCF < 0, +inf otherwise
- `c_partial_z`: Altman Z without the working-capital and retained-earnings terms: 3.3 EBIT/assets + 0.6 mcap/liabilities + revenue/assets
- `c_jump_dn_4w`: worst single-day return in the last 20 trading days (a news crash vs a drift)
- `c_jump_up_4w`: best single-day return in the last 20 trading days (lottery / MAX effect)
- `c_price_level`: log nominal share price: single-digit prices in the S&P 500 are distress
- `c_rev_accel`: revenue YoY growth minus the prior quarter's YoY growth: the cycle turning
- `c_rev_accel_4q`: revenue YoY growth minus the YoY growth four quarters earlier: the slower deceleration (PYPL, GNRC, SEDG)
- `c_margin_z_5y`: EBITDA margin (trailing 12m) standardized against the firm's own trailing 5 years: the cyclical peak (MOS 2008, homebuilders 2006)
- `c_gm_chg`: gross margin (trailing 12m) minus a year earlier: pricing power arriving or leaving
- `c_earn_react_last`: market-adjusted 2-day return around the last earnings 8-K (item 2.02): the true post-earnings drift
- `c_earn_react_mean4`: mean of that reaction over the last four earnings days: earnings-announcement-return momentum
- `c_earn_window`: 1 if the next earnings 8-K is expected inside the 4-week label window (last 2.02 date + the ticker's median gap)
- `c_seasonal_3y`: the stock's own return over the same 4-week calendar window in each of the prior 3 years, averaged
- `c_peer_mom_4w`: mean 4-week return of the peer group: industry momentum without a sector map
- `c_peer_mom_12w`: mean 12-week return of the peer group
- `c_own_vs_peer_4w`: own 4-week return minus the peer group's: the within-industry reversal
- `c_peer_rev_gap`: own revenue YoY minus the peer group's median: shrinking while peers grow
- `c_lev_x_peer_mom12`: debt / market cap times the peer group's 12-week return: the levered cyclical when its industry turns (THC 2009, FCX 2016, APA 2020)
- `c_beta_x_mkt4w`: 250-day beta to SPY times SPY's 4-week return: high beta after the market turns (the March-2009 junk rally)
- `c_margin_trough`: EBITDA margin (trailing 12m) minus its own 12-quarter high: depth of the margin depression (MU, AMD, oil 2020)
