# Decline / distress feature probe

Pre-registered `decline_features_probe_v1` (2026-09-04). 9 candidates (c_netdebt_mcap, c_ebitda_debt, c_divyield, c_fcf_cash, c_rev_3y, c_margin_3y, c_rev_vs_sector, c_hi_3y, c_ret_36m) computed point-in-time on the champion panel base; statistic = mean weekly Spearman IC vs label_4w over S&P members, weeks 2002-01-01 -> labels ending before 2024-07-19 (pre-holdout only), Newey-West t lag 4; secondary = same IC within the cached top-15 picks. KEEP iff universe |t| >= 2.0 AND same IC sign in both halves (split 2013-01-01). Keepers go to a sampled paired model exam next (owner's go), not into the champion.

Universe: 578,762 member-weeks over 1172 weeks, 2002-01-04 -> 2024-06-14 (last label ends before 2024-07-19). Picks: the champion's cached top-15 per week (1164 weeks). IC = mean weekly Spearman vs label_4w; NW t = Newey-West (lag 4). Scale (AGENTS.md): 0.01 is real, 0.02 is good.

## Candidates

| feature | coverage | IC | NW t | IC 2002-12 | IC 2013-24 | picks IC | picks t | verdict |
|---|---|---|---|---|---|---|---|---|
| c_netdebt_mcap | 99% | -0.0077 | -1.0 | +0.0069 | -0.0217 | -0.027 | -2.0 | drop |
| c_ebitda_debt | 98% | +0.0107 | +1.6 | +0.0071 | +0.0141 | +0.023 | +1.9 | drop |
| c_divyield | 99% | +0.0021 | +0.3 | +0.0120 | -0.0074 | +0.017 | +1.4 | drop |
| c_fcf_cash | 98% | +0.0102 | +1.8 | +0.0082 | +0.0121 | +0.010 | +0.9 | drop |
| c_rev_3y | 94% | +0.0027 | +0.4 | -0.0009 | +0.0061 | +0.007 | +0.6 | drop |
| c_margin_3y | 94% | +0.0056 | +0.9 | +0.0102 | +0.0011 | +0.005 | +0.4 | drop |
| c_rev_vs_sector | 95% | +0.0071 | +1.2 | +0.0042 | +0.0099 | +0.017 | +1.4 | drop |
| c_hi_3y | 99% | +0.0051 | +0.4 | +0.0077 | +0.0026 | -0.004 | -0.2 | drop |
| c_ret_36m | 98% | +0.0077 | +0.7 | +0.0087 | +0.0067 | -0.001 | -0.1 | drop |

## Reference: features already in the model

| feature | coverage | IC | NW t | IC 2002-12 | IC 2013-24 | picks IC | picks t | verdict |
|---|---|---|---|---|---|---|---|---|
| f_short_dtc | 100% | -0.0282 | -3.8 | +nan | -0.0282 | -0.063 | -3.3 | in the model |
| f_mom_4w | 100% | -0.0152 | -2.1 | -0.0099 | -0.0203 | -0.020 | -1.7 | in the model |
| f_log_mktcap | 100% | -0.0206 | -4.1 | -0.0334 | -0.0164 | -0.055 | -4.2 | in the model |
| f_leverage | 100% | +0.0050 | +1.0 | -0.0068 | +0.0088 | -0.005 | -0.4 | in the model |
| f_sf_de | 100% | +0.0061 | +1.1 | +0.0057 | +0.0065 | +0.001 | +0.1 | in the model |
| f_sf_debt_ebitda | 100% | -0.0087 | -1.5 | -0.0041 | -0.0130 | -0.030 | -2.5 | in the model |
| f_sf_fcf_yield | 100% | +0.0151 | +2.3 | +0.0150 | +0.0151 | -0.012 | -1.1 | in the model |
| f_sf_revenue_yoy | 100% | +0.0082 | +1.2 | +0.0053 | +0.0111 | +0.016 | +1.3 | in the model |
| f_hi_52w | 100% | -0.0007 | -0.1 | +0.0009 | -0.0022 | -0.018 | -1.3 | in the model |
| f_mom_52w | 100% | +0.0024 | +0.2 | +0.0047 | +0.0003 | -0.017 | -1.2 | in the model |

## Verdict

Keep: none. Next step for keepers is the sampled paired model exam (champion params, same weeks, top-6 4-week return and hit10 paired vs the champion's inputs, t > 2), on the owner's go. Nothing here changes the champion.

## Definitions

- `c_netdebt_mcap`: net debt / market cap (debt - cash over price x shares): market leverage, explodes as the equity collapses
- `c_ebitda_debt`: EBITDA / debt, signed: stays defined under losses (debt/EBITDA is blank there); no debt -> +/-inf by the sign of EBITDA
- `c_divyield`: trailing 12m dividends per share / price: a double-digit yield is a cut the market expects
- `c_fcf_cash`: trailing FCF / cash: runway (negative = burning cash)
- `c_rev_3y`: trailing revenue vs 3 years earlier: both tails (secular decline, post-boom spike)
- `c_margin_3y`: EBITDA margin minus 3 years earlier
- `c_rev_vs_sector`: revenue YoY minus the same-sector member median that week (the cap's sector map): shrinking while peers grow
- `c_hi_3y`: price / 3-year high - 1
- `c_ret_36m`: 36-month price return
