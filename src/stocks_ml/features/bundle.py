"""The champion's generated feature bundle: 40 formulas over the raw inputs
(``features/generated.py``: the panel's f_ columns before ranking, every SF1
field and its year-over-year change, the 8-K item counts, the nominal close),
each evaluated on the rank date and ranked within the week like every other
feature. Selected mechanically on 2006-01 -> 2015-11 alone by
``ops/openfe_arm_v2.py --generated-only`` (ledger preregistration
``openfe_arm_v3_generated_2006_2015``): every order-1 formula scored by the t
statistic of its weekly Spearman IC with the 4-week demedianed return, then a
greedy dedup in descending |t| against the 64 base features and each other
(|corr| <= 0.9), the first 40 kept. Graded once on 2016-01 -> 2024-07 with the
champion's own walk: $863 (+28.5%/yr, SR 1.13, DD 36%) vs the clean line $590
and the hand-written ideas bundle $772 (ledger ``openfe_arm_v3_generated_2016_2024``,
reports/openfe_arm_v3.md). Adopted by the owner on 2026-09-06 in place of the
ideas bundle, whose candidate ideas had been written after reading 2016-2024;
this bundle carries no such asterisk. The t beside each formula is its stage-1
score on the selection window (chance with 56,758 candidates: about 4.5).

The live job and the research walks compute the columns the same way:
``generated.add_generated(panel, generated.raw_inputs(store, cfg), FORMULAS)``.
"""
from __future__ import annotations

FORMULAS: dict[str, str] = {
    "g_00": "(r_close*r_sf_bvps)",                        # t -11.3
    "g_01": "(f_sf_sales_to_price/r_sf_bvps)",            # t +11.1
    "g_02": "(r_close/r_sf_roe)",                         # t -9.8
    "g_03": "max(f_dollar_vol,r_sf_bvps)",                # t -9.7
    "g_04": "(f_sf_sales_to_price-r_sf_bvps)",            # t +9.6
    "g_05": "(r_8k_2_02/r_sf_bvps)",                      # t +9.4
    "g_06": "(f_sf_debt_ebitda+r_close)",                 # t -9.4
    "g_07": "max(r_close,r_sf_bvps)",                     # t -9.3
    "g_08": "(f_log_mktcap+r_8k_2_03)",                   # t -9.1
    "g_09": "(f_sf_gross_prof/r_close)",                  # t +8.9
    "g_10": "(r_sf_bvps*r_sf_grossmargin)",               # t -8.9
    "g_11": "(f_idio_vol_60d*r_close)",                   # t -8.6
    "g_12": "(r_sf_bvps+r_sf_roe_yoy)",                   # t -8.5
    "g_13": "(r_sf_bvps/r_sf_de)",                        # t -8.5
    "g_14": "(r_8k_2_02/r_close)",                        # t +8.4
    "g_15": "(f_log_mktcap+r_8k_1_01)",                   # t -8.4
    "g_16": "(f_leverage-f_log_mktcap)",                  # t +8.3
    "g_17": "(f_log_mktcap-r_sf_dps_yoy)",                # t -8.3
    "g_18": "(r_sf_bvps/r_sf_revenue)",                   # t -8.2
    "g_19": "(r_close/r_sf_shareswa)",                    # t -8.2
    "g_20": "(f_sf_sales_to_price/r_sf_eps)",             # t +8.2
    "g_21": "min(f_month,r_sf_bvps)",                     # t -8.1
    "g_22": "(f_sfi_buyers_13w-r_sf_bvps)",               # t +8.0
    "g_23": "min(f_sf_fcf_yield,f_sf_gross_prof)",        # t +7.9
    "g_24": "(f_sf_roe/r_close)",                         # t +7.9
    "g_25": "(f_sf_fcf_yield/r_sf_bvps)",                 # t +7.9
    "g_26": "(f_log_mktcap-r_sf_fcf_yoy)",                # t -7.9
    "g_27": "(f_days_since_earnings_8k/r_sf_bvps)",       # t +7.8
    "g_28": "min(f_log_mktcap,r_sf_bvps)",                # t -7.8
    "g_29": "(r_sf_bvps-r_sf_divyield_yoy)",              # t -7.8
    "g_30": "(f_log_mktcap+r_sf_netinc_yoy)",             # t -7.8
    "g_31": "(f_book_to_market/r_sf_bvps)",               # t +7.8
    "g_32": "(f_log_mktcap+r_sf_epsusd)",                 # t -7.8
    "g_33": "(f_sf_current_ratio*r_sf_bvps)",             # t -7.8
    "g_34": "(f_cf_to_price/r_8k_7_01)",                  # t +7.8
    "g_35": "(f_sfi_buyers_13w+r_close)",                 # t -7.7
    "g_36": "min(f_days_since_filing,r_sf_bvps)",         # t -7.7
    "g_37": "(f_log_mktcap-r_8k_2_02)",                   # t -7.7
    "g_38": "(r_8k_9_01/r_sf_debt)",                      # t +7.7
    "g_39": "min(f_days_since_earnings_8k,r_sf_bvps)",    # t -7.6
}

FEATURES: list[str] = list(FORMULAS)          # the g_ columns, in bundle order
