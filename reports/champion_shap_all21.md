# Champion feature importance (Shapley values)

Recipe {'label': 'label_4w_sector_rank', 'train_years': 8, 'features': ['x_hl_pos_day_4w', 'x_hl_range_pos_20d', 'x_hl_range_20d', 'x_hl_park_vol_12w', 'x_hl_intraday_share', 'x_sv_vol_4w_sic', 'x_sv_vol_12w_sic', 'x_sv_downside_dev_sic', 'x_sv_idio_vol_60d_sic', 'x_sv_vol_4w_s11', 'x_sv_vol_12w_s11', 'x_sv_downside_dev_s11', 'x_sv_idio_vol_60d_s11', 'x_vx_vol_4w_x_size', 'x_vx_vol_12w_x_size', 'x_vx_downside_dev_x_size', 'x_vx_idio_vol_60d_x_size', 'x_cm_dip_1w', 'x_cm_dip_4w', 'x_cm_mom_4w', 'x_cm_corr'], 'drop': [], 'params': {'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1500, 'min_child_weight': 20, 'subsample': 0.85, 'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.0, 'gamma': 0.01, 'max_bin': 256, 'tree_method': 'hist'}}; one yearly fit at each year's first rank week (2007-01-05 → 2015-01-02), exact TreeSHAP on every member scored that week. mean |SHAP| in score units; share = of the total across features; sign = rank correlation between the feature's value and its contribution (positive: higher raises the score).

| rank | feature | mean abs SHAP | share | sd across years | sign |
|---|---|---|---|---|---|
| 1 | sales / price (`f_sf_sales_to_price`) | 0.0162 | 14.1% | 0.0110 | +0.84 |
| 2 | free-cash-flow yield (`f_sf_fcf_yield`) | 0.0105 | 9.2% | 0.0111 | +0.72 |
| 3 | book / market (`f_sf_book_to_market`) | 0.0040 | 3.5% | 0.0045 | +0.44 |
| 4 | x_cm_corr (`x_cm_corr`) | 0.0033 | 2.8% | 0.0030 | -0.00 |
| 5 | beta (60d) (`f_beta_60d`) | 0.0031 | 2.7% | 0.0013 | +0.09 |
| 6 | downside deviation (`f_downside_dev`) | 0.0027 | 2.4% | 0.0015 | -0.02 |
| 7 | negative EBITDA (flag) (`f_sf_neg_ebitda`) | 0.0025 | 2.2% | 0.0017 | +nan |
| 8 | gross profitability (`f_sf_gross_prof`) | 0.0025 | 2.2% | 0.0015 | -0.09 |
| 9 | insider net buying (13w) (`f_sfi_net_13w`) | 0.0024 | 2.1% | 0.0023 | +0.38 |
| 10 | dollar volume (`f_dollar_vol`) | 0.0023 | 2.0% | 0.0008 | +0.20 |
| 11 | x_hl_range_pos_20d (`x_hl_range_pos_20d`) | 0.0022 | 1.9% | 0.0027 | -0.58 |
| 12 | x_cm_mom_4w (`x_cm_mom_4w`) | 0.0021 | 1.8% | 0.0022 | +0.36 |
| 13 | x_sv_idio_vol_60d_sic (`x_sv_idio_vol_60d_sic`) | 0.0021 | 1.8% | 0.0015 | -0.21 |
| 14 | 12-week return (`f_mom_12w`) | 0.0020 | 1.7% | 0.0021 | -0.37 |
| 15 | market cap (`f_log_mktcap`) | 0.0020 | 1.7% | 0.0021 | -0.77 |
| 16 | fed funds rate (`f_macro_FEDFUNDS`) | 0.0019 | 1.7% | 0.0023 | +nan |
| 17 | insider net buying, Form 4 (13w) (`f_insider_net_13w`) | 0.0019 | 1.6% | 0.0019 | -0.22 |
| 18 | distance from 52-week low (`f_lo_52w`) | 0.0019 | 1.6% | 0.0013 | +0.40 |
| 19 | x_hl_park_vol_12w (`x_hl_park_vol_12w`) | 0.0018 | 1.6% | 0.0014 | +0.11 |
| 20 | x_sv_downside_dev_sic (`x_sv_downside_dev_sic`) | 0.0018 | 1.6% | 0.0013 | +0.03 |
| 21 | x_sv_vol_12w_s11 (`x_sv_vol_12w_s11`) | 0.0018 | 1.5% | 0.0015 | +0.23 |
| 22 | distance from 52-week high (`f_hi_52w`) | 0.0017 | 1.5% | 0.0009 | +0.04 |
| 23 | month (`f_month`) | 0.0016 | 1.4% | 0.0014 | +nan |
| 24 | 52-week return (`f_mom_52w`) | 0.0015 | 1.3% | 0.0007 | -0.12 |
| 25 | x_hl_pos_day_4w (`x_hl_pos_day_4w`) | 0.0015 | 1.3% | 0.0013 | -0.48 |
| 26 | debt / equity (`f_sf_de`) | 0.0014 | 1.2% | 0.0011 | +0.27 |
| 27 | 1-week return (`f_mom_1w`) | 0.0014 | 1.2% | 0.0010 | -0.59 |
| 28 | earnings yield (`f_sf_earnings_yield`) | 0.0014 | 1.2% | 0.0013 | +0.26 |
| 29 | market volatility (4w) (`f_mkt_vol_4w`) | 0.0014 | 1.2% | 0.0020 | +nan |
| 30 | insider buyers (13w) (`f_sfi_buyers_13w`) | 0.0014 | 1.2% | 0.0009 | +0.41 |
| 31 | net income growth (`f_sf_netinc_yoy`) | 0.0013 | 1.2% | 0.0008 | +0.17 |
| 32 | idiosyncratic volatility (60d) (`f_idio_vol_60d`) | 0.0013 | 1.1% | 0.0008 | -0.07 |
| 33 | revenue growth (`f_sf_revenue_yoy`) | 0.0013 | 1.1% | 0.0013 | +0.17 |
| 34 | fed funds change (`f_macro_FEDFUNDS_chg`) | 0.0013 | 1.1% | 0.0011 | +nan |
| 35 | x_cm_dip_1w (`x_cm_dip_1w`) | 0.0012 | 1.1% | 0.0011 | -0.68 |
| 36 | cross-sectional dispersion (`f_mkt_dispersion`) | 0.0012 | 1.1% | 0.0018 | +nan |
| 37 | x_cm_dip_4w (`x_cm_dip_4w`) | 0.0011 | 1.0% | 0.0009 | -0.52 |
| 38 | share issuance (`f_sf_issuance`) | 0.0011 | 1.0% | 0.0015 | -0.27 |
| 39 | debt / EBITDA (`f_sf_debt_ebitda`) | 0.0011 | 0.9% | 0.0007 | -0.34 |
| 40 | insider buyers, Form 4 (13w) (`f_insider_buyers_13w`) | 0.0011 | 0.9% | 0.0011 | -0.05 |
| 41 | x_vx_vol_4w_x_size (`x_vx_vol_4w_x_size`) | 0.0010 | 0.9% | 0.0018 | -0.13 |
| 42 | 12-week volatility (`f_vol_12w`) | 0.0010 | 0.9% | 0.0007 | +0.15 |
| 43 | x_sv_vol_12w_sic (`x_sv_vol_12w_sic`) | 0.0010 | 0.8% | 0.0007 | +0.03 |
| 44 | yield curve (10y-2y) (`f_macro_T10Y2Y`) | 0.0009 | 0.8% | 0.0012 | +nan |
| 45 | operating cash flow / assets (`f_ocf_to_assets`) | 0.0009 | 0.8% | 0.0017 | +0.29 |
| 46 | yield curve change (`f_macro_T10Y2Y_chg`) | 0.0009 | 0.7% | 0.0008 | +nan |
| 47 | market 26-week return (`f_mkt_mom_26w`) | 0.0008 | 0.7% | 0.0006 | +nan |
| 48 | 26-week return (`f_mom_26w`) | 0.0008 | 0.7% | 0.0004 | +0.13 |
| 49 | week of quarter (`f_woq`) | 0.0008 | 0.7% | 0.0010 | +nan |
| 50 | post-earnings drift (`f_pead`) | 0.0008 | 0.7% | 0.0013 | +0.34 |
| 51 | EBITDA margin (`f_sf_ebitda_margin`) | 0.0008 | 0.7% | 0.0005 | +0.03 |
| 52 | x_vx_vol_12w_x_size (`x_vx_vol_12w_x_size`) | 0.0008 | 0.7% | 0.0006 | +0.36 |
| 53 | overnight return share (4w) (`f_overnight_4w`) | 0.0008 | 0.7% | 0.0006 | -0.22 |
| 54 | market 4-week return (`f_mkt_mom_4w`) | 0.0007 | 0.6% | 0.0005 | +nan |
| 55 | 4-week return (`f_mom_4w`) | 0.0007 | 0.6% | 0.0006 | -0.20 |
| 56 | x_sv_vol_4w_sic (`x_sv_vol_4w_sic`) | 0.0006 | 0.6% | 0.0005 | +0.01 |
| 57 | current ratio (`f_sf_current_ratio`) | 0.0005 | 0.5% | 0.0005 | +0.21 |
| 58 | x_hl_range_20d (`x_hl_range_20d`) | 0.0005 | 0.5% | 0.0004 | -0.31 |
| 59 | return on equity (`f_sf_roe`) | 0.0004 | 0.4% | 0.0004 | +0.24 |
| 60 | x_sv_downside_dev_s11 (`x_sv_downside_dev_s11`) | 0.0004 | 0.3% | 0.0002 | +0.11 |
| 61 | 4-week volatility (`f_vol_4w`) | 0.0004 | 0.3% | 0.0004 | +0.14 |
| 62 | days since earnings (`f_days_since_earnings_8k`) | 0.0004 | 0.3% | 0.0004 | -0.04 |
| 63 | x_vx_downside_dev_x_size (`x_vx_downside_dev_x_size`) | 0.0004 | 0.3% | 0.0004 | +0.15 |
| 64 | earnings yield (EDGAR) (`f_earnings_yield`) | 0.0004 | 0.3% | 0.0005 | -0.19 |
| 65 | x_vx_idio_vol_60d_x_size (`x_vx_idio_vol_60d_x_size`) | 0.0003 | 0.3% | 0.0006 | +0.12 |
| 66 | intraday return share (4w) (`f_intraday_4w`) | 0.0003 | 0.3% | 0.0002 | -0.03 |
| 67 | x_sv_idio_vol_60d_s11 (`x_sv_idio_vol_60d_s11`) | 0.0002 | 0.2% | 0.0001 | +0.05 |
| 68 | asset growth (`f_asset_growth`) | 0.0002 | 0.2% | 0.0004 | -0.26 |
| 69 | x_hl_intraday_share (`x_hl_intraday_share`) | 0.0002 | 0.2% | 0.0003 | -0.03 |
| 70 | x_sv_vol_4w_s11 (`x_sv_vol_4w_s11`) | 0.0002 | 0.2% | 0.0002 | +0.13 |
| 71 | leverage (`f_leverage`) | 0.0002 | 0.1% | 0.0003 | -0.18 |
| 72 | book / market (EDGAR) (`f_book_to_market`) | 0.0001 | 0.1% | 0.0002 | -0.13 |
| 73 | sales / price (EDGAR) (`f_sales_to_price`) | 0.0001 | 0.1% | 0.0001 | +0.33 |
| 74 | abnormal volume (`f_abn_volume`) | 0.0001 | 0.1% | 0.0001 | +0.02 |
| 75 | ROE (EDGAR) (`f_roe`) | 0.0001 | 0.1% | 0.0001 | -0.26 |
| 76 | f_resid_ret_lag4w (`f_resid_ret_lag4w`) | 0.0001 | 0.1% | 0.0001 | +0.12 |
| 77 | gross profitability (EDGAR) (`f_gross_profitability`) | 0.0001 | 0.0% | 0.0001 | -0.22 |
| 78 | days since filing (`f_days_since_filing`) | 0.0000 | 0.0% | 0.0001 | +0.11 |
| 79 | cash flow / price (EDGAR) (`f_cf_to_price`) | 0.0000 | 0.0% | 0.0000 | +0.35 |
| 80 | insider buy in the last 2w (flag) (`f_evt_insider_buy_2w`) | 0.0000 | 0.0% | 0.0000 | +nan |
| 81 | 8-K in the last week (flag) (`f_evt_8k_7d`) | 0.0000 | 0.0% | 0.0000 | +nan |
| 82 | short interest ratio (`f_short_ratio`) | 0.0000 | 0.0% | 0.0000 | +nan |
| 83 | short interest, days to cover (`f_short_dtc`) | 0.0000 | 0.0% | 0.0000 | +nan |
| 84 | filing in the last 5 days (flag) (`f_evt_filed_5d`) | 0.0000 | 0.0% | 0.0000 | +nan |
| 85 | earnings 8-K in the last week (flag) (`f_evt_earnings_8k_7d`) | 0.0000 | 0.0% | 0.0000 | +nan |

## The top-3 picks and what drove them

Each week's three highest-scored members; the three largest contributions to each score (score units; positive pushed the name up).

| week | rank | ticker | score | biggest drivers |
|---|---|---|---|---|
| 2007-01-05 | 1 | MRO | +0.063 | dollar volume +0.053; sales / price +0.040; free-cash-flow yield +0.040 |
| 2007-01-05 | 2 | F | +0.060 | dollar volume +0.056; free-cash-flow yield +0.032; month -0.015 |
| 2007-01-05 | 3 | NUE | +0.051 | free-cash-flow yield +0.045; sales / price +0.041; x_hl_park_vol_12w +0.030 |
| 2008-01-04 | 1 | ETFC | +0.104 | sales / price +0.112; free-cash-flow yield +0.093; 26-week return +0.084 |
| 2008-01-04 | 2 | MBI | +0.092 | free-cash-flow yield +0.098; sales / price +0.090; distance from 52-week high +0.058 |
| 2008-01-04 | 3 | PHM | +0.076 | sales / price +0.084; free-cash-flow yield +0.074; distance from 52-week high +0.056 |
| 2009-01-02 | 1 | GOOGL | +0.075 | dollar volume +0.169; sales / price -0.043; beta (60d) +0.010 |
| 2009-01-02 | 2 | C | +0.071 | sales / price +0.047; free-cash-flow yield +0.028; market volatility (4w) +0.027 |
| 2009-01-02 | 3 | ACAS | +0.071 | free-cash-flow yield +0.052; sales / price +0.050; distance from 52-week high +0.029 |
| 2010-01-08 | 1 | TMUS | +0.111 | 26-week return +0.071; sales / price +0.037; 52-week return +0.036 |
| 2010-01-08 | 2 | S2 | +0.064 | x_sv_vol_12w_s11 +0.039; sales / price +0.038; free-cash-flow yield +0.028 |
| 2010-01-08 | 3 | FDO | +0.062 | x_sv_vol_12w_s11 +0.035; sales / price +0.032; free-cash-flow yield +0.025 |
| 2011-01-07 | 1 | SVU | +0.109 | x_sv_vol_12w_s11 +0.078; 12-week return +0.075; return on equity +0.031 |
| 2011-01-07 | 2 | DFODQ | +0.109 | x_sv_vol_12w_s11 +0.074; 12-week return +0.026; sales / price +0.024 |
| 2011-01-07 | 3 | S2 | +0.099 | x_sv_vol_12w_s11 +0.064; return on equity +0.023; sales / price +0.021 |
| 2012-01-06 | 1 | VLO | +0.004 | sales / price +0.020; x_cm_corr +0.011; beta (60d) +0.006 |
| 2012-01-06 | 2 | HWM | +0.003 | sales / price +0.017; x_cm_corr +0.011; fed funds rate +0.007 |
| 2012-01-06 | 3 | ROK | +0.003 | x_cm_corr +0.011; beta (60d) +0.006; distance from 52-week low +0.004 |
| 2013-01-04 | 1 | APOL | +0.070 | 52-week return +0.066; 12-week return +0.043; distance from 52-week high +0.042 |
| 2013-01-04 | 2 | EXPE | +0.054 | post-earnings drift +0.055; x_sv_vol_12w_s11 +0.023; x_sv_idio_vol_60d_sic -0.011 |
| 2013-01-04 | 3 | TSN | +0.048 | post-earnings drift +0.033; x_sv_vol_12w_s11 +0.021; sales / price +0.011 |
| 2014-01-03 | 1 | FSLR | +0.233 | post-earnings drift +0.224; market cap +0.032; x_vx_vol_12w_x_size +0.024 |
| 2014-01-03 | 2 | EXPE | +0.139 | post-earnings drift +0.169; 12-week volatility +0.025; x_sv_vol_12w_sic +0.025 |
| 2014-01-03 | 3 | CMG | +0.124 | post-earnings drift +0.140; dollar volume +0.034; operating cash flow / assets +0.028 |
| 2015-01-02 | 1 | NEM | +0.066 | earnings yield (EDGAR) +0.077; debt / EBITDA -0.019; leverage +0.015 |
| 2015-01-02 | 2 | NRG | +0.055 | post-earnings drift +0.070; earnings yield (EDGAR) +0.048; x_sv_vol_12w_s11 +0.015 |
| 2015-01-02 | 3 | MA | +0.035 | post-earnings drift +0.048; beta (60d) +0.007; x_sv_vol_12w_s11 +0.006 |

Mean signed contribution over all picks, largest first: post-earnings drift +0.0272, sales / price +0.0238, free-cash-flow yield +0.0208, x_sv_vol_12w_s11 +0.0157, dollar volume +0.0110, distance from 52-week high +0.0103, 12-week return +0.0100, 26-week return +0.0092, 52-week return +0.0058, earnings yield (EDGAR) +0.0047, x_hl_park_vol_12w +0.0041, downside deviation -0.0035
