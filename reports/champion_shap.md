# Champion feature importance (Shapley values)

Recipe {'label': 'label_4w_sector_rank', 'train_years': 8, 'features': [], 'params': {'max_depth': 3, 'learning_rate': 0.02, 'n_estimators': 1500, 'min_child_weight': 20, 'subsample': 0.85, 'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 1.0, 'gamma': 0.01, 'max_bin': 256, 'tree_method': 'hist'}}; one yearly fit at each year's first rank week (2007-01-05 → 2024-01-05), exact TreeSHAP on every member scored that week. mean |SHAP| in score units; share = of the total across features; sign = rank correlation between the feature's value and its contribution (positive: higher raises the score).

| rank | feature | mean abs SHAP | share | sd across years | sign |
|---|---|---|---|---|---|
| 1 | sales / price (`f_sf_sales_to_price`) | 0.0099 | 10.0% | 0.0112 | +0.46 |
| 2 | market cap (`f_log_mktcap`) | 0.0081 | 8.2% | 0.0069 | -0.67 |
| 3 | free-cash-flow yield (`f_sf_fcf_yield`) | 0.0073 | 7.4% | 0.0101 | +0.60 |
| 4 | dollar volume (`f_dollar_vol`) | 0.0046 | 4.6% | 0.0029 | +0.50 |
| 5 | 4-week return (`f_mom_4w`) | 0.0037 | 3.7% | 0.0067 | -0.60 |
| 6 | 1-week return (`f_mom_1w`) | 0.0030 | 3.0% | 0.0041 | -0.78 |
| 7 | book / market (`f_sf_book_to_market`) | 0.0025 | 2.5% | 0.0039 | +0.51 |
| 8 | gross profitability (`f_sf_gross_prof`) | 0.0025 | 2.5% | 0.0023 | +0.05 |
| 9 | revenue growth (`f_sf_revenue_yoy`) | 0.0024 | 2.4% | 0.0032 | +0.19 |
| 10 | downside deviation (`f_downside_dev`) | 0.0022 | 2.3% | 0.0023 | +0.38 |
| 11 | asset growth (`f_asset_growth`) | 0.0021 | 2.1% | 0.0028 | -0.40 |
| 12 | 12-week return (`f_mom_12w`) | 0.0021 | 2.1% | 0.0052 | -0.39 |
| 13 | fed funds rate (`f_macro_FEDFUNDS`) | 0.0021 | 2.1% | 0.0022 | +nan |
| 14 | distance from 52-week low (`f_lo_52w`) | 0.0019 | 2.0% | 0.0033 | +0.26 |
| 15 | insider net buying (13w) (`f_sfi_net_13w`) | 0.0019 | 2.0% | 0.0030 | +0.23 |
| 16 | fed funds change (`f_macro_FEDFUNDS_chg`) | 0.0019 | 1.9% | 0.0028 | +nan |
| 17 | beta (60d) (`f_beta_60d`) | 0.0018 | 1.8% | 0.0015 | +0.22 |
| 18 | post-earnings drift (`f_pead`) | 0.0017 | 1.7% | 0.0022 | +0.09 |
| 19 | 12-week volatility (`f_vol_12w`) | 0.0017 | 1.7% | 0.0018 | +0.18 |
| 20 | yield curve (10y-2y) (`f_macro_T10Y2Y`) | 0.0017 | 1.7% | 0.0020 | +nan |
| 21 | negative EBITDA (flag) (`f_sf_neg_ebitda`) | 0.0016 | 1.7% | 0.0020 | +nan |
| 22 | distance from 52-week high (`f_hi_52w`) | 0.0016 | 1.6% | 0.0015 | -0.12 |
| 23 | idiosyncratic volatility (60d) (`f_idio_vol_60d`) | 0.0015 | 1.5% | 0.0012 | +0.03 |
| 24 | overnight return share (4w) (`f_overnight_4w`) | 0.0015 | 1.5% | 0.0018 | -0.37 |
| 25 | earnings yield (`f_sf_earnings_yield`) | 0.0014 | 1.4% | 0.0014 | +0.01 |
| 26 | net income growth (`f_sf_netinc_yoy`) | 0.0014 | 1.4% | 0.0017 | -0.01 |
| 27 | debt / equity (`f_sf_de`) | 0.0014 | 1.4% | 0.0014 | +0.47 |
| 28 | 52-week return (`f_mom_52w`) | 0.0014 | 1.4% | 0.0011 | -0.03 |
| 29 | market volatility (4w) (`f_mkt_vol_4w`) | 0.0013 | 1.3% | 0.0013 | +nan |
| 30 | intraday return share (4w) (`f_intraday_4w`) | 0.0013 | 1.3% | 0.0023 | -0.33 |
| 31 | share issuance (`f_sf_issuance`) | 0.0012 | 1.2% | 0.0015 | -0.13 |
| 32 | book / market (EDGAR) (`f_book_to_market`) | 0.0011 | 1.1% | 0.0020 | -0.35 |
| 33 | insider buyers, Form 4 (13w) (`f_insider_buyers_13w`) | 0.0011 | 1.1% | 0.0014 | +0.02 |
| 34 | gross profitability (EDGAR) (`f_gross_profitability`) | 0.0010 | 1.1% | 0.0026 | +0.01 |
| 35 | week of quarter (`f_woq`) | 0.0010 | 1.0% | 0.0035 | +nan |
| 36 | insider buyers (13w) (`f_sfi_buyers_13w`) | 0.0010 | 1.0% | 0.0014 | +0.24 |
| 37 | debt / EBITDA (`f_sf_debt_ebitda`) | 0.0010 | 1.0% | 0.0011 | -0.03 |
| 38 | month (`f_month`) | 0.0010 | 1.0% | 0.0013 | +nan |
| 39 | return on equity (`f_sf_roe`) | 0.0010 | 1.0% | 0.0015 | +0.11 |
| 40 | earnings yield (EDGAR) (`f_earnings_yield`) | 0.0009 | 0.9% | 0.0017 | -0.21 |
| 41 | insider net buying, Form 4 (13w) (`f_insider_net_13w`) | 0.0008 | 0.8% | 0.0012 | +0.13 |
| 42 | EBITDA margin (`f_sf_ebitda_margin`) | 0.0008 | 0.8% | 0.0013 | +0.10 |
| 43 | market 4-week return (`f_mkt_mom_4w`) | 0.0008 | 0.8% | 0.0010 | +nan |
| 44 | current ratio (`f_sf_current_ratio`) | 0.0007 | 0.7% | 0.0008 | +0.30 |
| 45 | 26-week return (`f_mom_26w`) | 0.0007 | 0.7% | 0.0011 | +0.13 |
| 46 | market 26-week return (`f_mkt_mom_26w`) | 0.0006 | 0.7% | 0.0010 | +nan |
| 47 | yield curve change (`f_macro_T10Y2Y_chg`) | 0.0006 | 0.6% | 0.0006 | +nan |
| 48 | cross-sectional dispersion (`f_mkt_dispersion`) | 0.0006 | 0.6% | 0.0009 | +nan |
| 49 | leverage (`f_leverage`) | 0.0006 | 0.6% | 0.0015 | +0.09 |
| 50 | sales / price (EDGAR) (`f_sales_to_price`) | 0.0005 | 0.5% | 0.0006 | +0.16 |
| 51 | 4-week volatility (`f_vol_4w`) | 0.0005 | 0.5% | 0.0007 | +0.08 |
| 52 | operating cash flow / assets (`f_ocf_to_assets`) | 0.0004 | 0.4% | 0.0006 | +0.39 |
| 53 | ROE (EDGAR) (`f_roe`) | 0.0004 | 0.4% | 0.0006 | +0.03 |
| 54 | cash flow / price (EDGAR) (`f_cf_to_price`) | 0.0004 | 0.4% | 0.0014 | +0.07 |
| 55 | days since earnings (`f_days_since_earnings_8k`) | 0.0003 | 0.3% | 0.0005 | +0.12 |
| 56 | days since filing (`f_days_since_filing`) | 0.0002 | 0.2% | 0.0004 | +0.48 |
| 57 | abnormal volume (`f_abn_volume`) | 0.0002 | 0.2% | 0.0002 | -0.12 |
| 58 | f_resid_ret_lag4w (`f_resid_ret_lag4w`) | 0.0001 | 0.1% | 0.0002 | +0.02 |
| 59 | short interest, days to cover (`f_short_dtc`) | 0.0001 | 0.1% | 0.0003 | -0.47 |
| 60 | short interest ratio (`f_short_ratio`) | 0.0000 | 0.0% | 0.0001 | -0.16 |
| 61 | insider buy in the last 2w (flag) (`f_evt_insider_buy_2w`) | 0.0000 | 0.0% | 0.0000 | +nan |
| 62 | 8-K in the last week (flag) (`f_evt_8k_7d`) | 0.0000 | 0.0% | 0.0000 | +nan |
| 63 | earnings 8-K in the last week (flag) (`f_evt_earnings_8k_7d`) | 0.0000 | 0.0% | 0.0000 | +nan |
| 64 | filing in the last 5 days (flag) (`f_evt_filed_5d`) | 0.0000 | 0.0% | 0.0000 | +nan |

## The top-3 picks and what drove them

Each week's three highest-scored members; the three largest contributions to each score (score units; positive pushed the name up).

| week | rank | ticker | score | biggest drivers |
|---|---|---|---|---|
| 2007-01-05 | 1 | Q1 | +0.055 | free-cash-flow yield +0.052; debt / equity +0.052; sales / price +0.029 |
| 2007-01-05 | 2 | S2 | +0.032 | sales / price +0.038; free-cash-flow yield +0.027; overnight return share (4w) +0.013 |
| 2007-01-05 | 3 | HAL | +0.028 | free-cash-flow yield +0.033; sales / price +0.029; month -0.024 |
| 2008-01-04 | 1 | PHM | +0.023 | sales / price +0.103; free-cash-flow yield +0.094; distance from 52-week high +0.044 |
| 2008-01-04 | 2 | FHN | +0.023 | sales / price +0.077; free-cash-flow yield +0.072; 12-week volatility +0.023 |
| 2008-01-04 | 3 | BIGGQ | +0.023 | sales / price +0.092; free-cash-flow yield +0.080; 12-week volatility +0.024 |
| 2009-01-02 | 1 | GOOGL | +0.373 | dollar volume +0.374; sales / price -0.048; 1-week return +0.026 |
| 2009-01-02 | 2 | X | +0.160 | sales / price +0.066; free-cash-flow yield +0.053; distance from 52-week low +0.045 |
| 2009-01-02 | 3 | LNC | +0.129 | sales / price +0.087; free-cash-flow yield +0.055; distance from 52-week low +0.033 |
| 2010-01-08 | 1 | TJX | +0.003 | sales / price +0.031; dollar volume +0.021; fed funds rate +0.009 |
| 2010-01-08 | 2 | XOM | +0.003 | distance from 52-week high -0.028; sales / price +0.026; dollar volume +0.022 |
| 2010-01-08 | 3 | F | +0.003 | sales / price +0.038; dollar volume +0.022; free-cash-flow yield +0.018 |
| 2011-01-07 | 1 | SVU | +0.370 | 12-week return +0.163; return on equity +0.144; 26-week return +0.056 |
| 2011-01-07 | 2 | ANDV | +0.174 | return on equity +0.065; sales / price +0.032; earnings yield +0.024 |
| 2011-01-07 | 3 | SWY | +0.163 | return on equity +0.077; 12-week return +0.033; sales / price +0.028 |
| 2012-01-06 | 1 | GOOGL | +0.007 | dollar volume +0.099; current ratio -0.032; sales / price -0.020 |
| 2012-01-06 | 2 | AAPL | +0.007 | dollar volume +0.187; free-cash-flow yield +0.037; market cap -0.032 |
| 2012-01-06 | 3 | PSKY | +0.001 | distance from 52-week low +0.013; sales / price +0.012; fed funds rate +0.005 |
| 2013-01-04 | 1 | AAPL | +0.017 | dollar volume +0.157; cross-sectional dispersion +0.034; operating cash flow / assets +0.026 |
| 2013-01-04 | 2 | GOOGL | +0.013 | dollar volume +0.156; cross-sectional dispersion +0.029; sales / price -0.008 |
| 2013-01-04 | 3 | KLAC | +0.007 | operating cash flow / assets +0.033; market cap +0.011; yield curve change -0.011 |
| 2014-01-03 | 1 | FSLR | +0.040 | post-earnings drift +0.112; intraday return share (4w) -0.050; 12-week volatility +0.029 |
| 2014-01-03 | 2 | EXPE | +0.037 | post-earnings drift +0.131; 12-week volatility +0.034; idiosyncratic volatility (60d) +0.018 |
| 2014-01-03 | 3 | PHM | +0.025 | post-earnings drift +0.049; idiosyncratic volatility (60d) -0.006; share issuance +0.004 |
| 2015-01-02 | 1 | NRG | +0.109 | post-earnings drift +0.112; earnings yield (EDGAR) +0.107; earnings yield +0.018 |
| 2015-01-02 | 2 | MA | +0.067 | post-earnings drift +0.110; abnormal volume +0.008; 12-week volatility -0.006 |
| 2015-01-02 | 3 | EXPE | +0.066 | post-earnings drift +0.084; abnormal volume +0.007; net income growth -0.005 |
| 2016-01-08 | 1 | FFIV | +0.064 | market cap +0.202; gross profitability +0.080; gross profitability (EDGAR) +0.043 |
| 2016-01-08 | 2 | COR | +0.041 | sales / price +0.068; revenue growth +0.024; earnings yield (EDGAR) +0.024 |
| 2016-01-08 | 3 | ALLE | +0.039 | market cap +0.093; return on equity +0.041; free-cash-flow yield -0.007 |
| 2017-01-06 | 1 | NVDA | +0.215 | dollar volume +0.091; asset growth +0.087; 12-week volatility +0.068 |
| 2017-01-06 | 2 | NRG | +0.188 | market cap +0.118; earnings yield (EDGAR) +0.037; sales / price +0.029 |
| 2017-01-06 | 3 | BKNG | +0.121 | dollar volume +0.097; post-earnings drift +0.032; debt / equity +0.011 |
| 2018-01-05 | 1 | KR | +0.094 | sales / price +0.165; book / market -0.020; debt / equity +0.016 |
| 2018-01-05 | 2 | MCK | +0.072 | sales / price +0.134; free-cash-flow yield -0.019; debt / equity +0.016 |
| 2018-01-05 | 3 | NI | +0.072 | market cap +0.094; asset growth +0.038; yield curve (10y-2y) +0.008 |
| 2019-01-04 | 1 | URI | +0.123 | market cap +0.080; 12-week volatility +0.023; earnings yield +0.018 |
| 2019-01-04 | 2 | PCG | +0.114 | idiosyncratic volatility (60d) +0.126; asset growth +0.074; cash flow / price (EDGAR) +0.054 |
| 2019-01-04 | 3 | ORLY | +0.080 | dollar volume +0.065; ROE (EDGAR) +0.037; book / market (EDGAR) +0.013 |
| 2020-01-03 | 1 | DVA | +0.096 | market cap +0.095; idiosyncratic volatility (60d) +0.018; earnings yield (EDGAR) +0.017 |
| 2020-01-03 | 2 | DVN | +0.081 | market cap +0.092; asset growth +0.034; idiosyncratic volatility (60d) +0.023 |
| 2020-01-03 | 3 | NOW | +0.070 | earnings yield (EDGAR) +0.073; dollar volume +0.044; downside deviation +0.036 |
| 2021-01-08 | 1 | HST | +0.028 | market cap +0.073; 12-week volatility +0.050; debt / EBITDA +0.029 |
| 2021-01-08 | 2 | REG | +0.018 | market cap +0.057; 12-week volatility +0.027; 4-week return +0.011 |
| 2021-01-08 | 3 | OXY | +0.018 | market cap +0.043; distance from 52-week high +0.037; 12-week volatility +0.034 |
| 2022-01-07 | 1 | BIIB | +0.160 | gross profitability (EDGAR) +0.114; distance from 52-week high +0.075; net income growth +0.028 |
| 2022-01-07 | 2 | NCLH | +0.142 | market cap +0.085; sales / price -0.031; idiosyncratic volatility (60d) +0.026 |
| 2022-01-07 | 3 | FTNT | +0.124 | gross profitability (EDGAR) +0.081; dollar volume +0.060; distance from 52-week low +0.050 |
| 2023-01-06 | 1 | ALGN | +0.006 | 4-week return -0.035; market cap +0.031; gross profitability (EDGAR) +0.026 |
| 2023-01-06 | 2 | ON | +0.006 | 4-week return +0.039; market cap +0.026; 12-week volatility +0.015 |
| 2023-01-06 | 3 | GNRC | +0.006 | market cap +0.070; 4-week return -0.036; 12-week volatility +0.017 |
| 2024-01-05 | 1 | DG | +0.028 | 52-week return +0.024; market cap +0.019; sales / price +0.016 |
| 2024-01-05 | 2 | WBD | +0.027 | distance from 52-week high +0.026; cash flow / price (EDGAR) +0.025; market cap +0.023 |
| 2024-01-05 | 3 | UAL | +0.024 | market cap +0.048; sales / price +0.027; cash flow / price (EDGAR) +0.020 |

Mean signed contribution over all picks, largest first: dollar volume +0.0266, market cap +0.0224, sales / price +0.0197, post-earnings drift +0.0127, 12-week volatility +0.0089, free-cash-flow yield +0.0088, asset growth +0.0057, return on equity +0.0054, earnings yield (EDGAR) +0.0047, gross profitability (EDGAR) +0.0047, idiosyncratic volatility (60d) +0.0047, distance from 52-week high +0.0044
