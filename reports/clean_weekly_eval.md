# clean_weekly: the one look

Walk `data/experiments/clean_weekly` (sector-relative 4w rank label / 8y / top-3 / halfgate / cap 2) at K=16, settings top-3 / halfgate / stop None / cap 2 decided by `stocks-ml procedure`'s code on its 2006-2015 segment. $100 at each span's start; pre-tax (Roth); pre-holdout only. Generated 2026-09-21 12:37:52 by `stocks-ml eval`.

| model | 2006-2024: $100, %/yr | SR, DD | 2016-2024: $100, %/yr | SR, DD | 2006-2015: $100, %/yr | SR, DD | weekly t vs sp500 (06-24 / 16-24) |
|---|---|---|---|---|---|---|---|
| clean_weekly | $1,290, +14.8% | 0.58, 54% | $219, +9.6% | 0.43, 49% | $588, +19.4% | 0.71, 54% | +1.28 / -0.02 |
| sp500 | $621, +10.3% | 0.64, 55% | $316, +14.3% | 0.87, 32% | $197, +7.0% | 0.46, 55% | — |

Leak audit: PASS — select: identity PASS, score-vs-split-factor +0.039 (t +5.6), IC +0.0029 (t +0.3), retention 1.547; extend: identity PASS, score-vs-split-factor +0.055 (t +7.6), IC +0.0135 (t +1.2), retention 0.993; feature scan: 6 of 50 beyond 0.15 (f_sf_debt_ebitda +0.21, f_sf_book_to_market +0.21, f_sf_roe -0.16, f_sf_gross_prof -0.16, f_sf_sales_to_price +0.16, f_sfi_buyers_13w +0.15); missingness scan: 0 of 50 whose blanks predict.

| window | metric | point | seed-only 95% | history-only 95% | nested 95% |
|---|---|---|---|---|---|
| 2016-2024 | excess CAGR vs sp500 %/yr | -4.2 | -9.4 … +0.7 | -18.1 … +13.5 | -18.1 … +11.3 |
| 2016-2024 | CAGR %/yr | +9.6 | +3.6 … +15.2 | -11.4 … +36.0 | -10.8 … +34.1 |
| 2016-2024 | terminal $100 | $219 | $136 … $336 | $36 … $1,388 | $38 … $1,229 |
| 2016-2024 | P(excess > 0), nested | 0.28 | | | |
| 2006-2024 | excess CAGR vs sp500 %/yr | +4.0 | -0.8 … +6.3 | -6.8 … +16.7 | -7.9 … +14.9 |
| 2006-2024 | CAGR %/yr | +14.8 | +9.4 … +17.4 | -1.0 … +34.5 | -2.4 … +31.8 |
| 2006-2024 | terminal $100 | $1,290 | $530 … $1,935 | $82 … $24,013 | $64 … $16,660 |
| 2006-2024 | P(excess > 0), nested | 0.68 | | | |
| 2006-2015 | excess CAGR vs sp500 %/yr | +11.6 | +5.3 … +14.3 | -3.8 … +30.5 | -5.2 … +28.7 |
| 2006-2015 | CAGR %/yr | +19.5 | +12.7 … +22.4 | -3.0 … +47.7 | -4.1 … +46.0 |
| 2006-2015 | terminal $100 | $588 | $328 … $747 | $74 … $4,871 | $66 … $4,342 |
| 2006-2015 | P(excess > 0), nested | 0.90 | | | |

Seed noise: the K copies resampled with replacement and re-simulated (200 draws). History noise: a circular block bootstrap of the weeks, 8-week blocks, strategy and S&P on the same weeks (4000 draws). Nested: both.

![clean_weekly vs sp500](clean_weekly_vs_sp500_2016_2024.png)
