# champion: the one look

Walk `data/experiments/labels/rank_k16` (sector-relative 4w rank label / 8y / top-3 / halfgate / cap 2) at K=16, settings top-3 / halfgate / stop None / cap 2 decided by `stocks-ml procedure`'s code on its 2006-2015 segment. $100 at each span's start; pre-tax (Roth); pre-holdout only. Generated 2026-09-14 19:13:42 by `stocks-ml eval`.

| model | 2006-2024: $100, %/yr | SR, DD | 2016-2024: $100, %/yr | SR, DD | 2006-2015: $100, %/yr | SR, DD | weekly t vs sp500 (06-24 / 16-24) |
|---|---|---|---|---|---|---|---|
| champion | $4,940, +23.4% | 0.82, 49% | $844, +28.2% | 0.92, 38% | $585, +19.3% | 0.73, 49% | +2.53 / +1.71 |
| incumbent (ls_w8, top-10 / halfgate / stop None / cap None) | $1,994, +17.5% | 0.70, 63% | $660, +24.6% | 0.88, 33% | $302, +11.7% | 0.53, 63% | +1.90 / +1.52 |
| sp500 | $621, +10.3% | 0.64, 55% | $316, +14.3% | 0.87, 32% | $197, +7.0% | 0.46, 55% | — |

Falsification (paired weekly excess vs the incumbent on 2016-2024, t < -2.0 rejects): 2016-2024 t +0.76 on 446 weeks -> **not rejected**. Edge over the incumbent, %/yr: 2006-2015 +7.63 (selection window), 2016-2024 +3.63 (the one look).

Leak audit: PASS — select: identity PASS, score-vs-split-factor -0.049 (t -6.8), IC +0.0063 (t +0.8), retention 0.912; extend: identity PASS, score-vs-split-factor -0.069 (t -11.0), IC +0.0466 (t +6.0), retention 1.015.

| window | metric | point | seed-only 95% | history-only 95% | nested 95% |
|---|---|---|---|---|---|
| 2016-2024 | excess CAGR vs sp500 %/yr | +12.2 | +7.4 … +16.7 | -4.3 … +32.4 | -4.8 … +33.1 |
| 2016-2024 | CAGR %/yr | +28.4 | +22.9 … +33.5 | +4.3 … +57.8 | +3.8 … +58.7 |
| 2016-2024 | terminal $100 | $844 | $581 … $1,181 | $144 … $4,931 | $138 … $5,177 |
| 2016-2024 | P(excess > 0), nested | 0.92 | | | |
| 2006-2024 | excess CAGR vs sp500 %/yr | +11.9 | +6.8 … +13.2 | +0.6 … +24.8 | -1.4 … +22.8 |
| 2006-2024 | CAGR %/yr | +23.4 | +17.9 … +24.9 | +6.9 … +42.9 | +5.7 … +40.3 |
| 2006-2024 | terminal $100 | $4,940 | $2,117 … $6,108 | $346 … $74,460 | $277 … $52,578 |
| 2006-2024 | P(excess > 0), nested | 0.95 | | | |
| 2006-2015 | excess CAGR vs sp500 %/yr | +11.6 | +3.3 … +12.5 | -3.0 … +28.5 | -5.7 … +25.0 |
| 2006-2015 | CAGR %/yr | +19.4 | +10.6 … +20.4 | -0.8 … +43.9 | -3.9 … +40.8 |
| 2006-2015 | terminal $100 | $585 | $272 … $636 | $92 … $3,765 | $67 … $3,019 |
| 2006-2015 | P(excess > 0), nested | 0.86 | | | |

Seed noise: the K copies resampled with replacement and re-simulated (200 draws). History noise: a circular block bootstrap of the weeks, 8-week blocks, strategy and S&P on the same weeks (4000 draws). Nested: both.

![champion vs sp500](champion_vs_sp500_2016_2024.png)
