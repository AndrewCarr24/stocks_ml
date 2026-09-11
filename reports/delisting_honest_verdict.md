# The delisting-honest world: verdict

Graded 2026-09-10 against the preregistration `nominal_clean_dl_2006_2024`
(trials ledger, written before any number; owner's go "Sure" to the
delisting-honest proposal). Clean nominal line, K=16 at the champion's standing
settings (top-6 / cap 2 / no stop / 70-30), next-open fills at 5 bp a side,
every window ending at `selection.HOLDOUT_START` (2024-07-19) exclusive.
Holdout untouched.

What changed (gated: `Config.delist_labels` / `STOCKS_ML_DELIST_LABELS`,
default `drop`, so the live job is untouched until adoption):

1. **Labels** — a name whose series ends inside the 4-week window grades to its
   final print (the live ledger's exit fallback) instead of NaN, enters the
   member-median recentring, and enters training (the walk refits weekly).
   1,119 member-weeks gained a label; the weekly medians moved 318,847
   existing labels by at most 27 bp.
2. **Universe** — `slice_row` uses live's traded-within-7-days rule
   (`r5.rank_members`) instead of label finiteness, so the backtest can rank
   and buy a name that will delist mid-hold, exactly as live can.

Store `data/sharadar_world2000_nominal_dl` (tables symlinked to the nominal
world, own panel), walk `data/experiments/nominal_clean_{2006_2015,2016_2024}_dl`
(967 rank weeks, 0 duplicates), graded by `ops/nominal_program.py grade` under
`STOCKS_ML_RUN_SUFFIX=_dl` → `reports/nominal_board_dl.md`,
`data/experiments/nominal_grades_dl.json`, ledger row `nominal_clean_dl_k16`.

## Leak audit — PASS

`ops/leak_audit.py`, per segment, gated on the worst:

| segment | Spearman(score, adj. factor) | IC | residual IC | retention | verdict |
|---|---|---|---|---|---|
| 2006-2015 | −0.010 (t −0.9) | −0.019 (t −1.7) | −0.019 | 1.005 | PASS |
| 2016-2024 | −0.010 (t −0.9) | +0.028 (t +2.6) | +0.029 | 1.038 | PASS |

Identity checks exact for the three largest in-window splitters (CMG ×50,
NVDA ×40, BKNG ×25). Delisting-within-8w: top-15 0.58% / 0.21% per week vs
universe 0.80% / 0.69%. The standing clean line reads the same on the same
segments (IC −0.016 / +0.028, retention 1.03 / 1.04): the near-zero
whole-universe IC on 2006-2015 is a property of the clean line in both worlds,
not of the delisting change.

## The board (same bar, same windows)

| K=16, champion settings | 2006-2015 | 2016-2024 | pre-holdout |
|---|---|---|---|
| clean, delisting-honest world | $229 (+8.6%/yr, SR 0.46, DD 59%) | $449 (+19.1%/yr, SR 0.80, DD 42%) | $1,028 (+13.4%/yr, SR 0.62, DD 59%) |
| clean, standing world | $262 (+9.9%/yr, SR 0.51, DD 60%) | $560 (+22.2%/yr, SR 0.90, DD 39%) | $1,468 (+15.6%/yr, SR 0.69, DD 60%) |
| sp500 | $197 (+7.0%/yr, SR 0.46, DD 55%) | $316 (+14.3%/yr, SR 0.87, DD 32%) | $621 (+10.3%/yr, SR 0.64, DD 55%) |

K=4 sub-ensemble spread (four per world, eight draws in all):

| window | standing world | delisting-honest world | sp500 |
|---|---|---|---|
| 2006-2015 | $161 – $270 | $181 – $278 | $197 |
| 2016-2024 | $481 – $706 | $484 – $676 | $316 |
| pre-holdout | $879 – $1,625 | $1,057 – $1,366 | $621 |

## Where the gap comes from — a same-basis cross-grade

Each model's saved K=16 scores, simulated under each world's rules
(`scratchpad/dl_crossgrade.py`, same code path as the grade):

| | survivor simulator (`drop`) | honest simulator (`last_print`) |
|---|---|---|
| standing model | $262 / $560 / $1,468 | $263 / $560 / $1,472 |
| honest model | $228 / $465 / $1,058 | $229 / $449 / $1,028 |

- **The simulator change is nearly free.** Letting the book buy names that then
  die, and booking their exits at the last print, moves the standing model by
  +$4 and the honest model by −$30 on the full span. The clean line's picks
  are not survivorship-flattered — the pre-stated expectation ("small net
  change, worse 2008-2009 drawdowns") holds for the mechanism that was being
  priced; max drawdown is unchanged (59% vs 60%).
- **The rest is the model refit, and it is inside the noise floor.** The two
  K=16 ensembles agree at Spearman 0.915 and share 4.35 of 6 top-6 picks per
  week — *more* than two seed-halves of the same walk agree with each other
  (0.72, 2.95 of 6; both worlds). The paired weekly return difference is
  +3.8 bp/week, t = 1.67 over 966 weeks (2016-2024: +4.8 bp, t 1.81); 73% of
  the log-wealth gap sits in 20 weeks, led by 2009-03-20 (+7.7%) and
  2008-10-17 (+5.8%).

Verdict on the preregistered question: **the delisting-honest world is the
faithful world and reads the same strategy.** A near-identical model
(labels differing on 0.2% of rows by ordinary amounts — the gained labels
average +0.2% with sd 7.8%, S&P 500 delistings being mostly cash-outs near
the last print — plus sub-27 bp median shifts) lands 30% lower in terminal
wealth over 18 years. That is the precision of a single 6-name path, and it
is the same lesson the K=4 spread teaches: terminal-wealth figures from this
book carry a band of roughly ±30%, and any comparison inside that band is not
a comparison.

## What it means for the open decisions

- The clean line's claim survives on its own terms: on 2016-2024 every one of
  the eight K=4 draws and both K=16 paths beat SPY ($449–$706 vs $316); on
  2006-2015 the draws straddle SPY ($161–$278 vs $197) — the selection-window
  edge is thin and was thin in the standing world too.
- Going forward the delisting-honest world should be *the* research world
  (train + backtest), because it is the one that matches live; its numbers
  are the ones to quote. Adopting `last_print` for the live job's own training
  is the owner's call (`config.yaml: delist_labels`), as is adopting the
  nominal basis and the clean line itself; nothing is pushed.
- The registered adoption gates are met: leak audit PASS per segment, graded
  once at the standing settings beside the incumbent, holdout untouched.
