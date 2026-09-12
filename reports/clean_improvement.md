# Improving the clean line — stages A to E, and the adoption

Generated 2026-09-11/12 by `ops/clean_program.py`; registered before any
number in reports/clean_improvement_registration.md. Everything below is on
the delisting-honest research world (`data/sharadar_world2000_nominal_dl`).
Stages A-D read 2006-2015 only; stage E read 2016-2024 once, for the frozen
package. The package was adopted as the champion on 2026-09-12 (last
section); until then nothing here changed the deployed spec.

## Stage A — strategy layers on the incumbent (K=16, 522 weeks of 2006-2015)

The cascade as `run_cascade` decides it, on the incumbent's saved K=16
predictions: **book 6 / floor 60-40 / no stop / cap 2**. Evidence: book by
cost-adjusted compounded %/yr 3 → 2.32, 6 → 3.15, 10 → 1.99; floor by
Sharpe at book 6: none 0.348, halfgate 0.400, 80-20 0.386, 70-30 0.410,
60-40 0.438; stop −25% 0.391 vs none 0.438; cap 2 0.487 vs none 0.438.

Book × floor grid (Sharpe over 2006-2015, no cap):

| book | none | halfgate | 80/20 | 70/30 | 60/40 |
|---|---|---|---|---|---|
| 3 | 0.353 | 0.434 | 0.388 | 0.413 | 0.437 |
| 6 | 0.348 | 0.400 | 0.386 | 0.410 | 0.438 |
| 10 | 0.302 | 0.375 | 0.347 | 0.370 | 0.400 |

With cap 2: deployed 6 / 70-30 → SR 0.459, $229, DD 59%; the cascade's
6 / 60-40 → SR 0.487, $229, DD 52%. Same terminal, seven points less
drawdown, +0.03 Sharpe: inside one draw's path noise. The layers are
re-decided on the frozen model in stage E; the 2016-2024 look grades them.

Postscript (2026-09-11, later the same day): the owner ruled that keeping
70-30 by hand at the clean-line switch was an override of the mechanical
rule, and that the cascade must be code that writes the spec. `stocks-ml
procedure` (src/stocks_ml/procedure.py) now runs `selection.decide_strategy`
on the K=16 walk and wrote **6 / 60-40 / no stop / cap 2** into
models/champion_spec.json; the live job reads its settings from there
(tests/test_procedure.py). The deployed floor is 60-40 from the 2026-09-12
signal on. Stage C's "deployed settings" column reads the spec's decision.

## Stage B — model sweep at sample scale (131 weeks = every 4th of 2006-2015, K=4)

Metric (ranking only): mean over sample weeks of the top-6 pick's 4-week
forward return minus the universe mean that week, for the K=4 ensemble of
copies 1-4; the incumbent's own copies 1-4 on the same weeks are the
baseline (paired). Weeks where a single copy scored fewer than 20 distinct
values are dropped for that copy (see the finding below).

| variant | K=4 mean excess %/4w | copies 1-4 | vs incumbent (pp) | paired t | win rate | collapsed fits | weeks |
|---|---|---|---|---|---|---|---|
| lab_sector | +0.444 | +0.19, +0.40, +0.20, −0.24 | +0.876 | +1.89 | 0.54 | 44% | 128 |
| win8 | +0.222 | +0.00, +2.34, +0.16, +0.43 | +0.701 | +1.02 | 0.49 | 45% | 128 |
| lr01 | +0.024 | +0.47, −0.28, −0.88, +0.12 | +0.510 | +1.52 | 0.47 | 48% | 128 |
| fixed300 | −0.018 | −0.30, −0.25, +0.16, +0.22 | +0.440 | +0.74 | 0.54 | 0% | 131 |
| mcw100 | −0.070 | +0.54, +0.54, −0.63, −1.16 | +0.416 | +1.13 | 0.48 | 43% | 128 |
| col50 | −0.141 | −0.34, +0.24, −0.63, +0.45 | +0.384 | +0.92 | 0.54 | 39% | 131 |
| lr05 | −0.204 | −0.31, +0.23, −0.11, −0.58 | +0.316 | +1.00 | 0.50 | 40% | 130 |
| mcw5 | −0.289 | +0.15, −0.22, −0.56, −1.28 | +0.197 | +0.89 | 0.39 | 43% | 128 |
| depth4 | −0.364 | −0.27, −0.48, −0.23, −1.03 | −0.004 | −0.01 | 0.50 | 36% | 131 |
| depth6 | −0.437 | +0.91, −0.45, −0.70, −0.59 | +0.031 | +0.06 | 0.55 | 24% | 131 |
| lab_gauss | −0.455 | −0.84, −0.46, −0.30, +0.38 | +0.020 | +0.04 | 0.50 | 29% | 131 |
| col30 | −0.471 | +0.02, +1.14, −0.09, −0.06 | +0.063 | +0.15 | 0.52 | 38% | 131 |
| **incumbent** | −0.486 | +0.93, −0.17, −0.68, −0.97 | 0 | — | — | 43% | 128 |
| depth2 | −0.620 | −0.41, −0.24, −0.61, −0.68 | +0.097 | +0.22 | 0.46 | 48% | 123 |
| win3 | −0.654 | −1.15, +0.30, −0.67, −0.45 | −0.207 | −0.34 | 0.53 | 34% | 131 |

Reading it: the incumbent's own four copies span −0.97 to +0.93 pp on the
same weeks, so a 1.9 pp copy spread is the noise floor and no variant's K=4
difference (max +0.88 pp, t 1.89) clears it. The sample ranks, as
registered; it decides nothing. The mechanical stage C picks are the four
highest: **lab_sector, win8, lr01, fixed300**.

## Finding: the early stop collapses ~40% of the copies

Every copy of the incumbent early-stops on the weekly Spearman of the last
10% of training weeks (~26 weeks, purged). Reading the fitted boosters at
eleven sample weeks: best_iteration was 679, 130, **0**, 111, **0**, 7, 25,
2, 556, 28, 31. A copy stopped at round 0-7 is one to a few depth-3 trees:
it scores ~490 names with fewer than 20 distinct values and contributes
almost nothing to the ensemble mean.

- Incumbent, 2006-2015: **43%** of the 16 × 522 single-copy fits are
  collapsed (per copy 38-49%); 2016-2024: 28%. By year: 2006 18%, 2007 48%,
  2008 42%, 2009 42%, 2010 45%, 2011 72%, 2012 52%, 2013 44%, 2014 36%,
  2015 35%, 2016 30%, 2017 36%, 2018 17%, 2019 10%, 2020 24%, 2021 32%,
  2022 48%, 2023 27%, 2024 25%.
- Effective copies per week (≥ 20 distinct scores): mean 10.2 of 16, min 0;
  29% of weeks have 8 or fewer real copies.
- Every early-stopped variant collapses at a similar rate (24-48%);
  `fixed300` (300 rounds, no early stop) never does and its four copies sit
  within 0.5 pp of each other (the incumbent's within 1.9 pp).
- The live job fits the same estimator, so each Saturday's K=16 carries the
  same share of near-constant copies. Not a leak, not a bug in the sense of
  wrong arithmetic: the stopping signal (26-week IC on a 6-name-scale panel)
  is too noisy to be trusted round by round, and "stop at the first tree"
  wins it often.

Path noise across copies is partly this: a copy's contribution is a coin
flip between a real model and a constant.

## Proposed stage C (needs a go)

Six candidates at K=4 on every week of 2006-2015 (522 weeks), frozen by the
registered selection metric (top-6 cost-adjusted compounded %/yr) against
the incumbent's copies 1-4, with the K=4 spread and the paired weekly t
reported, and the leak audit on each candidate walk:

1. lab_sector (registered pick)
2. win8 (registered pick)
3. lr01 (registered pick)
4. fixed300 (registered pick; also the collapse repair)
5. lab_sector + win8 (combination of the top two)
6. lab_sector + win8 + fixed300 (top two on the repaired stop)

~4 h (fixed300 ~30 min, win8 ~50 min, the rest ~35 min each).

## Stage C — frozen model selection (every week of 2006-2015, K=4) — run 2026-09-11 on the owner's go

Metric (frozen decision): the cascade's book metric, cost-adjusted
compounded %/yr of the top-6 book held 4 weeks, for the K=4 ensemble of
copies 1-4, on the 488 weeks every walk ranked with a real ensemble (each
walk ranked all 522 rank weeks; a week whose K=4 mean scores fewer than 20
distinct values — all four copies collapsed — is dropped, 6-11 weeks per
early-stopped walk, none for the fixed-round walks; the common set is the
intersection). Paired t: weekly top-6 excess over the universe mean vs the
incumbent's copies 1-4 on the same weeks. Deployed-settings column: the
spec's decision (6 / 60-40 / cap 2 / no stop), $100 from 2006-01, for
information only.

| candidate | top-6 compounded %/yr | copies 1-4 | vs incumbent (pp/4w) | paired t | 6/60-40/cap2: $100, SR, DD | leak audit | collapsed fits |
|---|---|---|---|---|---|---|---|
| **ls_w8** (sector label + 8y window) | **+7.57** | +15.80, +8.99, +5.56, +10.25 | +0.421 | +0.95 | $365, 0.59, 61% | PASS (0.89) | 44% |
| lr01 | +7.19 | +5.12, +1.93, +1.53, +10.40 | +0.047 | +0.21 | $248, 0.54, 56% | PASS (0.97) | 50% |
| ls_w8_f300 | +6.07 | +4.38, +8.05, +10.23, +0.28 | +0.168 | +0.46 | $298, 0.54, 65% | PASS (0.78) | 0% |
| lab_sector | +5.98 | +8.03, +7.69, +6.21, +8.76 | −0.018 | −0.05 | $221, 0.46, 55% | PASS (0.96) | 44% |
| incumbent | +5.44 | +8.24, +4.34, +2.25, +11.37 | 0 | — | $278, 0.57, 55% | — | 44% |
| win8 | +4.88 | +7.70, +11.93, +10.57, +11.64 | +0.129 | +0.31 | $296, 0.53, 64% | FAIL (0.49) | 47% |
| fixed300 | +4.57 | +0.72, +2.45, +10.26, +1.70 | −0.063 | −0.16 | $215, 0.44, 62% | PASS (0.96) | 0% |

**Frozen model: ls_w8** (the argmax; `data/experiments/clean_program/stage_c.json`).

Reading it, on the same terms as stage B: the incumbent's own four copies
span +2.25 to +11.37 %/yr on these weeks, so a ~9-point copy spread is the
noise floor of a single 6-name book over ten years. ls_w8 is 2.1 points
above the incumbent at K=4 with a paired t of 0.95 — reported, not gated:
the registered rule is the argmax, and it names ls_w8. Its four copies
(5.6-15.8) sit as high as or higher than the incumbent's four; at the
deployed settings it compounds to $365 against $278 with a deeper drawdown
(61% vs 55%; both are 2008-2009). win8's K=4 metric (+4.88) below all four
of its single copies (7.7-11.9) is the same path noise: the K=4 book is a
different six names than any copy's.

Leak audit (`ops/leak_audit.py`, per walk, 522 weeks): identity PASS
everywhere; delisting-within-8-weeks rate in the top-15 (0.4-0.7%) below
the universe's 0.8% for every candidate. Factor gate: five PASS, win8
FAIL at retention 0.49 — but read the numbers: win8's raw full-universe
weekly IC is 0.0019 (t 0.2) and its residual IC 0.0009 (t 0.1); the
retention is a ratio of two zeros, and its Spearman with the adjustment
factor is −0.015 (t −1.75), the wrong sign for a look-ahead leak. The gate
was built on the leaky champion, whose raw IC was distinguishable from
zero. win8 is not the frozen model so nothing turns on it, but the audit
as registered would gate a winner on noise: proposed amendment (not
applied) — the retention gate applies when |ic_t| ≥ 2, otherwise the
factor test is report-only with the Spearman's sign and t as the evidence.

Stage C decided the model. The strategy layers for it are decided in
stage E by `stocks-ml procedure` on the K=16 walk; nothing in the spec
changes now.

## Stage D — feature families on the frozen model — run 2026-09-11 on the owner's go

One command, `ops/clean_program.py stage_d`, reads stage C's winner from
its output file and runs build -> probe -> dedup -> walk -> exam, each step
checkpointed. Features: `stocks_ml.features.derived` — for each of the 50
per-stock ranked base features, the feature minus its sector's median that
week (50) and its change over 4, 13 and 26 panel weeks (150); 200
candidates, read from the panel alone, point-in-time (tests/
test_derived_features.py proves a later week cannot move an earlier value).
The 14 other base features are within-week constants or event flags, where
both transforms give nothing.

Probe (feature_screen.probe on 2006-2015, label = ls_w8's sector-relative
4-week return): keep iff |Newey-West t| >= 2 and the same IC sign in both
halves of the decade — 17 of 200. Dedup at pairwise |corr| <= 0.7, greedy by
|t| — 16 (d_mom4_f_vol_12w dropped). Fifteen are rank-momentum terms
(free-cash-flow yield improving over 4/13/26 weeks +; volatility and
downside deviation rising over the month −; time since the last earnings
filing; earnings yield, OCF/assets, current ratio, debt/EBITDA, insider
net, 1-week momentum changes); one is sector-relative size, which is the
plain size feature in all but name (|corr| 0.98). ICs 0.006-0.021 in
magnitude. `data/experiments/clean_program/stage_d/screen.json`.

Exam: ls_w8 with the 16 vs without, K=4, every week of 2006-2015, on the
510 weeks both ranked with a real ensemble (the same-basis rule: ls_w8's
+7.57 in stage C was on the 488 weeks all seven walks shared; on these 510
it is +6.45).

| arm | top-6 compounded %/yr | copies 1-4 | vs without (pp/4w) | paired t | 6/60-40/cap2: $100, SR, DD | leak audit | weeks |
|---|---|---|---|---|---|---|---|
| ls_w8 | +6.45 | +15.10, +8.70, +4.39, +8.80 | 0 | — | $353, 0.58, 61% | — | 510 |
| ls_w8 + engineered features* | +7.02 | +16.56, +11.21, +10.10, +3.70 | +0.047 | +0.24 | $313, 0.55, 58% | PASS | 510 |

**Admitted** — the registered rule is "with > without" on the selection
metric, and +7.02 > +6.45. Read honestly: 0.6 points a year with a paired t
of 0.24, the four copies overlapping the base model's four, and a lower
terminal at the deployed settings ($313 vs $353). This is inside the noise
of one 6-name book; the bundle is admitted because the rule says so, not
because the evidence says it helps. It carries the asterisk into stage E.

Leak audit on the with-bundle walk: identity PASS; delisting-within-8-weeks
rate in the top-15 0.5% vs the universe's 0.8%; factor gate PASS, but as in
stage C the retention (1.87) is a ratio of two noise-level ICs (raw −0.0016,
t −0.19). The score's Spearman with the vendor adjustment factor is −0.030
(t −3.5): the wrong sign for a look-ahead leak (a leak scores future
splitters higher), stronger than any stage C walk's; noted, report-only.

`data/experiments/clean_program/stage_d.json`; the walk under
`stage_d/walk/` (spec.json records the recipe and the bundle).

## Where the champion stands — 2016-2024 read on the owner's ask (2026-09-11, before stage E)

`ops/clean_program.py extend --arm ls_w8`, `extend --arm ls_w8_x`, `assess`.
Both stage D arms at K=4 on every rank week of 2016-01-08 -> 2024-07-12
(445 weeks; nothing at or past the holdout 2024-07-19 is read), joined to
their 2006-2015 walks; the incumbent from its saved dl walks at K=4 (same
basis) and K=16 (as deployed). Deployed settings 6 / 60-40 / cap 2 / no
stop, $100 at each span's start. This is the registration's one look at
2016-2024 for these arms: it grades, it selects nothing.

| model | 2006-2015: $100, %/yr, SR, DD | 2016-2024: $100, %/yr, SR, DD | 2006-2024: $100, %/yr, SR, DD | weekly t vs sp500 (2016-24 / 2006-24) |
|---|---|---|---|---|
| ls_w8 | $349, +13.3%, 0.58, 61% | $515, +21.0%, 0.88, 41% | $1,794, +16.8%, 0.71, 61% | +1.40 / +1.98 |
| ls_w8 + engineered features* | $327, +12.6%, 0.56, 58% | $443, +18.9%, 0.82, 41% | $1,449, +15.5%, 0.67, 58% | +1.06 / +1.68 |
| incumbent K=4 | $271, +10.5%, 0.56, 55% | $446, +19.1%, 0.89, 34% | $1,208, +14.3%, 0.71, 55% | +1.26 / +1.63 |
| incumbent K=16 (deployed) | $229, +8.7%, 0.49, 52% | $420, +18.2%, 0.83, 38% | $963, +13.0%, 0.65, 52% | +1.04 / +1.15 |
| sp500 | $203, +7.3%, 0.47, 55% | $316, +14.3%, 0.87, 32% | $640, +10.5%, 0.65, 55% | — |

Reading it: ls_w8 earns more than the S&P 500 in both eras (+21.0 vs
+14.3 %/yr on 2016-2024; +16.8 vs +10.5 over the 18.5 years) at about the
same Sharpe (0.88 vs 0.87; 0.71 vs 0.65) and a deeper drawdown (41% vs 32%;
61% vs 55% in 2008-09). The extra return is bought with extra risk, not
free. The weekly excess over the S&P has t 1.40 on 2016-2024 and 1.98 on
2006-2024 — suggestive, not decisive. Against the deployed champion (K=16),
ls_w8 at K=4 is ahead in both eras ($515 vs $420 on 2016-2024). Note that
40% of every book here is the S&P itself under the trend ballast, so the
Sharpe similarity is partly by construction.

The admitted bundle: lower in 2016-2024 ($443 vs $515, SR 0.82 vs 0.88),
as it was at the deployed settings on 2006-2015. It won the registered
"with > without" coin flip on the selection metric and nothing else. The
registration admits it; this read cannot un-admit it (that would be
selecting on the grading window). The owner decides what stage E carries
and what deploys, after seeing this; the choice is recorded as theirs.

`data/experiments/clean_program/assess.json`; walks under `extend/`.

## Stage E — the package at K=16 — run 2026-09-12 on the owner's go ("Ok do stage e")

What the owner chose to carry: the frozen model ls_w8 alone
(`label_4w_sector`, 8 training years, the 64 base features). The
registration's default package carried the 16 admitted engineered
features too; the owner dropped them after the K=4 2016-2024 read above.
That is recorded as the owner's discretionary call, made after seeing
2016-2024.

What ran (`ops/clean_program.py stage_e --arm ls_w8`): 16 copies on every
rank week of 2006-2015 (copies 1-4 reused from stage C — same seeds —
and copy 1 re-derived at the first week to prove it; copies 5-16 new),
then `stocks_ml.procedure.decide` on that walk, then 16 copies on every
week of 2016-2024. Walks 3.7 h. The spec was not written.

The procedure's decision on the K=16 2006-2015 walk: **book 10, halfgate
ballast, no stop, no sector cap.** Evidence: book by compounded %/yr
{3: 2.14, 6: 7.86, 10: 8.11}; floor by Sharpe {none 0.455, halfgate
0.532, 80/20 0.480, 70/30 0.496, 60/40 0.514}; stop {none 0.532, −25%
0.482}; cap {none 0.532, 2 0.523}. The margins are thin (book 10 over 6
by 0.25 %/yr; halfgate over 60/40 by 0.018 Sharpe). Halfgate means: hold
the book at 1 − g/2 of capital, where g is the share of the 30/40/52-week
SPY trend gates that are down, the rest in IEF. At the time of the run
`live/r5.py` could not express it (the ledger took a fixed fraction) and
the procedure refused to write it to the spec; `ledger.floor_split` has
run it live since 2026-09-12 (see the adoption below).

The one look (`data/experiments/clean_program/stage_e.json`). K=16,
$100 at each span's start, pre-tax (Roth account, so post-tax = pre-tax):

| model | pre-holdout 2006-2024: $100, %/yr | SR, DD | 2016-2024: $100, %/yr | SR, DD | 2006-2015: $100, %/yr | SR, DD | weekly t vs sp500 (06-24 / 16-24) |
|---|---|---|---|---|---|---|---|
| ls_w8 at its procedure settings (10 / halfgate / no cap / no stop) | $1,994, +17.5% | 0.70, 63% | $660, +24.6% | 0.88, 33% | $302, +11.7% | 0.53, 63% | +1.90 / +1.52 |
| ls_w8 at the deployed settings (6 / 60-40 / cap 2 / no stop) | $1,700, +16.5% | 0.69, 59% | $555, +22.1% | 0.91, 39% | $306, +11.8% | 0.53, 59% | +1.88 / +1.51 |
| incumbent (deployed settings) | $963, +13.0% | 0.65, 52% | $420, +18.2% | 0.83, 38% | $229, +8.7% | 0.49, 52% | +1.15 / +1.04 |
| sp500 | $640, +10.5% | 0.65, 55% | $316, +14.3% | 0.87, 32% | $203, +7.3% | 0.47, 55% | — |

Reading it. The model change is worth more than the settings change: at
the same deployed settings the package beats the incumbent by 3.5 points
a year over the 18.5 years (16.5 vs 13.0) and by 3.9 on 2016-2024 (22.1
vs 18.2). The procedure's settings add another point a year, all of it
after 2016 (on 2006-2015 the two settings tie at $302 vs $306; the
procedure chose by Sharpe, not by terminal wealth). Against the S&P 500
the package is ahead in every span, at a similar Sharpe (0.70 vs 0.65
pre-holdout; 0.88 vs 0.87 on 2016-2024) and a deeper 2008-09 drawdown
(63% vs 55% — the 10-name book with the softer halfgate ballast falls
further than the 6-name 60/40 book's 59%). The weekly excess over the
S&P has t 1.90 pre-holdout and 1.52 on 2016-2024: suggestive, not
decisive, as it was at K=4.

Falsification test (registered: paired weekly excess vs the incumbent on
2016-2024, t < −2 rejects): t **+1.32** on 446 weeks → not rejected. The
package's edge over the incumbent is +3.0 %/yr on 2006-2015 (where it was
chosen) and +6.4 %/yr on 2016-2024 (where it was not). Selection
inflation would show the opposite ordering.

Leak audit (`ops/leak_audit.py`), as registered: **FAIL**, on the
2006-2015 segment, retention −1.31. The numbers behind it: the
whole-universe weekly IC is 0.00019 (t 0.02); with the split factor
partialled out it is −0.00025 (t −0.03); the retention is the ratio of
two zeros. The direct leak test — rank correlation of the score with each
stock's future split factor — is −0.017 (t −1.8) on 2006-2015 and −0.041
(t −5.0) on 2016-2024: the wrong sign for a look-ahead leak. Identity
checks on the top splitters pass in both segments; picks delist within 8
weeks less often than the universe (0.6% vs 0.8%; 0.15% vs 0.7%). The
2016-2024 segment passes: IC 0.035 (t 4.0), retention 1.02. This is the
failure mode flagged at stage B on win8 (above), with the amendment
proposed there and still not applied: gate on retention only when the raw
IC has |t| ≥ 2, otherwise judge the factor test by the sign and t of the
score-vs-factor correlation. Under the amendment the package passes both
segments. Whether to apply it is the owner's decision; nothing in the
code has changed.

Owner's ruling (2026-09-12, 02:40): neither factor test can separate a
leak from a legitimate proxy. A model that favours strong, rising
companies favours future splitters for good reasons, so the correlation
is not evidence of a leak; and the future split factor is partly future
return, so residualizing it removes real skill. The structural check is
the proof: for the largest splitters, is the stored per-share value
restated to the day's unadjusted basis by the right factor? It is (NVDA:
stored 0.125 x 39.95 = 4.99). `ops/leak_audit.py` now gates on that
identity check alone; the factor and delisting numbers are reported per
segment, not gated. Re-run: **PASS** on both segments (identity PASS;
score-vs-split-factor −0.017 t −1.8 and −0.041 t −5.0; retention −1.31
and 1.02, report-only). AGENTS.md rule updated; ledger row refreshed via
`stage_e_amend`.

Confidence (method of reports/clean_line_confidence.md: 200 seed draws,
4000 history draws with 8-week blocks, nested 20 per seed; 95%
intervals):

| window | metric | point | seed-only 95% | history-only 95% | nested 95% |
|---|---|---|---|---|---|
| 2016-2024 | excess CAGR vs sp500 %/yr | +9.0 | +7.3 … +10.5 | −3.9 … +24.8 | −3.9 … +24.2 |
| 2016-2024 | terminal $100 | $660 | $576 … $740 | $141 … $3,181 | $140 … $3,130 |
| 2016-2024 | P(excess > 0), nested | 0.91 | | | |
| 2006-2024 | excess CAGR vs sp500 %/yr | +6.5 | +5.2 … +7.6 | −2.1 … +16.7 | −2.5 … +16.2 |
| 2006-2024 | terminal $100 | $1,994 | $1,592 … $2,397 | $181 … $24,731 | $188 … $21,793 |
| 2006-2024 | P(excess > 0), nested | 0.91 | | | |
| 2006-2015 | excess CAGR vs sp500 %/yr | +4.4 | +2.3 … +6.3 | −7.3 … +18.3 | −6.8 … +18.8 |
| 2006-2015 | terminal $100 | $302 | $246 … $361 | $50 … $1,868 | $52 … $1,947 |
| 2006-2015 | P(excess > 0), nested | 0.76 | | | |

Seed noise is small (about ±1.5 points a year). History noise is not:
the 2016-2024 excess over the S&P runs from −4 to +25 points a year
across resampled histories, and the nested probability that the package
beats the S&P is 0.91 (0.76 on 2006-2015). A 9-point-a-year edge with a
one-in-ten chance of being nothing.

Character (scratchpad `character.py`, same session): the whole-universe
ranking has no skill on 2006-2015 (the IC above); the edge sits in the
top of the ranking and mostly after 2016. The book buys high-volatility,
beaten-down names (three quarters of picks in the top volatility
quintile, far below the 52-week high, negative 1/3/12-month momentum,
beta tilt +0.36); its good years are rebound years (2009, 2016, 2019,
2020, 2021) and its bad years are the ones where the sell-off kept
going (2007, 2008). A quarter of the model's splits are market-regime
features. No single feature has standalone IC above 0.015.

What adoption needed, in order: (1) the leak audit — PASS under the
2026-09-12 ruling; (2) halfgate in the live job, so the procedure would
write the decision — done 2026-09-12 (`ledger.floor_split`, the one rule
the backtest and `live/r5.py` share); (3) the holdout untouched until the
owner's go to grade — still untouched. Ledger row `stage_e_ls_w8_k16`
(kind `clean_program_stage_e`).

## Adopted — 2026-09-12 (owner: "Ok go")

The package is the champion. Three pieces, all code paths, no hand edits:

1. **The sector label is in the package.** `features/panel.sector_label`
   computes `label_4w_sector` inside `build_panel` (the 4-week return
   minus the same-date sector median; the week's median where the sector
   is unknown), so the live job's weekly panel rebuild carries it;
   `selection.Ctx` computes it for older panels. Checked equal to the
   graded formula on both real panels (research 661,071 rows, live
   658,701; 0% of rows without a sector; correlation with the week-centred
   label 0.957). `ops/clean_program.add_labels` calls the same function.
2. **The procedure writes the model fields.** `stocks-ml procedure` reads
   the walk's own record (`<walk>/spec.json`: label, training years) and
   writes `horizon.label` and `training_window_years` into the spec next
   to the strategy layers; `live/r5.py` fits on the spec's label and
   window; tests fail if either side is typed by hand.
3. **The procedure ran on the Stage E walk.** `stocks-ml procedure --preds
   data/experiments/clean_program/stage_e/ls_w8/select/preds.parquet`
   (2026-09-12 12:07:43) wrote book 10 / halfgate / no stop / no cap and
   `label_4w_sector` / 8 years to models/champion_spec.json — the same
   decision and evidence as the run above — regenerated PROCEDURE.md and
   added the ledger row
   `procedure_2006-01-01_2015-12-31_clean_program_stage_e_ls_w8_select`;
   `stocks-ml procedure --check` matches.

Dry run of the live job on the 2026-08-28 panel: 502 names ranked in
20 s; header "halfgate trend ballast (book 100% of NAV this week), top-10
four-sleeve stagger, sector cap None, sector-relative 4-week label, 8-year
window, K=16"; all three SPY gates up, so the book is fully invested. The
2026-09-11 signal (run 2026-09-12 on the committed spec of the time: top-6,
70/30, cap 2, 5 years) is the last on the old settings; the first on the
new ones is 2026-09-19. The paper ledger's 6-name sleeves refill to 10 as
each rotates over the following four weeks.

Charts: reports/champion_vs_sp500_2006_2024.png and
reports/champion_vs_sp500_2016_2024.png (`ops/clean_program.py
champion_chart`: the spec's walk at the spec's settings; $1,994 vs SPY $621
on the same weeks, $660 vs $316).
