# stocks_ml — agent context

ML system that ranks S&P 500 stocks on a one-month horizon, wraps the ranking
in a rules-based long-only book with an index/Treasury ballast, and runs that
system every week as a $100 paper portfolio. Point-in-time Sharadar data; a
mechanical, pre-registered selection procedure; a sealed two-year holdout.
The owner's goal: turn $100 into more, without ever blowing up the account.

**Pruned 2026-09-02.** The repository now holds only the champion and its
three workflows (weekly job, selection procedure, procedure card). The legacy
one-week pipeline — ingest/tune/train/backtest/pipelines/signals/ledger/
torture, the model zoo, tuned recipes, its reports, ledgers and signals — was
deleted, not rewritten; every file this document mentions that is no longer
in the tree is at the annotated tag `legacy-final` (commit d04d0d6, the last
commit before the prune): `git show legacy-final:<path>` or
`git worktree add ../legacy legacy-final`. The history sections below keep
their original file references on purpose.

## Current state (2026-09-02)

- **Champion: r5** (`models/champion_spec.json`, rendered to PROCEDURE.md by
  `stocks-ml procedure-card`), declared 2026-09-01 after the month-horizon
  rebuild on the Sharadar world: depth-3 XGBoost (MODEL_PARAMS in
  selection.py, untuned by design), 4-week open-to-open label minus the
  week's member median, 35-day purge, weekly refit on the trailing 5 years
  with a purged time-tail early stop, K=16 week-bootstrap copies averaged
  (K=4 until 2026-09-07);
  top-6 equal weight in four staggered sleeves (one rotates per week, every
  name held four weeks), sector cap 2; 70% book / 30% ballast, the ballast
  SPY shifting to IEF one third per breached SPY trailing mean (30/40/52
  weeks); fills at the next session's open, 5 bp a side. **Since 2026-09-06
  the model also trains on the generated bundle of 40 features**
  (`features/bundle.py`: formulas over the panel's raw inputs, chosen on
  2006-2015 alone by the OpenFE arm's v3 — see "Feature engineering"; the
  spec's `features` + `formulas`, the `g_` columns `build_world_panel`
  computes on every run, ~40 s). It replaced the screened bundle of 7
  ideas adopted 2026-09-05 (x_8k_distress_26w, x_cash_runway, x_dps_yoy,
  x_earn_react_mean4, x_earn_window, x_price_level, f_vol_chg_12w; ledger
  `champion_r5_7030_cap2_bundle`, its record $3,058 vs SPY $628 with the
  ideas' asterisk stays as history). Record as deployed, on the K=16 walk
  (`data/experiments/k16_champion_2006_2015/preds.parquet` joined with
  `k16_seed_spread_2016_2024/preds.parquet`, 2006-01 → 2024-07, 967 rank
  weeks, ranks before the holdout): **$100 → $4,256 (+22.4%/yr, Sharpe
  0.92, DD 55%) vs SPY $608 (+10.2%/yr, 0.63, 55%)**, no asterisk (no
  grading year was read to choose the features; ledger
  `champion_r5_7030_cap2_generated_k16`, reports/k16_champion_record.md;
  2016 → 2024-07 $900, SR 1.14, DD 35% vs the clean line at K=16 $666 and
  SPY $310; 2006-2015, the bundle's own selection window, $473 vs SPY
  $197). The K=4 walk of record until 2026-09-07 (v3's,
  `openfe_v3_2006_2015/holdings_4w_5y_x724d05_s0`; ledger
  `champion_r5_7030_cap2_generated`) graded $3,498 (+21.1%/yr, Sharpe
  0.88, DD 50%; 2016 → 2024-07 $863 vs the clean K=4 line $590, the ideas
  line* $772) and stays as history: it is copies 1-4 of the K=16 walk,
  which reproduce it row for row. Before
  any bundle the same settings graded $1,553 (+15.9%/yr, Sharpe 0.72, DD
  62%) vs SPY $623 (+10.4%/yr, 0.64, 55%) on 2006 → 2024-06 (1,195 rank
  weeks), read with the spec's caveats (era-concentrated edge, ~+3%/yr of
  design-iteration shine on dollars, size for SPY-like adverse regimes).
  **The seed spread is measured (2026-09-07, `ops/k16_seed_spread.py`,
  preregistration `k16_seed_spread_2016_2024`, reports/k16_seed_spread.md):**
  sixteen copies of the champion's model on every rank week 2016-01-08 →
  2024-07-12, each copy's scores saved (data/experiments/k16_seed_spread_2016_2024/).
  Copies 1-4 reproduce v3's walk row for row ($863); the three other
  disjoint K=4 draws grade $597, $614, $697 — the record was the luckiest
  of four (CAGR 23.1-28.5 %/yr, Sharpe 0.98-1.13); twelve random K=4
  subsets average $643 ($513-$758). Two disjoint K=8 draws $733 and $738;
  K=16 $896 (SR 1.14). Random subsets by K (exploratory, overlapping):
  K=2 $645, K=4 $643, K=8 $762, K=12 $843, K=16 $896 — averaging more
  copies raises the graded result monotonically. Two disjoint K=4 ensembles
  agree at week-averaged Spearman 0.70 (2.8 of 6 top names shared), two
  K=8 at 0.80 (3.4 of 6). So the generated line's margin over the clean
  record ($590, one K=4 draw) was inside the K=4 seed noise.
  **The K=16 program (owner's go 2026-09-07, `ops/k16_program.py`,
  preregistrations `k16_clean_2016_2024` and `k16_champion_2006_2015`,
  verdicts `k16_clean_verdict` and `k16_champion_2006_2015_verdict`):**
  (1) the clean line at sixteen copies on 2016-01 → 2024-07, the same bar
  at the same K (reports/k16_clean_line.md): clean K=16 $666 (+24.7%/yr,
  SR 0.98, DD 42%) vs generated K=16 $896 (+29.1%/yr, SR 1.14, DD 35%),
  SPY $310; clean K=4 draws $584 / $700 / $626 / $553, K=8 $660 / $599 —
  **the generated bundle stays**. Copies 1-4 reproduce the clean walk of
  record on 397 of 441 common weeks and every other week is explained
  with no score differing: 39 are tie reorders (shallow trees give many
  names identical scores; `slice_row` breaks ties by arrival order) and 5
  are weeks where the record lists a name that left the tape within the
  horizon (ADT1, WFM, SBNY: `regrade_campaign.graded_rows` keeps the
  cached ticker list while the returns skip unlabelled names; worth $1 on
  the record's grade; the live job ranks every tradable name, a
  backtest-vs-live gap left as is). (2) The champion at sixteen copies on
  2006-01 → 2015-12 (2.7 h), joined with the seed-spread walk = the K=16
  record above (reports/k16_champion_record.md); copies 1-4 reproduce
  v3's walk on 522 of 522 weeks. 2006-2015: K=4 draws $405 / $580 / $345 /
  $347 (mean $419), K=8 $487 / $469, K=16 $473, SPY $197. 2006-2024: K=4
  draws $3,498 / $3,434 / $2,493 / $2,218, K=8 $3,560 / $3,554, K=16
  $4,256. Two disjoint K=4 ensembles agree at Spearman 0.72 (2.7 of 6 top
  names), K=8 0.81 (3.3 of 6). The cascade's book-down layers re-decided
  on the K=16 holdings over 2006-2015 alone (`run_cascade`'s own rules,
  no new fits): top-6 (17.88 %/yr vs top-3 15.76, top-10 15.19), floor
  **halfgate** (Sharpe 0.830 vs 70-30 0.743, 60-40 0.760, 80-20 0.730,
  none 0.701), no stop (0.830 vs 0.827), **cap none** (0.830 vs cap 2
  0.808); the four K=4 draws pick halfgate every time, top-3 three times,
  cap 2 twice. Per the preregistration the champion's settings stand
  (top-6 / 70-30 / no stop / cap 2); halfgate and cap-none are the
  owner's separate decision, not adopted, and no alternative setting has
  read 2016-2024. (3) **K=16 is the protocol since 2026-09-07**
  (`selection.K_COPIES`, the spec's `k_copies`, replication.py, the live
  job's signal header — every finalist alike; `tests/test_r5.py` pins the
  spec's `k_copies` to `K_COPIES`; the live fit is 16 copies instead of
  4, well under two minutes of a ~25 min job); ledger
  `champion_r5_7030_cap2_generated_k16`. Uncommitted until the owner
  commits the change set (before the Saturday 2026-09-12 run).
  Every clean campaign number was re-graded twice:
  on the fixed rank-date join (commit dccca9a: Thursday-dated picks had been
  paid for the week that had already ended; `ops/regrade_campaign.py`,
  reports/rank_date_regrade.md; before the fix the record read $1,586 vs
  $563) and on the live job's fill rules (2026-09-03: `selection.simulate`
  runs `stocks_ml.ledger`, the code the weekly job trades with — next-open
  fills, 5 bp a side, dust threshold — and `slice_row` grades open-to-open
  like the training labels; `ops/fill_basis_regrade.py`,
  reports/fill_basis_regrade.md; the close-basis record read $1,644,
  +16.3%/yr). No cascade layer's argmax moved either time. The clean nested
  OOS grade as deployed is **$521 vs SPY $316** (procedure v3, ledger
  `nested3_verdict`, 2026-09-04; 2016-01-01 -> 2024-07-19 exclusive, 446
  weeks: +21.2%/yr, SR 1.00, DD 34% vs +14.3%/yr, SR 0.87, DD 32%). The v2
  record stays in the ledger as history: $463 vs $307
  (`nested2_verdict_amended`; $464 on the close basis, $497 vs $295 as first
  graded; $469 vs $316 on v3's 446 weeks).
- **Selection is mechanical** (`stocks-ml select`, selection.py): a fixed
  cascade — horizon, training window, book size, ballast, stop-loss, sector
  cap — one decision per layer by a metric declared in advance, run on the
  selection window only; every evaluated configuration is a row in
  `models/trials_ledger.json` (legacy rows stay as history). Validated by the
  nested test in `reports/nested_selection_protocol.md`. Re-selection only
  on a structural trigger (new data source passes its gate, a pre-registered
  kill criterion fires, or the owner directs it) — never on a calendar.
  **Procedure v3 (2026-09-04, owner mandate: no decision on a sample):**
  every layer now reads every week of the selection window. Until then the
  training-window layer read ~116 spaced sampled weeks and the book layer
  29 of them (one 4-week phase of the sample); re-running that cascade on
  the same caches flipped the book from top-10 to top-3 (OOS $463 -> $695
  vs SPY $307), so the v2 nested line was partly a coin flip. v3: the
  window sweep is a population holdings walk at every window (1-5y), the
  book is decided on the chosen window's population holdings, and
  compounded statistics average the four phases (`compounded_pct`). Both
  OOS grades were seen before the change; the rule was registered
  (`nested3_v1`, `nested3_features_v1`) before its numbers. Result
  (`nested3_frozen_config`): 4w / 5y / top-6 / 60-40 / stop -25% / cap 2 —
  the 5y window won clearly (+3.05 %/yr edge vs random, 1-4y all negative),
  top-6 beat top-10 (2.94 vs 1.90 %/yr) and top-3 came last (-0.11), so the
  v2 sample's top-3 was noise; graded $521 vs SPY $316. The champion was
  selected under v2 (its whole-window cascade `select_2006-01-01_2024-07-18`
  froze top-3 on a sampled book layer); re-selection under v3 needs four
  more population walks over 2006-2024 and is the owner's separate call.
- **Feature engineering is a stage of the same procedure** (2026-09-04;
  `stocks-ml select --screen`, feature_screen.py): candidate columns
  (`x_`, features/candidates.py — the 30 "tails" ideas) ride in the panel
  invisibly; a run probes them on its own selection window and examines
  the keepers as one bundle against its own base holdings in a second
  population walk before the cascade sees them. **Admission rule v3.1
  (owner, 2026-09-05):** the bundle enters iff the top-6 book's cost-
  adjusted compounded %/yr on the exam weeks is higher with it than
  without — the book layer's own metric, the argmax standard every other
  layer and the champion were held to; v3's bar (paired calendar-HAC t > 2)
  was a stricter standard than the incumbent ever faced ("we are imposing a
  higher standard on the challenger than on the incumbent") and now stays
  in the report as a statistic. The rule was changed after seeing the
  2006-2015 exam (t 1.61) and before any grading-year number
  (preregistration `nested3_features_v2`). A bundle reaches the champion
  only on the owner's go (`frozen_config["features"]`, then the spec's
  `features`, which the live job reads). Asterisk rule (owner, 2026-09-04): the
  candidate *ideas* were written from the whole 2006-2024 record, so any
  nested grade with the bundle is reported as a separate "+ engineered
  features" line, marked as such, beside — never in place of — the clean
  nested result (v3 `nested3_v1`; v2's was $463 vs $307). The windowed
  screen on 2006-2015 (`feature_screen` row `nested3_v1`, 2026-09-04,
  reports at data/experiments/nested3_v1/screen.md) kept 7 of 48 candidates
  (x_8k_distress_26w, x_cash_runway, x_dps_yoy, x_earn_react_mean4,
  x_earn_window, x_price_level, f_vol_chg_12w) and examined them as one
  bundle in a population walk with them: top-6 +0.45%/4w (0.98% -> 1.43%
  per hold), HAC t 1.61 on 508 weeks — below v3's bar of 2, **nothing
  admitted** under v3 (`nested3_features_verdict`). **Under v3.1
  (`nested3_v2`, 2026-09-05, same caches, data/experiments/nested3_v2/):**
  the top-6 book compounds 10.16 vs 4.03 %/yr with the bundle on those
  weeks, so it was admitted; the with-bundle cascade froze top-3 / halfgate
  / no stop / cap 2 (book top-3 14.74 vs top-6 11.49 vs top-10 8.61 %/yr;
  halfgate beat 60/40 by 0.001 Sharpe) and was graded once on 2016 ->
  2024-07-19: **$1,379 (+35.8%/yr, Sharpe 1.14, max DD 40%) vs the clean
  line $521 (SR 1.00, DD 34%) vs SPY $316 (SR 0.87, DD 32%)**, 446 weeks
  (`nested3_features_v2_frozen_config`, `nested3_features_v2_verdict`).
  The bundle line beats the clean line out of sample; its deeper drawdown
  comes with the riskier layer its own selection chose. **Adopted into the
  champion 2026-09-05** (owner: "the pros of adding engineered features
  typically outweigh the cons. XGB is good at not using bad features, so
  worst case scenario the new features give little lift or slight
  degradation"): the spec's `features` carries the bundle, the live job
  trains with it (dry run on 2026-08-28: 502 names ranked in 7 s; 4 of the
  6 picks overlap the no-bundle signal), and the champion's record is the
  with-bundle line under "Champion" (`ops/champion_bundle_regrade.py`
  grades it on the walk `champion_2006_2024/holdings_4w_5y_xeeaf48_s0`,
  nested3_v2's rows extended back to 2001-06, ~9 min). The champion keeps
  its own book and ballast (top-6 / 70-30); the with-bundle cascade's
  top-3 / halfgate pick would grade $7,684 (+26.3%/yr, SR 0.94, DD 55%) on
  the champion's span and stands in the ledger as an option
  (`champion_bundle_top3_halfgate_2006_2024`), not adopted. The pooled
  sample exam that had passed (`tails_exam_v3`, 346 weeks, +0.91%/4w,
  t 2.75) chose and graded its 8-feature bundle on weeks from the whole
  2006-2024 record, grading years included; the windowed screen chose its
  own bundle (5 of those 8, plus x_dps_yoy and f_vol_chg_12w) on 2006-2015
  alone, and that was the bundle adopted — until the generated bundle
  replaced it (below).
- **Generated features — the OpenFE arm** (features/generated.py,
  ops/openfe_arm.py, ops/openfe_arm_v2.py; 2026-09-05/06). The owner's
  purpose: replace the ideas bundle, whose candidates were written after
  reading 2016-2024, with a bundle "based only on 2006-2015". Inputs
  written down as a rule (the 64 base features before ranking, every SF1
  field as level and yoy, every 8-K item code as a 26-week count, the
  close: `raw_inputs`), OpenFE's order-1 formula space, every generated
  column ranked per week like any feature. Three arms, each one walk at the
  champion's settings and one grade on 2016-01 → 2024-07 (2006-2015 is
  in-sample): **v1** (`openfe_2006_2015`, OpenFE's own two-stage selector,
  14 features) $760; **v2** (`openfe_v2_2006_2015`, the weekly-Spearman-IC
  t statistic on 56,758 candidates, dedup |corr| ≤ 0.9, 40 features — but
  run WITH the 7 ideas riding along, a mistake against the purpose) $697;
  **v3** (`openfe_v3_2006_2015`, `--generated-only`: v2's selector, the
  dedup held against the 64 base features only, no ideas anywhere)
  **$863 (+28.5%/yr, SR 1.13, DD 36%) vs the clean line $590 vs the
  ideas line* $772 vs SPY $310** (preregistration
  `openfe_arm_v3_generated_2006_2015`, grade
  `openfe_arm_v3_generated_2016_2024`, reports/openfe_arm_v3.md). The bar
  for a clean bundle is the clean line. **Adopted 2026-09-06 on the
  owner's decision** in place of the ideas (`champion_r5_7030_cap2_generated`):
  `features/bundle.py` holds the 40 formulas; `build_world_panel` computes
  the `g_` columns for every row (`generated.add_generated` on
  `generated.raw_inputs`, the research walks' recipe), so live and research
  use one code path — checked 2026-09-06: the research store through
  `Ctx` + `ensemble_preds` + `rank_members` reproduces v3's walk top-15 at
  2024-07-12 exactly. Known about the bundle: it is redundant (mean
  |pairwise corr| 0.28, PC1 33%; mostly size and nominal price written many
  ways — the reason v2 with the ideas did worse, not a lack of signal:
  35/40 formulas keep their IC sign out of sample, median OOS/IS |IC|
  0.63). Its ranking is also the most sensitive of the three feature sets to
  the data snapshot: the Mac's live store (Sharadar refreshed 2026-09-02)
  and the research store differ by a hair per cell (mean 0.002-0.003 on the
  rank scale, 502 vs 505 names a week), yet the same Friday's top-6 shares
  4/6 names across the two stores for the clean and the ideas models and
  1-3/6 for the generated one (Spearman of the rankings 0.55-0.72 vs
  0.73-0.94; three Fridays, 2022-2024). Not a bug in the path — a property
  of the model plus the bundle; what a backtest promises is the
  distribution of picks, not the names. Ideas never enter the arm (v2's
  seven are a fixed list in `ops/openfe_arm_v2.py`, not the spec's features).
- **The holdout (2024-07-19 onward) is a single-use exam.** It has not been
  graded for r5 and nothing may touch it without the owner's explicit go.
- **One grade-window convention (2026-09-09):** every pre-holdout window ends
  at `selection.HOLDOUT_START` (2024-07-19) as an EXCLUSIVE bound, so the
  label credited at the first holdout close is never counted. Before this,
  some graders used `Timestamp("2025")` (967/447-week grades crediting that
  label; the $4,256/$900 record rows) and the nested verdicts used
  07-19-exclusive (966/446): numbers from the two bases differ by one week
  and must not be compared without saying so.
- **Live:** `.github/workflows/champion.yml` is the only workflow; it runs
  `stocks-ml r5-weekly --commit` every Saturday (see "r5 weekly job"). The
  paper record is `ledger_r5.json` + `signals_r5/<friday>.{md,json}`. The
  legacy shadow race (`signals/`, `ledger.json`, `ledger_ltr.json`) ended
  2026-09-01 and its files live at the tag.
- **Why the legacy pipeline was retired** (its verdicts still bind): week-
  ahead ranking skill existed around 2001–2004 and has been noise since;
  month-scale structure is real, concentrated in a handful of features, and
  Sharadar fundamentals add to it at that horizon; no in-sample screen (CV
  rank IC, realized top-k excess, Optuna objectives) predicted out-of-sample
  dollars — walked exams are the only evidence accepted; re-tuning on a
  calendar was destructive at every cadence tested; the free-data world was
  missing ~200 delisted constituents and the edge it showed was a data
  artifact.

## Commands

```bash
uv sync                      # install (Python 3.12 — what the champion is locked and run on)
uv run pytest                # 207 tests (1 skipped); MUST stay green with 0 warnings
uv run stocks-ml r5-weekly [--as-of F] [--no-refresh] [--no-sec] [--dry-run] [--commit]
                             # the champion's weekly signal (Actions runs it; below)
uv run stocks-ml select --sel-start A --sel-end B [--eval-start C --eval-end D] [--screen]
                             # the full selection procedure on a window
                             # (stages grid/wsweep/holdings/[screen]/cascade; PROCEDURE.md;
                             # wsweep = population holdings at every window; --screen =
                             # the feature screen, feature_screen.py; `--screen --stage
                             # cascade` = the with-bundle arm, frozen_config_x/eval_x.json)
uv run stocks-ml procedure-card   # regenerate PROCEDURE.md from models/champion_spec.json
.venv/bin/python ops/regrade_campaign.py {build,cascade,report,charts}
                             # the campaign's numbers on the fixed rank-date join
                             # (needs data/experiments/r4w_campaign_cache/, git-ignored)
.venv/bin/python ops/live_emulation.py   # champion graded by the live ledger's rules (8 s)
.venv/bin/python ops/champion_bundle_regrade.py {grade,charts,all}
                             # the champion with the bundle on its recorded basis (~9 min;
                             # reports/champion_bundle_regrade.md; charts need `uv run --with matplotlib`)
.venv/bin/python app/oos/build.py [oos|oos_x|select|champion]
                                  # explorers -> reports/{oos,oos_x,select,champion}_explorer.html
                                  # (nested3 OOS test $521 vs $316; the same with the bundle $1,379;
                                  # the v2 cascade's pick on 2006-2024 $1,424; the champion as deployed
                                  # $3,058 vs $628; git-ignored; ~20 s each)
/opt/homebrew/Caskroom/miniconda/base/bin/python -m pytest tests/e2e   # their Playwright beta test
ops/r5_seed.sh               # re-seed the Actions cache with the Mac's live world
ops/r5_weekly.sh             # the weekly cycle by hand on the Mac (no commit)
```

`select` needs the frozen research world `data/sharadar_world2000/` (built
once from Sharadar; git-ignored, never refreshed); `r5-weekly` needs
`data/r5_live/` and a Sharadar key (`data/.sharadar_key` or
`SHARADAR_API_KEY`). Sample first: cheap sampled runs by default, full-
population runs only on the owner's explicit go.

## r5 weekly job (2026-09-01)

The production champion (`models/champion_spec.json`, PROCEDURE.md) gets its
signal from `stocks-ml r5-weekly` (src/stocks_ml/live/r5.py), run by the
GitHub Actions workflow `.github/workflows/champion.yml` every Saturday
13:00 UTC with `--commit` (dispatchable by hand with `as_of` / `dry_run`;
~25 min, nearly all data refresh — the K=16 ensemble fits in well under two
minutes). Outputs
are tracked and committed by the job as "r5: signal <friday>":
`signals_r5/<friday>.md` (+ `.json`) and `ledger_r5.json`; the signal also
lands in the run's summary page. A run fails loudly (GitHub emails the
owner) rather than signal on stale data.

**Licensed data never enters the public repo.** The Sharadar key is the
`SHARADAR_API_KEY` repo secret. The live world (`data/r5_live/`, ~0.5 GB)
travels between runs as an AES-256 tarball in the Actions cache
(`r5-live-<run_id>`, restored by prefix), encrypted with the `R5_STORE_KEY`
secret — the same key is in git-ignored `data/.r5_store_key` on the Mac. A
snapshot is saved only after a successful cycle, so a failed run retries from
the previous one. When the cache is empty (first run, or evicted: GitHub
drops caches idle for 7 days or beyond 10 GB per repo — the Wednesday
`keepalive` job restores it midweek so Saturday-to-Saturday never counts as
idle) the run seeds itself from the *draft* release "r5 seed" that
`ops/r5_seed.sh` uploads from the Mac (drafts are invisible to the public;
the run deletes it once the snapshot is in the cache). If a run fails with
"no live world in the Actions cache and no seed release", run
`ops/r5_seed.sh` and re-dispatch. Draft-release download and the secrets
API need the `contents: write` token the workflow already has.

`ops/r5_weekly.sh` runs the same cycle on the Mac by hand (it does not
commit; logs in git-ignored `logs/`). Its launchd schedule was retired on
2026-09-02 when the job moved to Actions — two champions refreshing and
rotating independently would only race each other; the plist stays in
`ops/` in case Actions ever has to be replaced (copy it to
`~/Library/LaunchAgents/` and `launchctl bootstrap`).

What it does, in order:
1. `data/world.py` refreshes the live world store `data/r5_live/`
   (bootstrapped once from the frozen research world
   `data/sharadar_world2000/`, which stays untouched): sp500 → membership;
   SEP/SFP → prices, incremental by `lastupdated` and upserted by
   (ticker, date) — Sharadar bumps `lastupdated` on a ticker's whole history
   when a dividend re-adjusts it, so the upsert is exact; SF1 ARQ/ART →
   fundamentals (full refetch, refuses to shrink >2%); SF2 → insiders and a
   Form 4 *bridge* (SF2 non-derivative P/S rows in the SEC Form 4 schema,
   filed after the frozen SEC file's 2026-03-31 — the SEC quarterly zips lag
   and the job never fetches them). Then the free-data ingesters: EDGAR
   companyfacts for every current member (replace semantics — a refetched
   ticker's rows replace its stored rows, `refresh_days=0`), 8-K, FINRA short
   interest, FRED. Direct API limits: ≤30 tickers and ≤200 chars per
   `ticker` filter (`_chunks`). **Symbol renames rewrite history**: when
   Sharadar renames a company (EQR → VMRK, 2026-09-01: same `permaticker`
   197624, `relatedtickers="EQR"`, EQR gone from sp500/tickers/SEP), the old
   symbol would look like a departed member and its history would be dropped
   as an orphan while the new symbol refetched from scratch. `detect_renames`
   (permaticker unchanged, old symbol gone, new symbol unheld, unambiguous)
   rewrites the old symbol in every stored table first, so membership stints,
   fundamentals and insider history carry over. EDGAR CIKs for a renamed
   symbol resolve through `relatedtickers`.
2. `build_world_panel` reruns the research recipe (build_panel with the
   world's backtest_start, then the Sharadar fundamental/insider features and
   rank_normalize). Two fidelity rules, each learned from a failed bit-for-bit
   check: **IEF must not be in the panel's price frame** (f_mkt_dispersion is
   a cross-section over every price series; the research panel was built
   before IEF was appended for ballast pricing — `_PanelStore` drops it), and
   **membership stints are rebuilt with the pre-snapshot fix**: Sharadar's
   sp500 events start 1998-01-09, before its first `historical` snapshot
   (1998-03-31); the research builder opened a second, never-closed stint for
   the five tickers added in between, so LEHMQ/BIGGQ/SUB1/MTL1 stayed
   "members" forever with all-neutral features and NaN labels (508 open
   stints vs 503 current). Closing them changes no other panel row.
3. `selection.ensemble_preds` ranks the Friday's members exactly as the
   research did (K=16 week-bootstrap copies since 2026-09-07, K=4 before;
   label_4w, 5-year window, purge
   35 d; `ctx.extra = SPEC["features"]`, the generated bundle since
   2026-09-06 (the screened ideas 2026-09-05 → 2026-09-06) — the panel
   carries every `x_` candidate and every `g_` bundle column, computed by
   `build_world_panel` on each run, and the job refuses a panel that lacks
   a bundle column, never a silent neutral fill). A
   name must have a close in the last 7 days to be rankable (≥100
   required, else the job fails loudly rather than trade a thin universe).
4. Sleeve schedule (`ledger.rotate_sleeves`, the code `simulate` runs): four
   6-name sleeves (sector cap 2 from the top-15), one rotates per week —
   `((friday_of(t) − 2001-01-05) // 7 days) mod 4`, so a holiday Thursday
   rotates the Friday's sleeve; empty sleeves fill at once (the first week);
   a sleeve ≥5 weeks old (the job skipped its week) rotates too. Ballast:
   per 30/40/52-week third, IEF when SPY's weekly close is below the
   trailing mean, else SPY. Weights: 0.7 × 1/24 per sleeve slot, 0.3 split
   across the thirds.
5. Paper ledger (`ledger.Ledger`): fills LAST week's pending weights at the
   first open after the decision date (Monday; a name with no open within
   five sessions closes at its last print), sells first, buys sized net of
   a 5 bp fee (never overdrawn; rebalances under 0.5% of NAV skipped, full
   exits always run), then marks NAV at Friday's close against SPY
   buy-and-hold from the same $100. Units are on the closeadj (total-return)
   basis, so each mark stores a reference close per position and the next
   run rescales units by old/new reference close before anything else
   (value-preserving, tested). Orders decided on the signal date wait for
   the next run.

Fails loudly (no signal written) when the panel's last date is not the last
Friday (prices not refreshed), when the ensemble returns nothing, or when the
rankable universe is thin. `--as-of` reruns an older Friday; `--no-refresh`
ranks the stored panel; `--dry-run` writes nothing.

First run 2026-09-01 (signal 2026-08-28, all four sleeves filled at once as in
`simulate`'s first week): the live rebuild reproduced the research `panel_sf`
on its 657,195 common rows — p99 |Δ| ≤ 0.02 on every feature (rank drift from
the vendor's dividend re-adjustments and the cross-section fixes below);
material diffs (>0.2) in 0.5% of rows, all data the research world lacked:
EDGAR facts for T/MCK/BRK.B/BF.B, SF1 for VMRK(EQR)/PAS/ECO1 and 15 other
symbols, AEP's 8-Ks, insider filings after 2026-03-31 (the bridge); plus 3,371
phantom-stint rows removed. Sharadar SEP also serves some rows twice (249 in
the research pull) — `prices_from_sep` dedupes on (ticker, date).

**launchd (the retired local schedule) cannot read `~/Documents` (macOS
TCC) without a grant:** the first kickstart died with exit 127 — `/bin/zsh: can't open input file:
…/ops/r5_weekly.sh` (`ls ~/Documents` → "Operation not permitted" from any
launchd agent; deep paths are blocked too). Resolved 2026-09-02: the owner
gave `/bin/zsh` Full Disk Access (System Settings → Privacy & Security; no
CLI can do it — the TCC database is SIP-protected). Two gotchas: the plist
must run `/bin/zsh <script>` explicitly — with the script as the program
(shebang) zsh itself could read the repo but `uv` died with "Current
directory does not exist", because children are attributed to the job's
*program* for privacy checks; and a job that starts within a second of the
grant still fails. The alternative remains moving the repo out of
`~/Documents` (also ends the iCloud problems below; then update `ops/`
paths and re-`launchctl bootstrap`). Reload after editing the plist:
`launchctl bootout gui/$UID/com.stocks-ml.r5-weekly && launchctl bootstrap
gui/$UID ~/Library/LaunchAgents/com.stocks-ml.r5-weekly.plist`; manual
trigger `launchctl kickstart gui/$UID/com.stocks-ml.r5-weekly`.

**Pipeline league (2026-08-11, owner's design; legacy, at the tag):**
pipelines may differ in training objective, strategy, and cadence, but are
all judged by one $100 cost-aware walk-forward exam (backtest/pipelines.py). Wave 1: incumbent
regression; weekly learning-to-rank (WeekGroupedXGBRanker, rank:ndcg with
week query groups, per-week quintile relevance grades); monthly-cadence
regression on the panel's `label_4w` (4-week horizon; purge 42 days —
consumers must purge past the label span); P(top-quintile) classifier
(TopQuintileClassifier, extremes-only training) traded via ConfidenceTopK
(floor 0.5) + SpyFloor. Challengers are deliberately untuned in wave 1; the
league grows the Sharpe-deflation trial count; the shadow ledger stays the
final judge.

The champion workflow commits `signals_r5/` and `ledger_r5.json` every
Saturday ("r5: signal <friday>"), so the repository may need to be
synchronized before local work.
However, do not run `git` commands unless the owner explicitly requests them;
when synchronization cannot be confirmed, work from the current checkout and
state that limitation in the final summary.

## Architecture

```
src/stocks_ml/
  data/       store.py (parquet DataStore + manifest), world.py (the live world:
              Sharadar refresh, rename detection, panel rebuild for r5),
              sharadar.py (Direct API transport: key + paginated fetch_table),
              membership.py (normalize_symbol, members_asof), prices.py
              (corrupt-series filter), edgar.py (companyfacts, filing-dated),
              sec8k.py, shortint.py (FINRA), fred.py (macro; only audited
              T10Y2Y/FEDFUNDS admitted)
  features/   panel.py (build_panel = the one place features/labels are made;
              REJECTED/PENDING feature sets), fundamentals.py, events.py,
              insiders.py, sharadar_fundamentals.py, ranking.py,
              candidates.py (the x_ candidate columns: built from the store,
              ranked, appended to panel_sf by build_world_panel; invisible to
              feature_cols until a screen admits them)
  models/     xgb.py (TimeTailEarlyStopXGB + dated_features), walk.py
              (walk_forward_predictions: staggered no-lookahead walk),
              replication.py (WeekBootstrapEstimator, the K-copy protocol),
              trials.py (models/trials_ledger.json)
  selection.py     the selection procedure; ensemble_preds is the champion's
                   model call (also what r5 ranks with; ctx.extra = admitted
                   features); simulate replays the ledger over a holdings file
                   (trace hook for attribution)
  feature_screen.py  the feature screen (probe + bundle exam), a stage of
                   select: rules in its docstring; screen.json / screen.md
                   under the experiment, `_x` holdings = the with-bundle arm
  features/bundle.py  the champion's generated bundle: 40 formulas (FORMULAS)
                   and the g_ column names (FEATURES); provenance in its docstring
  features/generated.py  raw_inputs (the formulas' inputs for every panel row)
                   and add_generated (evaluate, rank per week) — the one code
                   path behind the g_ columns, research walks and the live job
  ledger.py        the live rules and paper ledger, shared by simulate and the
                   weekly job: sleeve schedule, pick_capped, ballast, target
                   weights, next-open fills at 5 bp a side, NAV marks
  procedure_card.py  PROCEDURE.md from models/champion_spec.json
  live/r5.py       the weekly job: rank, then ledger.py's rules; the report
  cli.py           r5-weekly | select | procedure-card
.github/workflows/champion.yml   the champion's Saturday cycle (the only workflow)
ops/          r5_seed.sh (seed the Actions cache from the Mac), r5_weekly.sh +
              com.stocks-ml.r5-weekly.plist (manual / retired local schedule),
              regrade_campaign.py (the campaign re-graded on the fixed join;
              close basis, not to be re-run), fill_basis_regrade.py (the same
              on the live fill rules), live_emulation.py (what fill timing
              is worth), openfe_arm.py + openfe_arm_v2.py (the generated
              features arm: v1 / v2 / v3 `--generated-only`; outputs under
              data/experiments/openfe_*, reports/openfe_arm*.md)
models/       champion_spec.json, trials_ledger.json      (tracked)
reports/      nested_selection_protocol.md, source_point_in_time_audit.md,
              rank_date_regrade.md, fill_basis_regrade.md, live_emulation.md,
              champion_bundle_regrade.md, the two champion-vs-SPY charts
              (clean and with-bundle lines)                (tracked)
app/oos/      build.py + app.html: interactive explorers of a chosen configuration,
              week by week (replays selection.simulate with its trace hook). Four
              variants: `oos`, the v3 nested OOS test (data/experiments/nested3_v2's
              clean walk, 2016 -> 2024-07-19, $521 vs $316); `oos_x`, the same
              test's with-bundle arm (its frozen_config_x, top-3 / halfgate,
              $1,379 vs $316); `select`, what the v2 cascade froze on the whole
              selection window (4w/5y/top-3/60-40/stop -25%/no cap) graded on
              that window, in-sample: $1,424 vs SPY $623 (ledger select_2006_2024_
              mechanical_pick_graded); and `champion`, the spec's configuration
              read from models/champion_spec.json (features included) on the
              champion's own walk, 2006 -> 2024-07: $3,058 vs $628. Every page's
              notes carry the screen's asterisked bundle line (features_line).
              The built pages embed per-hold weekly returns, so they are
              git-ignored.
signals_r5/, ledger_r5.json                                  (tracked, job-written)
```

Labels: `label` = forward 5-trading-day return, open-to-open, minus the
cross-sectional member median that week (so the task is *ranking* stocks, not
predicting the market); `label_4w` = the 4-week analogue, the champion's
target (consumers must purge past its span: 35 days).
Features are rank-normalized to (-1,1] per week; prefixes: `f_` = model feature,
`aux_` = raw helper (never ranked), `f_evt_`/`f_mkt_`/`f_macro_`/`f_sec_` =
rank-exempt (time-only or binary), `x_` = screen candidate (ranked, carried
by panel_sf, never a model input unless admitted by name). Only ALFRED-audited `T10Y2Y` and `FEDFUNDS`
macro features are admitted; revision-prone macro series and all sector-derived
features are excluded (the free-data world's Wikipedia sectors were not
effective-dated; the sector cap uses Sharadar's sector only to cap, never as a
feature). See `reports/source_point_in_time_audit.md`. Diagnostic metric: mean
weekly Spearman rank IC ("IC"). For scale: 0.01 is real, 0.02 is good, 0.05+
means suspect a bug — but selection is by walked dollars, never by IC.

## Iron rules (breaking these silently corrupts everything)

0. **Strategy evaluation leads with the HOLDOUT section/chart**, never the
   full-history headline — the selection window overlaps model selection.
   Owner-mandated. (For r5 the holdout is not yet graded; until the owner
   opens it, the nested-selection frozen grade is the honest number.)

1. **No lookahead.** Every feature at week t uses only data knowable at t's
  close: date-only EDGAR/Form 4 records become available the next calendar day;
  revision-prone FRED and non-effective-dated sector features are model-excluded;
   labels start the next trading day, training is purged past the label's
   span (10 days for `label`, 35 for `label_4w`), early stopping uses a
   time-ordered tail (never a random split). `tests/test_no_lookahead.py`
   proves this by corrupting future data and asserting past outputs unchanged —
   if it fails, fix the code, NEVER the test.
2. **The holdout (2024-07-19 onward) is untouchable.** No tuning or selection
  may see it; `select` runs on the selection window only. Holdout results may
  be reported only after all choices are frozen, and only on the owner's go.
3. **Every week must be rankable.** Constant (unrankable) predictions in a
  week are a degeneracy, not a virtue (history #4: the legacy tournament's
  "test weeks" column caught it). r5 refuses to signal on fewer than
  MIN_UNIVERSE (100) rankable names or an empty ensemble; a member that
  early-stops to ~no trees only adds a near-constant offset to the K-copy
  mean (walk.py docstring).
4. **Tests green, zero warnings**, no network in tests (fetchers are injectable;
   fixtures only). Silence third-party noise via their own APIs, not warning filters.
5. Money math is guarded: weights ≥ 0, sum ≤ 1, cost-netted buys, sells
   before buys, never overdrawn, no leverage — one engine, `ledger.Ledger`,
   for the research exam (`selection.simulate`) and the paper account
   (tests/test_ledger.py, test_selection.py). Backtests emulate the live
   deployment: a change to the live rules is a change to the engine.
6. **Champion selection is mechanical (owner-mandated 2026-08-20; metric
   amended 2026-08-21).** The champion is the argmax of PRE-TAX EARNINGS
   (terminal $ from 100) on the 2001-2024 extended pre-holdout, SR tiebreak
   ("SR weights risk too heavily" — owner). Deflated Sharpe reported from
   ledger counts. (The legacy leaderboard tool, `stocks-ml leaderboard`,
   is at the tag.) NO discretionary overrides: any concern about a winner must take
   the form of a pre-registered falsification test (predictions written down
   BEFORE the result, e.g. seed replication) run before promotion. Ensembling
   over a dimension (seeds, params) is permitted only when replication has
   MEASURED that dimension to be noise. Origin: Claude overrode the rule for
   a "winner's curse" argument; the owner challenged; seed replication proved
   the argmax right (history #9).
   The canonical K-copy procedure (K=16 since 2026-09-07, K=4 from
   2026-08-20; random_state=c + whole-week training
   bootstrap seeded by c; average the copies' predictions) is
   `selection.ensemble_preds` over `models/replication.WeekBootstrapEstimator`
   — use it, never an ad-hoc variant, so no finalist is advantaged by its
   ensembling.

## Missing-data and feature-admission policy

The current 4-fold, 2015-03+ evaluation design is fixed. New features must fit
the benchmark; a feature's late start or sparse history is not a reason to move
fold boundaries or shorten the primary evaluation window.

1. Every candidate must produce rankable predictions for every week in the same
  evaluation calendar. The count is derived from the current panel, fixed
  `eval_start`, and rolling holdout rather than hard-coded, so it grows as new
  data arrives (the current artifact has 488 weeks). Never drop rows or weeks
  because an optional feature or prediction is missing or constant; incomplete
  coverage makes a candidate ineligible.
2. Distinguish structural pre-source absence, cross-sectional gaps, incidental
  retrieval failures, and economically meaningful absence. Track and report
  weekly and cross-sectional coverage by feature family.
3. Optional ranked features are neutral-filled after ranking (`0` in rank
  space) rather than causing observations to be dropped. Add point-in-time
  missingness indicators only where absence may carry information. Any fitted
  imputation parameters must use training data only.
4. Keep a stable, full-history core feature set. Evaluate additions by ablation
  on identical folds, weeks, and eligible stocks, with and without missingness
  indicators where appropriate.
5. Late-starting features must be tested both in the official full benchmark
  (neutral before availability) and, when useful, in a separately labeled,
  predeclared common-window ablation. The secondary window never replaces or
  mixes with the primary leaderboard.
6. Prefer improvements that are stable across folds, not gains concentrated in
  one period. Before production adoption, stress-test realistic random gaps
  and complete outages of the feature family.

Model-native missing-value support does not override this policy: it may handle
scattered company-level gaps, but it cannot justify missing weeks, unequal
scoring calendars, or all-missing folds.

## Hard-won history (why things are the way they are)

File references in this section are as of tag `legacy-final`; the verdicts
are current.

1. **automl_tool** (owner's package) selects by MAE, and the label's median is 0
   by construction — so it picks constant-predictors. Structurally excluded by
   the eligibility gate. Fix would be adding a scoring param upstream.
2. **Corrupt price data:** free sources garble delisted tickers (CPWR showed
   +300% "weeks" from broken split adjustments). `drop_corrupt_series` removes
   series with repeated split-sized jumps. Known false positive: GME (real
   squeeze) — accepted. ~200 delisted tickers (incl. Lehman) have NO free price
   data at all → residual survivorship bias, quantified in `data/manifest.json`
   and stress-tested in `stocks-ml torture` (verdict: removal-exit channel
   clean; missing-bankruptcies channel unquantifiable with free data).
3. **First backtests were invalidated twice** (corrupt data → fake 43% CAGR;
   then a constant-predictor champion). Treat any CAGR > ~25% or IC > ~0.05 as
   a bug until proven otherwise. `docs/run-notes-2026-07-11.md` documents both.
4. **The Optuna gaming incident:** short-interest data only exists from 2017-12.
   In earlier 5-fold CV, the 2012-15 fold had all-NaN short columns; some model
   fits collapsed to constant predictions there, those weeks were silently
   dropped from scoring, and Optuna found a config that "won" (fake IC 0.0281
   on 369/611 weeks; honest value 0.0143). Resolution (owner's design): drop
   that fold — 4 folds from 2015-03 — and keep short features (a controlled A/B
   showed they add ~+0.002 IC consistently). CatBoost still partially collapses
   (visible as 371 test weeks in selection.md) but loses, so it's tolerated.
5. **What moved the needle vs what didn't:** hyperparameter tuning (+42% rel.)
   and new features (+32% rel.) worked; insider data at weekly horizon, model
   zoo (LightGBM/CatBoost), and prediction-ensembles were nulls. Lesson: new
   *information* helps; new *algorithms* mostly don't. ElasticNet nearly ties
   XGBoost — the signal is largely linear in rank space.
6. **Old IC numbers are not comparable across evaluation designs.** The metric
   window/folds changed (5-fold-2012+ → 4-fold-2015+). Compare only within
   the current design.
7. **The survivorship reckoning (2026-08-19, Sharadar re-grade).** On the
   survivorship-clean world (official 500-member menus 1998+, 100% price
   coverage incl. delisted via suffixed tickers, dividend-adjusted opens),
   NO config beats SPY on pre-holdout Sharpe: best 0.57 (seccap4) vs SPY
   0.60; the old world claimed 0.73. SPY reproduces exactly across worlds
   (665 vs 666) — the build is validated; the historical edge was data
   artifact. What survives: the family ORDERING (seccap 0.57 > eased 0.54 >
   banded 0.45 ≥ k16 0.44 — the strategy-layer mechanics are real), and the
   holdout-era edge (1.20-1.27 vs SPY 1.11; near-complete data in both
   worlds). The honest thesis is now recent-era-only. Clean-world store:
   data/sharadar_world/; walk: data/walks/sharadar_champion.parquet. All
   pre-Sharadar absolute levels vs SPY are deprecated for decisions.
8. **The clean-world retrain (2026-08-19, post-reckoning).** Exam-objective
   ASHA on the Sharadar world (24 configs incl. incumbent, rungs incl. real
   GFC): incumbent params scored 0.391 on true 2005-2015 (not even top-8);
   winner = config-4 region (depth 2, lr 0.003, 500 trees, reg_alpha 1.34 —
   MORE conservative than the incumbent). Final = 7-member neighborhood-
   averaged book: pre-holdout SR 0.640 vs SPY 0.60, +2.7%/yr paired t=1.56,
   DSR 0.864 (N=554) — the first measured real historical edge; suggestive,
   not proven. Holdout 1.05 vs old-params-on-clean 1.25: era tension — old
   params look better recently, new params are real historically. Both
   should shadow-race on the clean pipeline. Final book:
   scratch world_final_book.parquet; neighbor walks data/walks/world_*. All
   six pending feature families re-rejected on clean data (t -1.4 to -2.3).
9. **The untuned-knobs campaign + the seed-replication reversal (2026-08-20,
   owner-directed).** Multi-objective Optuna (pre-tax SR + post-tax earnings,
   walked exams, no proxies) over never-tuned XGB dims found t12: depth-2 +
   lossguide + dart (rate_drop 0.29) + num_parallel_tree 4 + gamma 0.16.
   Claude declared the 7-jitter neighborhood-avg champion on winner's-curse
   grounds; the owner challenged; a PRE-REGISTERED seed-replication test
   (seeds 1-3 of exact t12 params) came back 0.722/0.666/0.735 — t12's score
   is real skill, the ±20-40% jitters genuinely degrade it (siblings mean
   0.636). Lesson: average over MEASURED noise, select over MEASURED signal —
   and adjudicate which is which by replication, not assumption. CHAMPION =
   t12 4-seed ensemble: pre-holdout SR 0.697 ($1,249), holdout 1.12/19%,
   +3.9%/yr vs SPY paired t=2.30 (first conventionally significant result;
   caveat: computed on the selection window), DSR 0.920 (N=570, bar 0.95).
   Book: scratch champion_t12_seed_ensemble_book.parquet; walks
   data/walks/world_t12*. Multi-objective
   Optuna (pre-tax SR + post-tax earnings, both from walked exams — no proxies)
   over the never-tuned XGB dimensions found a new champion region: depth-2 +
   lossguide + dart dropout + num_parallel_tree 4 + gamma pruning — every
   winning knob is another form of internal ensembling/skepticism. t12
   neighborhood-avg (7 books): pre-holdout SR 0.653 ($1,098 vs SPY $674),
   holdout 1.11 at SPY-equal 19% DD, +3.3%/yr paired t=1.79, DSR 0.882
   (N=562) — supersedes the config-4 region (0.640/t=1.56) as research
   champion. First config to beat SPY post-tax ($847 vs $536, t12 solo).
   Params: scratch optuna_world_finalists.json[0]; neighbors t12_neighbors
   .json; book t12_final_book.parquet; walks data/walks/world_t12*.
9b. **Strategy face-off under the K-copy standard (2026-08-20).** The 21x64
   model-strategy matrix (1,344 exam cells, all ledger-counted) showed
   single-eval strategy leaderboards are noise (twin models agree at rank
   corr ~0.19); family-averaged (11 twin walks) leaderboards have real
   resolution. Its top config beat the incumbent chassis under identical
   K=4 bootstrap-copy ensembles: CHAMPION is now t12 (4 bootstrap copies,
   replication.py standard) x cfg15 chassis (k=10, plain entry, exit 14,
   lam 0.54, sector cap 2, equal weight, SpyFloor): pre-holdout SR 0.669
   ($1,159), holdout 1.35 / DD 19%, +3.6%/yr vs SPY (t=1.87), DSR 0.961 at
   N=1,917 — first result to clear the 0.95 deflation bar (caveat: adding
   1,344 tightly-clustered strategy cells lowers ledger cross-trial
   variance, which flatters DSR; the paired-t is the cleaner stat).
   cfg15-vs-incumbent margin: +0.81%/yr, t=1.61 (leaderboard rule applied).
   Books: scratch faceoff_cfg15_book.parquet; walks world_t12boot1-4.
9c. **The trend floor (2026-08-20, owner's downturn-alarm idea).** The SPY
   floor (~80% of the champion book) now switches to IEF when SPY sits below
   its trailing weekly MA; deliverable = average of the 30/40/52-week-window
   books (scratch champion_trendfloor_book.parquet). Exam: pre-holdout SR
   0.797 / DD 27% / $941 vs 0.669 / 60% / $1,159 pure-SPY floor; holdout
   1.41 / DD 12%. 2008 DD -57% -> -27%, 2020 -34% -> -13%. Ridge across all
   six window/refuge variants (0.755-0.802). Honest mechanics: paired return
   vs pure floor is -2.1%/yr (t=-0.85) — the SR gain is entirely risk
   reduction, not extra return; ~2-4 floor switches/yr; era favors trend
   (two mega-crashes in sample; 2022-style choppy declines ~neutral).
   Point-in-time (MA through decision date), costs included, K-copy
   standard inherited (floor mod is deterministic on the ensemble book).
9d. **The pre-sample verdict (2026-08-21, dot-com extension).** The frozen
   champion was graded on 2001-2004 — an era no selection ever touched
   (panel data/sharadar_world2000, walks world2000_t12boot1-4). It PASSED:
   trend-floor champion $130 / SR 0.64 / DD 19% vs SPY $100 / 0.09 / 42%
   over the dot-com bear + recovery; pure-floor $131 / 0.44 / 38% (the
   model's stock picks alone added ~31% over a flat market). The activation
   gate opened out-of-sample (48-83% active weeks 2001-04, peaking in the
   2003-04 recovery) — the recovery-confirmer mechanism generalizes to a
   crash of a different species. Consistency check: 2005-2024 on extended
   walks reproduces 0.80/27% exactly. Expected honest degradation: pre-
   sample SR 0.64 < selected-era 0.80. 2001-2004 is now SPENT as a test and
   may be folded into future training/validation; the 2024+ holdout remains
   the only virgin era.
9e. **The earnings-era board (2026-08-21, owner's metric).** Under pre-tax
   earnings ranking with joint (strategy, floor) selection, the champion is
   ext-t20 (depth-5 gbtree, npt 3 — from the extended-window search, ranked
   mid-pack by SR and #1 by earnings) x cfg57 (k10 cap3) x trend->IEF floor:
   $2,678 / SR 0.671 / DD 55% on 2001-2024 (4-copy standardized ensemble),
   vs t12's best $1,443 and SPY $656. Model slots by measured ensemble
   earnings: t20 >> t12 > c4 > t23. cfg57 won for 3 of 4 models (strategy
   ridge is cross-model). Note: for t20 the trend floor ADDS earnings —
   its crashes were deep enough that avoidance out-compounds whipsaw.
   Registry: data/leaderboard_books/; `stocks-ml leaderboard` renders.
9f. **The nested-selection verdict, Phase A (2026-08-21, pre-registered in
   reports/nested_selection_protocol.md).** The full selection procedure —
   24-trial earnings search, K=4 ensembling, 8x4 strategy/floor battery —
   was run using only 2001-2012; its winner C0 (depth-5 dart x cfg47 x
   static 60/40 floor) was frozen and walked on never-selected-on
   2013-2024: $337 / SR 0.715 / DD 31% vs SPY $462 / 0.876 / 34%. The
   procedure's honest out-of-sample estimate LOSES to buy-and-hold. The
   in-window-selected champion's slice of the same window ($414) also
   trails SPY: the measured edge lives in the crash/recovery eras inside
   the selection window. Winner's curse in miniature: the 60/40 floor won
   selection by $6 and was the worst floor out-of-sample ($337 vs pure
   $441, trend $373 — best variant still under SPY). Every full-window
   backtest number carries this haircut; only holdout and live are exempt.
   Phase B (2026-08-22): re-search cadences do not close the gap — P2 (2y)
   $398 / SR 0.606 / DD 54%, P4 (4y) $348 / 0.580 / 51%, vs P1 frozen
   $337 / 0.715 / 31% and SPY $462 / 0.876 / 34%. Re-tuning bought a few
   earnings dollars at ~0.1 SR and ~20 DD points: every re-search
   re-picked cfg57 (the strategy ridge is stable across selection
   windows) but flipped the floor to chase the last regime (60/40 ->
   trend -> 60/40 -> trend -> pure SPY, adopting pure right before the
   2022 bear). Procedures beat SPY in 2 of 9 deployment segments. Live
   policy implication: freeze the config; re-search rarely, adopt only on
   decisive evidence.
   Phase C final (2026-08-24, annual cadence + offset variants): P3 (1y)
   $272 / SR 0.488 / DD 54% — the worst procedure; annual churn is
   destructive. Offsets expose the noise floor: same-cadence chains
   anchored one year apart differ by $45-94 (P2 $398 vs oP2 $304; P4
   $348 vs oP4 $393) — the spread BETWEEN cadences is no larger than the
   spread WITHIN one, so the Phase B cadence ranking was anchor luck.
   Robust findings across all six chains: every procedure < SPY ($462);
   every re-tuned chain carries DD 51-56% vs frozen P1's 31%; P1 has the
   best SR (0.715) of any procedure; cfg57 won the strategy slot at all
   11 anchors (a real ridge); the floor flipped constantly (curse churn).
   P3 beat SPY in 2 of 12 annual segments. Experiment closed: the
   pipeline's honest out-of-sample product is SPY-minus with worse risk
   unless frozen; calendar re-tuning at any cadence adds risk without
   reliable return.
9g. **The raw-signal verdict (2026-08-26, owner's test).** Chassis-free
   question: do models selected on data through T rank stocks better than
   SPY after T? Statistic: weekly mean return of the raw top-10 minus SPY,
   on cached predictions from honest post-selection weeks only. C0
   (selected ≤2012, all 603 weeks of 2013-2024): −0.048%/wk, t −0.38.
   Pooled 12 nested winners on their deployment years: +0.088%/wk,
   t +0.61, hit rate 0.50. Individual winners scatter −16% to +27%
   annualized (|t| ≤ 1.2) — the same noise that drove selection churn.
   Verdict: NO measurable average-week ranking signal vs SPY in the
   price/volume feature set; power rules out anything > ~+12%/yr. OOS
   dollars earned vs SPY (holdout, dot-com) trace to episodic regime
   positioning + floor, not weekly stock-picking. Related earlier finding:
   among 11 cached finalists, NO in-sample metric (earnings, hit rate,
   top-k excess, IC) rank-predicts OOS dollars (rho -0.42..+0.14, N=11);
   in-sample hit rates cluster at 0.49-0.51. Standing implication: model
   selection among current-feature XGB siblings optimizes noise; the
   productive margins are new data (SF1 fundamentals) or acceptance of
   SPY.
   Better-data test (2026-08-27, pre-registered sf_prototype_v1/v2):
   Sharadar full-history fundamentals + insider flow (16 f_sf_*/f_sfi_*
   features, 87-99% coverage in every era incl. 2002, point-in-time by
   filing date; stores + modules committed) do NOT create next-week skill.
   Paired population exam, all 455 weeks 2013-2021, champion params:
   baseline hit10 0.495 / IC -0.007; fundamentals 0.494 / -0.000; paired
   diffs hit10 t 0.42, ex10 t 0.94, IC t 1.70 — all below the registered
   t>2 bar. The small-sample +1.8pt hit10 was sampling luck (regressed at
   full n). Verdict: with these features the 1-week horizon is
   unpredictable; remaining untested levers are horizon (label_4w),
   universe, or acceptance of SPY.
9h. **The horizon discovery (2026-08-27) — first pre-registered positive.**
   Feature-level probe (seconds, no models; new standing screen order:
   feature probe -> sampled model exam -> full walk): week-ahead ICs are
   chance-level, but month-scale structure is real and concentrated in
   ~4 features (short-interest days-to-cover, size, overnight-return
   history, 1-month reversal), turning on at k=2w, plateauing k=4-6w,
   persisting to 13w; after bet-frequency adjustment the value peak is
   k=3-5w. Model-level 4-week exam (all 470 weeks 2013-2021, fresh
   champion-params fit per week, overlap-corrected block t): baseline
   panel hit10_4w 0.517 (t 1.25); WITH Sharadar fundamentals 0.536
   (t 2.58) — PASSES the pre-registered bar; fundamentals' paired
   contribution +2.0pts (t 2.02); paired IC slightly negative (gain is
   in hit rate/top-slice, not all metrics). Hyperparameter door closed:
   8-config sweep's best (slow_deep, 0.555 on 99 weeks) collapsed to
   0.499 on the full population — max-of-N sampling luck, as with every
   prior small-sample mirage. Era decomposition (9g addendum): weekly
   skill existed in 2001-04 (t12 pre-sample +0.78%/wk t 2.5), was
   marginal 2008-11, gone since; the holdout dollar edge traces to the
   banded strategy's multi-week holds (an accidental month-horizon
   harvester), not week-ahead picks. Proposed next campaign (owner
   decision pending): month-horizon rebuild — label_4w, matched purge,
   slower cadence, nested-protocol honest grading.
   POPULATION GRID (2026-08-31, 2,405 week-slots, simple-DT K=4, raw / no
   costs, vs expected random-k basket; figures re-graded 2026-09-03 on the
   fixed rank-date join): modern era 4w CONFIRMED — top-3 +16.9%/yr
   (block-t 2.4), top-6 +12.0 (t 2.2), top-10 +8.6 (t 2.0); compounded
   +27.8/+22.8/+18.5%/yr vs SPY +14.3. Modern 1w ~= 0 (fourth and final
   confirmation; sampled +10-14%/yr was luck; t <= 1.2 on the fixed join).
   2001-2012: 1w positive-unprovable over the full era (t 1.3-1.5;
   significance was concentrated in 2001-04 per the t12 pre-sample), 4w
   dead (t <= 0.8).
   Book-size ordering (3/6/10) flips between samples = noise; top-6
   stands as the robustness choice. Raw DD 49-88% everywhere — the floor
   layer is mandatory. Convergence: the vsRand estimate needs ~400 weeks
   for ~+/-4%/yr precision; 100-week samples swing +/-15-25%/yr
   (calibrates sample-first). Owner rule: cross-horizon comparisons only
   over equal calendar spans.
10. **Screens do not predict the exam (2026-08 model campaign).** Three screen
   designs — mean CV IC, top-8 realized excess on CV folds, and a tail-objective
   Optuna search — each produced winners (2-3x claimed improvements) that LOST
   the walked $100 exam by 0.08-0.21 Sharpe. Ten model challengers (lgbm,
   catboost, enet, RFF, MLP, 4y window, label variants, retuned params), zero
   wins; every book blend landed at 0.68-0.72 vs the champion's 0.73. Standing
   rule: walks are cheap (~40 min) — walk everything, screens are junk-filters
   only. The one live blend path: a neural sleeve at SR >= ~0.67 with rank corr
   ~0.1 (MLP hit 0.63/0.11, blend 0.72 — lost by 0.01); TabM is the candidate.
8. **The vol_scaled cash-lock (found 2026-08-18):** drawdown is measured vs the
   all-time peak, so a strategy at zero exposure freezes its NAV and its
   drawdown can never improve — the full stop was an absorbing state. It fired
   in COVID (2020-03-16, dd 32%) and vol_scaled sat in cash for the remaining
   6.4 years of every zoo backtest since. Fixed with a 13-week cool-off, then
   half exposure (`VolScaledTopK.REENTRY_WEEKS`); zoo reports generated before
   the fix understate vol_scaled. Lesson: any risk rule that can zero exposure
   must have a non-drawdown-based way back in.

## Data source quirks

- **yfinance / Wikipedia (legacy free-data world; ingesters at the tag):**
  yfinance is unofficial (batches of 100, per-ticker failures never fatal;
  Stooq fallback DEAD behind a JS anti-bot wall); Wikipedia's changes table
  is a MultiIndex with "Effective Date" and its format changed under the
  legacy job in 2026-08. The champion never touches either.
- **EDGAR:** 10 req/s, User-Agent required. Fundamentals sparse pre-2010.
  Insider Form 4 via quarterly bulk zips (2006+). 90-day staleness refresh
  (the r5 job uses `refresh_days=0`: replace every current member). SEC
  ticker format is BRK-B; Sharadar's is BRK.B — map through
  `normalize_symbol` (`world.sharadar_cik_map`).
- **Sharadar Direct:** `ticker` filter ≤30 tickers / ≤200 chars per call;
  `lastupdated.gte` on SEP/SFP returns only re-adjusted or new rows; SF1
  `lastupdated` is bumped for a ticker's entire history (incremental ≈ full,
  so refetch); SF2 `from`/`to` filter on the filing date; `tickers` with a
  ticker list returns one row per table (sicsector consistent per ticker);
  the `table=` filter combined with a ticker list returns nothing; a
  renamed company keeps its `permaticker` and lists the old symbol in
  `relatedtickers` (see "r5 weekly job").
- **FINRA short interest:** free unauthenticated Query API, history ~2018+,
  publication lag = settlement + 14 days.
- Tiingo probed as backfill for missing delisted tickers: covers 114/202 but
  none of the 2008 casualties. Not integrated (decided against the dependency).

## Environment gotchas

- **RESOLVED 2026-09-01 — the repo is iCloud-excluded.** It lives at
  `~/Documents/projects/stocks_ml.nosync` (iCloud never syncs a `.nosync`
  path); `~/Documents/projects/stocks_ml` is a symlink to it so old paths
  work. Keep both; never move the repo back under a synced name. The entries
  below are history: what happened while it was synced, and the cure if it
  ever is again. Last incident (2026-09-01): iCloud had evicted 35k `.venv`
  files and 1,081 `.git` objects; anything reading them blocked in
  `read()`/`mmap`, so pytest and git sat at 0% CPU for 14+ minutes — it
  looks like a hang or timeout and is neither. Diagnose with
  `find <dir> -flags +dataless -type f | wc -l`; cure with per-file
  `brctl download` (the directory form is a silent no-op).
- **iCloud syncs ~/Documents and repeatedly sets a hidden flag on `.venv`
  files; Python 3.12 then skips `.pth` files and `import stocks_ml` breaks.**
  Mitigations in place: `pythonpath = ["src"]` in pyproject (pytest immune) and
  a sitecustomize shim inside `.venv` (lost if venv is recreated). If imports
  break mysteriously, this is why. Durable fix: keep `.venv` out of iCloud.
- **iCloud also EVICTS large `data/*.parquet` contents under disk pressure**
  (2026-08-19: prices.parquet at 0 blocks, panel.parquet half-evicted).
  Symptom: `Parquet magic bytes not found in footer` on files that read fine
  hours earlier. Check `stat -f blocks:%b` before assuming corruption; fix
  with `brctl download` and verify blocks ≈ size/512. Never rebuild the store
  over an eviction — the local "corruption" is not data loss.
  Nastier variant (2026-08-22, post-reboot): partial materialization where
  `stat` reports full st_size, sequential reads (`cat`, `wc -c`) stream the
  full byte count, but random access returns empty (`tail -c 4` → nothing)
  so pyarrow fails on the footer — and st_blocks can stay 0 even after
  successful reads, so block count is not proof either way. `brctl download`
  did not clear it. Working fix: sequential `cp` to local tmp, verify the
  copy's `PAR1` footer + a real pyarrow open, then swap into place (keep the
  original aside as `*.evicted`; never delete it — deleting the placeholder
  can delete the cloud copy). Processes that already hold a file open keep
  working through an eviction — only fresh opens break.
- Sharadar is a DIRECT subscription (api.sharadar.com/v1.0/data/{table},
  x-api-key header; tables by modern name `fundamentals`/`insiders`/... or
  legacy code SF1/SF2; docs at sharadar.com/llms.txt; bulk via years=full).
  Do NOT call data.nasdaq.com — the key is foreign there, and Nasdaq answers
  with a scary "account temporarily disabled" throttle message that has
  nothing to do with the real Sharadar account (2026-08-27 incident).
  Owner's plan: Full History Bundle — includes ALL tables. Point-in-time
  join for fundamentals: `date` (filing date) <= decision Friday, ARQ/ART
  dimensions only (MR* rows are restated backward -> lookahead).
- macOS: no `timeout` command; GNU-isms differ.
- `models/`, `reports/`, `signals_r5/`, `ledger_r5.json` are git-TRACKED
  (audit trail; the job writes the last two); `data/` is not — the live world
  travels encrypted in the Actions cache, the research world stays on the Mac.

## Open items / likely next steps

- **Let the paper ledger accumulate** before real money — the plan since day
  one. Owner mandate (2026-09-03): training runs and backtests emulate the
  live deployment as closely as possible — hence the shared engine. Fill
  timing is worth -0.1%/yr over 2006-2024 (`ops/live_emulation.py`,
  reports/live_emulation.md: $1,580 -> $1,553; picks do not gap up over the
  weekend, +3.1 bp vs +5.7 bp for the names they replace); the rest of the
  -0.4%/yr between the close-basis $1,644 and the as-deployed $1,553 is the
  ledger's accounting, mostly the 5 bp on every rebalance trade. The
  pre-registered sizing/kill/promotion contract for the
  legacy champion (DEPLOYMENT.md at the tag) needs an r5 equivalent before
  any real order.
- **The holdout exam (2024-07-19 →)** is graded once, on the owner's go, with
  the champion frozen as specified; the result never feeds re-selection.
- **PENDING_ABLATION_FEATURES** (features/panel.py) stay out of the model
  matrix; all three candidate families were rejected at t≥3 on the legacy
  weekly label (reports/ablation_*.md at the tag). Re-admission is a
  structural re-selection trigger, not an edit; the screen probes them
  alongside the `x_` candidates.
- **nested3 (done 2026-09-04, owner's go):** `data/experiments/nested3_v1/`
  holds the population grid, the 1-5y window sweep (`holdings_4w_{1..5}y_s0`,
  5y extended to 2024-07-19), the with-bundle walk
  `holdings_4w_5y_xeeaf48_s0` (the stem's hash names the 7-feature bundle;
  the screen's second walk runs to the eval end, 2024-07-19, so an admitted
  bundle can be graded without another walk — its 2016+ rows were never
  read), screen_probe.parquet, screen.{json,md}, and the clean
  line's frozen_config.json / eval.json; the chain was `select ... --name
  nested3_v1 --stage wsweep | holdings | cascade | --screen --stage screen |
  --screen --stage cascade` (grid and 5y caches copied from
  `data/experiments/champion_2006_2024/`, the caches nested2_v1 graded).
  Clean line $521 vs $316; "+ engineered features" line under v3: nothing
  admitted (t 1.61), equals the clean line. `data/experiments/nested3_v2/`
  (2026-09-05) re-ran the screen and the with-bundle cascade on copies of
  the same caches under rule v3.1 (`--stage cascade | --screen --stage
  screen | --screen --stage cascade`, ~1 minute, no new walk): admitted,
  frozen_config_x.json / eval_x.json = $1,379 vs $316 (see "Feature
  engineering"). nested3_v1 keeps the v3 record. The clean 5y walk lacks 8
  weeks of 2010-2013 that the bundle walk has (paired numbers use the 508
  common weeks; the with-bundle book order holds on both sets). A full-
  window screen/re-selection (`select3_2006_2024`, 2026-09-05) was launched
  on a misread "go", killed after 12 minutes and WITHDRAWN in the ledger:
  reading 2016-2024 for selection leaves nothing to backtest on; its
  partial caches sit in `data/experiments/select3_2006_2024/`, unused.
  The bundle decision is taken (adopted 2026-09-05, "Feature
  engineering"); the with-bundle walk was extended back to 2001-06 into
  `data/experiments/champion_2006_2024/holdings_4w_5y_xeeaf48_s0` for the
  champion's record. One owner decision remains open, not started:
  champion re-selection under v3 on the full pre-holdout window (four more
  population walks; the with-bundle nested cascade's top-3 / halfgate is
  the hint that the book and ballast layers might move).
- **OpenFE arm (done 2026-09-06, "Feature engineering"):**
  `data/experiments/openfe_2006_2015/` (v1: inputs.parquet + inputs.json,
  the raw inputs every arm reads; timing.json; stage1_scores.parquet,
  stage2_gains.csv, formulas.json, generated.parquet, the walk
  `holdings_4w_5y_x<hash>_s0`, grades.json), `openfe_v2_2006_2015/`
  (stage1_ic.parquet — the IC t statistic of 56,758 candidates, the file v3
  copies; the with-ideas walk, 16 h) and `openfe_v3_2006_2015/` (bundle.csv,
  formulas.json = features/bundle.py, generated.parquet, the champion's
  walk `holdings_4w_5y_x724d05_s0` 2006-01-06 → 2024-07-12, grades.json).
  The research panel `data/sharadar_world2000/panel_sf.parquet` carries the
  g_ columns since 2026-09-06 (192 columns; identical to v3's
  generated.parquet), as the live panel does.
- **Structural triggers for re-selection** (the only ones): a new data
  source passing its gate, a pre-registered kill criterion firing, or the
  owner's direction. Calendar re-tuning is forbidden by evidence (history).
- Known weak spot to keep in mind when sizing: the edge is era-concentrated
  (index-like in whipsaw and mega-cap regimes); the spec's own caveat.

## Where to look

- `PROCEDURE.md` — the champion's procedure card (generated from
  `models/champion_spec.json`; edit the spec, not the card).
- `models/trials_ledger.json` — every evaluated configuration.
- `reports/nested_selection_protocol.md` — the pre-registered nested test of
  the selection procedure; `reports/source_point_in_time_audit.md` — the
  ALFRED/PIT audit; the two `reports/*.png` charts — champion vs SPY.
- `signals_r5/`, `ledger_r5.json` — the paper record, written by the job.
- `docs/research/` — notes on the papers the design leans on.
- Tag `legacy-final` — the legacy pipeline, its leaderboards
  (`models/selection.md`, `models/tuning_*.md`), reports (`reports/backtest.md`,
  `reports/survivorship_torture.md`, ablations, probes), DEPLOYMENT.md, the
  original design spec and build plan (`docs/superpowers/`) and
  `docs/run-notes-2026-07-11.md`.
- `.superpowers/sdd/progress.md` — full build/decision ledger (gitignored, local)
