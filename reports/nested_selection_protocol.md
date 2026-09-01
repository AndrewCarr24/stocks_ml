# Nested selection experiment — pre-registered protocol (2026-08-21)

Every leaderboard number grades a configuration on the same window that
selected it. This experiment grades the selection **procedure** on data it
never saw: an honest estimate of live performance, and a measurement of
whether periodic hyperparameter re-search earns its keep.

## Windows

- **Selection window:** 2001-01-05 → 2012-12-31 (inclusive).
- **Evaluation window:** 2013-01-04 (first decision Friday of 2013) →
  2024-07-19 (holdout start). The 2024-07-19+ holdout stays untouched.

## The selection procedure (identical at the base and at every re-search anchored at T)

1. **Search:** fresh Optuna TPE study, 24 trials (3 workers × 8), over the
   unchanged 16-hyperparameter XGB space of the extended-window search.
   Trial objective = pre-tax earnings (terminal $ per 100) of a single-copy
   walk on 2001→T under the fixed cfg15 chassis
   (SectorCapElite(10,10,14,2) + EasedWeights 0.54 + SpyFloor) — the same
   chassis every prior rung used. Sampler seeds: base S0 = 500+worker;
   re-search anchored at year Y = 1000 + 10·(Y−2000) + worker.
2. **Finalists:** top-2 trials by objective → K=4 ensembles each, per the
   replication standard (replication.py: WeekBootstrapEstimator, copies
   c = 1..4).
3. **Battery:** 8 shared strategy configs (interaction ids
   15, 32, 28, 57, 47, 27, 52, 37) × 4 floors (pure SPY / static 80/20 /
   static 60/40 / trend→IEF averaged over MA 30/40/52) on each finalist's
   ensemble book, graded by pre-tax earnings on 2001→T, SR tiebreak.
4. **Config = argmax cell** (model hyperparameters + strategy + floor).
   At S0 **nothing is enqueued** — the current champions' parameters were
   found by searches that saw post-2012 data, so seeding them would leak
   the future. At re-searches the deployed incumbent's hyperparameters ARE
   enqueued as one trial (they were selected legitimately on ≤T data);
   strategy and floor are always re-selected by the battery.

## Procedures (the contenders)

All graded on the identical evaluation window from $100, pre-tax, with the
sp500 row alongside. The current champion is **excluded** from the ranked
comparison — it was selected on 2012–2024 data; its evaluation-window slice
may be quoted only as a clearly-labeled diagnostic of selection inflation,
never as a contender.

- **P1 — frozen:** S0's config, weekly refits, config fixed for the whole
  evaluation window.
- **P2 — re-search every 2 years:** anchors end-2014, -2016, -2018, -2020,
  -2022.
- **P3 — every year:** anchors end-2013 … end-2023.
- **P4 — every 4 years:** anchors end-2016, end-2020.

Splice convention: a re-search anchored at Dec 31 of year Y deploys its
winner at the first decision Friday of Y+1; each segment's book starts
fresh (transition turnover is a real cost of re-tuning) and NAV chains
across segments.

## Phases

- **Phase A (this run):** S0 + P1. Deliverable: the honest
  future-performance estimate, plus the labeled inflation diagnostic.
- **Phase B:** even-year anchors → P2 and P4.
- **Phase C:** odd-year anchors → P3 and offset variants (2-year cadence
  starting 2013 vs 2014, etc.), which are free splices once the annual
  anchor set exists.

## Amendment (2026-08-21, registered before any re-search ran)

- Anchors are shared across cadences: the search and battery at anchor T
  run once; every cadence re-searching at T reads the same winner W_T.
  The enqueue at T is therefore the union of the deployed incumbents of
  the cadences that re-search there (each selected on ≤T data, deduped):
  S2014 {C0}; S2016 {W2014, C0}; S2018 {W2016}; S2020 {W2018, W2016};
  S2022 {W2020}. Total budget stays 24 trials including enqueues.
- Splice mechanics: each winner's book runs from its deployment start over
  its longest deployment span and is truncated per cadence (strategy
  causality makes truncation exact); segment NAVs chain multiplicatively.
  Entry costs of the incoming book are paid inside each segment; exit
  costs of the outgoing book (~bps of one turnover) are not modeled.

## Amendment 2 (2026-08-22, registered before any Phase C anchor ran)

- Odd-year anchors (2013, 2015, 2017, 2019, 2021, 2023) enqueue P3's
  deployed incumbent: the previous year's winner (S2013 {C0}; S2015
  {W2014}; S2017 {W2016}; S2019 {W2018}; S2021 {W2020}; S2023 {W2022}).
- Accepted deviation: the even-year anchors were run during Phase B with
  only the even-cadence incumbent unions enqueued; P3 reuses their
  winners as-is rather than re-running them with W_odd added to the
  enqueue. (The incumbent was kept at only 1 of 5 even anchors, so the
  enqueue's influence on the winner is measured-small.)
- Offset variants defined now: oP2 re-searches at odd anchors
  (segments 2014-15, 2016-17, …), oP4 at 2014/2018/2022 (segments
  2015-18, 2019-22, 2023-24). Winners' deployment books that span beyond
  a cached segment walk are covered by extension walks with the same
  copies (predictions are start-independent by construction — staggered
  refits — so concatenating cached prediction spans is exact).

## Accepted-in-advance caveats

- The search space, feature panel, strategy battery, floor families, and
  chassis were all developed with full-window knowledge. This experiment
  isolates the **final selection step's** inflation only; the truly virgin
  tests remain the holdout and the live future.
- Each cadence yields a single spliced path, so cadences are adjudicated
  with per-segment incumbent-vs-challenger tables and offset variants, not
  terminal dollars alone.
- Expected and acceptable outcome: re-searching may LOSE to frozen — each
  re-search spends a fresh winner's curse. That result would set the live
  policy directly (deploy-and-hold; replace only on decisive evidence).
