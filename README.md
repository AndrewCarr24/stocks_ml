# stocks_ml

`stocks_ml` ranks S&P 500 stocks on a one-month horizon with a small gradient-
boosted model, wraps the ranking in a rules-based long-only book with an
index/Treasury ballast, and runs that system every week as a $100 paper
portfolio. Its selection procedure is mechanical and pre-registered, its data
is point-in-time, and the last two years are a sealed holdout.

There is no broker integration. The weekly job proposes a book; a human
places the orders.

## The champion

The production system is `r5` — declared 2026-09-01 after a month-horizon
rebuild, specified in [models/champion_spec.json](models/champion_spec.json)
and summarized in [PROCEDURE.md](PROCEDURE.md). Its current form is the
package that came out of the clean-improvement program's Stage E, adopted
2026-09-12 ([reports/clean_improvement.md](reports/clean_improvement.md)).

| Component | Spec |
|---|---|
| Model | depth-3 XGBoost, untuned by design (hyperparameter search measured as noise) |
| Target | the stock's 4-week open-to-open return minus the same-week median of its sector; 35-day purge |
| Training | weekly refit on the trailing 8 years; early stopping on a purged, time-ordered tail's weekly rank correlation |
| Ensemble | K=16 copies (seed + whole-week bootstrap), predictions averaged (K=4 until 2026-09-07) |
| Features | the point-in-time panel's 64 standing columns (prices, fundamentals, filings, insider, short interest, macro) on the nominal per-share basis; no engineered bundle (the earlier bundles were retired 2026-09-11 when a split-adjustment leak was found in their inputs) |
| Book | top-10, equal weight, four staggered sleeves — one rotates each week, so every name is held four weeks; no sector cap; no stop-loss |
| Ballast | half-gate: the book holds 100 / 83 / 67 / 50% of NAV as 0 / 1 / 2 / 3 of SPY's trailing means (30 / 40 / 52 weeks) are breached; the rest sits in IEF |
| Costs | 5 bp one-way, fills at Monday's open |

The model fields (label, window) and the strategy layers (book, ballast,
stop, cap) are written into the spec by `stocks-ml procedure` from the
walk that decided them; tests fail on a hand edit of either the spec or
the live job.

**Record, K=16, graded as deployed (costs included), 2006-01 → 2024-07-18:**
$100 → **$1,994**, +17.5%/yr, Sharpe 0.70, max drawdown 63% — vs SPY $621
on the same weeks, +10.4%/yr, Sharpe 0.65, drawdown 55%
([reports/champion_vs_sp500_2006_2024.png](reports/champion_vs_sp500_2006_2024.png)).
On 2016-01 → 2024-07, where the label and window were not chosen, $660,
+24.6%/yr, Sharpe 0.88, drawdown 33% vs SPY $316, +14.3%/yr, 0.87, 32%
([reports/champion_vs_sp500_2016_2024.png](reports/champion_vs_sp500_2016_2024.png));
on 2006–2015, the selection window, $302 vs SPY $197. The excess over SPY
on 2016–2024 is +9.0%/yr with a 95% interval of −3.9 to +24.2 when both
the model's seeds and the history are resampled; the chance the edge is
real is about 0.91 ([reports/clean_improvement.md](reports/clean_improvement.md)).

Read it with the spec's own caveat. The edge is era-concentrated: the book
buys high-volatility, beaten-down names, so its good years are rebound
years (2009, 2016, 2019–2021) and it is index-like in 2010–2015 and
2022–2024; it fell further than SPY in 2008–09 (63% vs 55%). Every
pre-holdout number carries design-iteration shine — the label and window
were chosen among 15 candidates on 2006–2015, and 2016–2024 was read once,
for this grade. Sizing should assume SPY-like outcomes in adverse regimes.
The holdout (2024-07-19 onward) has not been graded; it is a single-use
exam. Every figure is graded by the live job's own rules: orders fill at
the next session's open, 5 bp a side on every trade.

Before 2026-09-12 the champion was the same clean model with the
week-centred label, a 5-year window, top-6, sector cap 2 and a fixed
book/ballast split: $963 vs SPY $621 on 2006 → 2024-07, $420 vs $316 on
2016 → 2024-07.
Before 2026-09-11 it also trained on a generated bundle of 40 features
whose record ($4,256 on 2006 → 2024-07) was contaminated by the
split-adjustment leak; those numbers are history and are kept in
[AGENTS.md](AGENTS.md) and the ledger for scale only.

### How it was chosen

The procedure is a fixed cascade run on the selection window (2006–2015)
only, one decision per layer, each by a metric declared in advance:

| step | menu | decided by |
|---|---|---|
| horizon | 1w vs 4w | cost-adjusted compounded return of the top-6 book |
| training window | 1–5 years | top-6 edge vs a random basket on identical weeks |
| label × window × fit (clean program, 2026-09-11) | 15 variants of the clean line — week-centred / sector-centred / gauss-rank labels, 3 / 5 / 8-year windows, fixed rounds, learning rate, depth, column and leaf settings — ranked on a 131-week sample; the top four plus their pairings walked on every week of 2006–2015 at K=4 | cost-adjusted compounded %/yr of the top-6 book (`ops/clean_program.py`) — the sector label on an 8-year window won, +7.57 vs the incumbent's +5.44 |
| book size | top-3 / 6 / 10 | cost-adjusted compounded return — top-10 |
| stagger | always on | mechanism, not searched |
| ballast | none / half-gate / 80-20 / 70-30 / 60-40 | Sharpe — half-gate |
| stop-loss | off / −25% | Sharpe (adopt only if higher) — off |
| sector cap | off / 2-of-book | Sharpe (adopt only if higher) — off |
| feature screen | candidate columns probed on the selection window | keepers examined as one bundle; admitted iff the top-6 book compounds faster with it — the clean program's 16 admitted features were dropped by the owner after the 2016–2024 read, so the champion carries none |

`stocks-ml procedure --preds <walk>` runs the strategy layers on a saved
K=16 walk of the selection window and writes the result into the spec;
`stocks-ml select` runs the older end-to-end cascade. The cascade was
validated by a nested test
([reports/nested_selection_protocol.md](reports/nested_selection_protocol.md)):
run on 2006–2015 alone it chose a near-champion configuration whose frozen
grade on 2016 → 2024-07 was $521 vs SPY $316 (on the split-leaky inputs of
the time). The Stage E package was held to the same standard: a
pre-registered falsification test (paired weekly excess over the previous
champion on 2016–2024, t < −2 rejects) read t +1.3, and a leak audit
passed on both walk segments before adoption. Every configuration ever
evaluated is in [models/trials_ledger.json](models/trials_ledger.json).
The champion is never re-tuned on a calendar: nested experiments showed
calendar re-selection at any cadence adds drawdown without reliable return.
Re-selection happens only on a structural trigger (a new data source passes
its gate, a pre-registered kill criterion fires, or the owner directs it).

## The weekly job

[`.github/workflows/champion.yml`](.github/workflows/champion.yml) is the only
workflow. Every Saturday 13:00 UTC it runs `stocks-ml r5-weekly --commit`
([src/stocks_ml/live/r5.py](src/stocks_ml/live/r5.py)):

1. **Refresh the live world** ([data/world.py](src/stocks_ml/data/world.py)):
   Sharadar S&P 500 membership, prices (incremental by `lastupdated`, upserted
   by ticker and date), fundamentals (full refetch, refuses to shrink), insider
   filings; then SEC EDGAR company facts and 8-Ks, FINRA short interest and
   FRED. Symbol renames are detected by permanent ticker id and rewritten in
   every stored table so history carries over.
2. **Rebuild the panel** with the research recipe (the same code the
   research walks use, including the sector-relative label), then fit the
   K=16 ensemble on the spec's label and trailing window (8 years) and rank
   every current member.
3. **Rotate the due sleeve** (the schedule is anchored so a rerun on the same
   Friday is a no-op), set the ballast state from SPY's trailing means, and
   write target weights.
4. **Keep the paper ledger** ([ledger_r5.json](ledger_r5.json)): fill last
   week's targets at Monday's open, sells before buys, 5 bp each way, never
   overdrawn, rebalances under 0.5% of NAV skipped; then mark NAV at Friday's
   close against SPY buy-and-hold from the same $100.
5. **Commit** `signals_r5/<friday>.md` (the book with per-name deltas, the
   sleeves, the top-15 candidates, data freshness) and the ledger, as
   `r5: signal <friday>`. The report is also on the run's summary page.

A run fails loudly — the panel must end on the most recent Friday — rather
than signal on stale data. Manual runs (`workflow_dispatch`) accept `as_of`
and `dry_run`.

**Licensed data never enters this public repository.** The Sharadar key is the
`SHARADAR_API_KEY` repository secret. The live data store (about 0.5 GB)
travels between runs as an AES-256 tarball in the Actions cache, encrypted
with the `R5_STORE_KEY` secret; a snapshot is saved only after a successful
run, and a Wednesday keepalive job stops GitHub's 7-day idle eviction. When
the cache is empty, the run seeds itself from a draft release uploaded by
[ops/r5_seed.sh](ops/r5_seed.sh) (drafts are invisible to the public) and
deletes it afterwards. [ops/r5_weekly.sh](ops/r5_weekly.sh) runs the same
cycle on a Mac by hand.

## Data

- **Sharadar** (Nasdaq Data Link, licensed; key required): S&P 500
  constituents with join/leave dates from 1998, daily prices for every
  historical member including delisted ones (dividend-adjusted opens and
  closes), quarterly fundamentals dated by filing, and insider transactions.
  This is what makes the universe survivorship-clean; the earlier free-data
  world was missing about 200 delisted constituents, and the edge it showed
  turned out to be a data artifact.
- **SEC EDGAR** company facts and 8-K filings, **SEC Form 4** insider filings
  (the Sharadar insider table bridges the gap after the last quarterly SEC
  file), **FINRA** short interest, **FRED** macro series (only the ALFRED-
  audited `T10Y2Y` and `FEDFUNDS` families are admitted as features).

Point-in-time rules: membership is effective-dated; filings become usable the
next calendar day; FINRA and macro observations carry their publication lags;
features are rank-normalized within each week; the label starts at the next
open and records when it becomes observable; training, early stopping and
validation are separated by purge gaps sized to the label horizon. The
no-lookahead tests corrupt future inputs and require past outputs to stay
unchanged — if they fail, fix the implementation, not the tests.

## Installation

Python 3.12 — what the champion is locked, graded and run on. Eight runtime
dependencies (pandas, numpy, pyarrow, pyyaml, requests, scikit-learn, xgboost,
scipy).

```bash
uv sync
uv run pytest        # must stay green with zero warnings
```

Running the champion locally needs `data/.sharadar_key` (or
`SHARADAR_API_KEY`) and a live store under `data/r5_live/`; `data/` is
git-ignored in full.

```bash
uv run stocks-ml r5-weekly [--as-of FRIDAY] [--no-refresh] [--no-sec] [--dry-run] [--commit]
uv run stocks-ml select --sel-start A --sel-end B [--eval-start C --eval-end D] [--screen]
uv run stocks-ml procedure --preds <K=16 walk>/preds.parquet [--check]   # decide + write the spec's layers
uv run stocks-ml procedure-card      # regenerate PROCEDURE.md from the champion spec
```

All commands read [config/config.yaml](config/config.yaml).

## History

The repository began as a one-week-horizon system on free data; that
pipeline was retired on 2026-09-01 and removed from the tree on 2026-09-02.
Its code, reports and paper ledgers are at the tag `legacy-final`, and the
reasons it lost are recorded in [AGENTS.md](AGENTS.md).

## Where to look

- [AGENTS.md](AGENTS.md) — the full project context: every campaign, verdict
  and rule, in order.
- [PROCEDURE.md](PROCEDURE.md) — the champion's procedure card (generated
  from the spec).
- [models/trials_ledger.json](models/trials_ledger.json) — every evaluated
  configuration.
- [reports/](reports/) — the clean-improvement program (registration and
  report), the nested-selection protocol, the point-in-time source audit,
  the regrades (rank date, fill basis, the engineered bundle) and the
  champion-vs-SPY charts.
- [docs/research/](docs/research/) — notes on the papers the design leans on.
