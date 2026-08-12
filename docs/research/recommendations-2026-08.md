# Top 3 changes the reading list implies (ranked by effort-to-impact)

Date: 2026-08-11 (updated same day with Hou–Xue–Zhang, Cohen–Malloy–Pomorski,
and the short-interest literature — the ranking survived; HXZ independently
corroborates #1 and #2). Derived from the thirteen notes in this directory; each
change cites the papers that back it. All three respect the fixed evaluation design (4 folds,
2015-03+, 10-day purge, untouchable holdout) and the missing-data policy.

## 1. Extend the momentum block past 12 weeks (skip-adjusted)

**Backed by:** Novy-Marx 2012 (intermediate horizon is the strong, large-cap-robust
part), Jegadeesh–Titman 1993 (skip the most recent week), Gu–Kelly–Xiu 2020
(multi-horizon price trends are the top predictor family), Hou–Xue–Zhang 2020
(momentum is the largest anomaly category surviving microcap screens — the
closest published analogue to our S&P 500-only universe).

**Change:** add to `features/panel.py::price_features`, rank-normalized like the rest:
- `f_mom_12w_skip1w` = close(t−5d)/close(t−60d) − 1
- `f_mom_26w` = close(t−5d)/close(t−130d) − 1
- `f_mom_52w_skip4w` = close(t−20d)/close(t−252d) − 1 (classic 12-1)
- `f_mom_interm` = close(t−130d)/close(t−252d) − 1 (Novy-Marx's t−12m…t−7m)

Ablate as one family on identical folds; adopt at paired ΔIC t ≥ 3 (see #3).

**Why top-ranked:** hours of work, zero new data dependencies, full history (no
coverage caveats), and it fills the panel's clearest documented gap — the current
longest lookback (12 weeks) sits entirely inside what the literature calls the
*weak* recent-horizon window. Project history says new information is what moves
IC; this is the cheapest new information available.

## 2. SUE + EDGAR survivors feature bundle

**Backed by:** Bernard–Thomas 1989 (SUE drift, concentrated at the next
announcement), Green–Hand–Zhang 2017 (non-microcap survivors: earnings-announcement
return — already have as `f_pead` — plus net share issuance, R&D/mktcap, nincr),
Novy-Marx 2013 (fundamentals pay at the 4-week horizon → test on `label_4w` too),
Hou–Xue–Zhang 2020 (SUE/earnings momentum and net issuance sit in their surviving
categories under value weights).

**Change:** from data already ingested (EDGAR company facts, filing-dated with the
existing next-calendar-day convention):
- `f_sue` = (E_q − E_{q−4}) / σ(last 8 seasonal diffs)
- `f_net_issuance` = YoY % change in shares outstanding
- `f_nincr` = count of consecutive quarterly earnings increases
- `f_rd_to_mktcap` (if R&D tags have usable coverage — report per policy #2)

Neutral-fill after ranking (policy #3); ablate family-wise on the weekly label
*and* `label_4w`.

**Why second:** moderate effort (fundamentals plumbing exists, but PIT care and
coverage reporting take real work), backed by the strongest multiple-testing-
surviving evidence on the list, and it feeds the monthly pipeline where our
fundamentals should differentially shine.

## 3. Honest trial accounting: trials ledger + adoption hurdle + corrected DSR

**Backed by:** Bailey–López de Prado 2014 (DSR needs cross-trial variance and a
complete N — `metrics.py:71` currently proxies with single-path estimator
variance), Harvey–Liu–Zhu 2016 (t ≥ 3 for home-grown effects), Timmermann 2006
(pre-declare that combination gains show up as stability, not mean IC, so they
aren't judged by cherry-picked means).

**Change:**
- Git-tracked trials ledger auto-appended by `tune` / `train` / `pipelines`
  (config hash, fold ICs, backtest Sharpe). Refit-anchor variants count as trials.
- `deflated_sharpe` takes (N, empirical cross-trial SR variance) from the ledger.
- Ablation harness reports the paired weekly-ΔIC t-statistic; families adopted at
  t ≥ 3.
- Holdout section of `reports/backtest.md` reports DSR next to Sharpe (iron rule
  #0 already makes holdout the headline; this makes the headline honest).

**Why third:** smallish code, but the impact is defensive — it doesn't raise IC,
it prevents the account-blowing-up failure mode the owner cares most about, and
both #1 and #2 become more trustworthy once their trials run through it. Ranked
below the feature work only because the features can produce alpha and this can't;
if the league keeps growing, this one becomes #1.

## Runners-up (worth one ablation each, in this order, when the top 3 land)

- **Short-interest changes** (`f_short_chg_8w`, abnormal-level variant) — no new
  ingestion; the level/DTC pair already earns +0.002 IC and BJZ's mechanism says
  changes carry the information (see short-interest note; publication lag caps
  the upside).
- **Opportunistic-insider features** — requires carrying `owner_cik` through the
  Form 4 ingest, then reuses existing windowed-sum machinery; retry the nulled
  insider family at `label_4w` per Cohen–Malloy–Pomorski.
- **RankBlend(xgb, enet) tournament candidate** — fixed 50/50 rank average,
  zero hyperparameters (Timmermann); judged on fold-stability, not mean IC.
- **What NOT to spend trials on** (Hou–Xue–Zhang veto): expanding the liquidity/
  idio-vol family as alpha, distress measures, and any anomaly that fails their
  value-weighted replication.
