# Bernard & Thomas (1989) — "Post-Earnings-Announcement Drift: Delayed Price Response or Risk Premium?"

PDF: `papers/bernard-thomas-1989-pead.pdf` (author-posted copy recovered from
Jake Thomas's archived Yale page; image scan, no text layer)

## What the paper shows

After sorting on **SUE** — standardized unexpected earnings, i.e. the seasonal
random-walk surprise (this quarter's earnings minus the same quarter last year)
scaled by the standard deviation of past surprises — abnormal returns keep
drifting in the surprise's direction for ~60 trading days: ≈6.3% annualized-25%
top-minus-bottom decile over the window, positive in 41 of 48 quarters. The
drift is a *delayed response*, not a risk premium: it is concentrated around the
**next quarter's announcement**, consistent with prices ignoring the
autocorrelation of seasonally-differenced earnings (nailed down in their 1990
companion). Effect is larger in small firms but present and exploitable in large
firms; risk-based explanations fail their tests.

## What it implies for stocks_ml

- **We trade the drift window but only see its price shadow.** `f_pead` in
  `features/events.py` is the announcement-window *price reaction*
  (close_hi/close_lo around the filing) — the literature's *ear*, which
  Green–Hand–Zhang confirm as a non-microcap survivor. Good. But Bernard–Thomas's
  actual sorting variable is the **earnings surprise itself**, and *ear* and SUE
  are imperfectly correlated (announcement-day price moves are contaminated by
  guidance, buybacks, market beta). We have everything needed for true SUE:
  quarterly earnings from EDGAR company facts, filing-dated (the next-calendar-day
  availability convention already implemented for `f_pead`). `f_sue` =
  (E_q − E_{q−4}) / σ(last 8 seasonal differences), neutral-filled per policy #3.
- **A 5-day horizon is well matched to drift trading.** The drift accrues over
  ~13 weeks with a burst at the next announcement; a weekly re-ranked model
  captures it as a persistent tilt. The monthly `label_4w` pipeline should see it
  even more cleanly (60-day drift ≈ 3 monthly labels).
- **The "next announcement" concentration is a feature interaction we can build.**
  Their sharpest result: the drift's largest chunk arrives in the 3-day window of
  the *following* earnings announcement, with the *sign of the last surprise*.
  We already compute `f_days_since_earnings_8k`; its interaction with `f_sue`
  (surprise sign × approaching-announcement) is learnable by XGB only if both
  exist. This is a rare case where the paper predicts *which* interaction should
  matter — a good post-ablation check on feature importances.
- **Coverage realism:** EDGAR fundamentals are sparse pre-2010, but our evaluation
  starts 2015-03 — SUE should have near-full coverage over the folds. Report the
  coverage stats per missing-data policy #2 anyway; missing σ (fewer than 8 prior
  quarters, fresh index joiners) is a structural absence worth a missingness
  indicator only if ablation says so.

## Concrete candidates

1. `f_sue` from EDGAR quarterly earnings (seasonal difference, scaled), filing-dated.
2. Optional second-order feature after SUE lands: `f_sue_lag1q` (last quarter's
   SUE) — their autocorrelation table says lags 1–3 predict with declining weight,
   lag 4 flips negative.
