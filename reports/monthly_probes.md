# Monthly pipeline robustness probes (2026-08-12)

Question (owner): is monthly_reg's since-2005 $10,876 (Sharpe 0.82) real, or
a bug/leakage artifact?

Method: one prediction walk (276 fits, identical to the league's monthly row),
three simulations. Baseline must reproduce the league; the other two stress the
distressed-rebound trade identified in the April-2009 autopsy (picks AIG, C,
FITB, HBAN, LNC, RF, F, CBRE at $1.86–$23 after −64%…−97% trailing years).

| probe | since-2005 | Sharpe | max DD | holdout |
|---|---|---|---|---|
| baseline (league row) | $10,876 | 0.82 | 65.4% | $145 |
| + empirical removal haircuts | $10,876 | 0.82 | 65.4% | $145 |
| + $5 minimum price on picks | **$913** | **0.65** | 55.2% | $139 |
| spy_hold (reference) | $930 | 0.65 | 55.2% | $139 |

Readings:

- **No mechanical leakage found** (separately verified: fold and inner
  early-stop purges are 42 days > the 29-day label span; CV IC 0.038 at 4 weeks
  ≈ 0.019 weekly-equivalent — champion-grade, not bug-grade).
- **The torture probe is uninformative here, not exonerating.** Measured
  removal events have median post-removal returns ≥ 0, so every haircut class
  resolves to 0.00% — the removal-exit channel it measures is clean, but it
  cannot see companies missing from the panel entirely (~200 delisted
  tickers), which is where the real bias lives.
- **The $5 price floor is the verdict: the entire excess over SPY comes from
  sub-$5 stocks.** Floored, the pipeline is statistically indistinguishable
  from buying the index ($913 vs $930, Sharpe 0.65 vs 0.65). The
  distressed-rebound trade that produced 2009 +87%, 2010 +100%, 2020 +102% is
  precisely the segment where (a) survivor-only data omits the total losses
  (Lehman/WaMu-class outcomes), and (b) the 5bps cost model is least
  realistic (2009 penny-stock spreads were orders of magnitude wider).

Conclusion: monthly_reg's headline is an artifact of survivor-only distressed
names plus optimistic penny-stock frictions, not an exploitable edge. Treat the
monthly pipeline as unproven until delisted-inclusive price data (see AGENTS.md
open items) allows an honest re-measurement. The weekly LTR challenger remains
the only pipeline with a live claim against SPY.
