# Harvey, Liu & Zhu (2016) — "…and the Cross-Section of Expected Returns"

PDF: `papers/harvey-liu-zhu-2016-cross-section.pdf` (author copy, Duke)

## What the paper shows

A census of 316 published return factors plus a multiple-testing framework
(Bonferroni, Holm, Benjamini–Hochberg–Yekutieli) for the whole *unobserved* set of
factors ever tried. Conclusion: given the accumulated data-mining across the
profession, a newly claimed factor needs **t ≳ 3.0** (not 1.96), the hurdle rises
over time, and roughly half the published cross-sectional literature is likely
false. Publication bias means even this understates the problem.

## What it implies for stocks_ml

- **Their argument is about *adoption thresholds*, and ours are informal.** Our
  feature-admission policy asks for ablation on identical folds and fold-stability,
  but sets no numeric hurdle. With 4 folds and ~488 weekly ICs, a paired
  comparison (mean weekly ΔIC / its standard error, using the week-by-week IC
  difference between with-feature and without-feature runs) gives exactly the
  t-statistic this paper wants ≥ 3. That is implementable in the existing ablation
  harness and turns "looks stable across folds" into a rule. For family-level
  decisions (e.g. "adopt the EDGAR bundle"), one test per *family*, not per
  feature, keeps our own N small.
- **We benefit from their false-discovery list.** Most of the 316 factors fail
  their hurdle. Before any future feature hunt, the paper's Table 6 / figure of
  surviving factor categories is the shopping list: the robust survivors are
  broad-category momentum, value, profitability/investment, and low-risk —
  precisely the families already in `panel.py` or proposed in these notes. The
  long tail of exotic anomalies is pre-filtered out for us; ignoring it saves
  trials, which keeps our deflated Sharpe honest.
- **The hurdle interacts with our trial ledger** (see Bailey–López de Prado note —
  same statistical machinery from the portfolio side). HLZ discipline what enters
  the *feature set*; DSR disciplines what the *strategy league* claims. Both need
  the same input we don't yet persist: a complete count of what was tried. One
  ledger serves both.
- **A sanity note on interpretation:** t ≥ 3 for *newly discovered* effects. Our
  core features are century-old published factors with out-of-sample decades —
  HLZ would not demand rediscovery at t≥3 for including book-to-market; the
  hurdle applies to *our* novel constructions (insider-flow variants, event
  recencies, anything home-grown) and to marginal-adoption decisions where the
  base rate of false positives is ours, not the literature's.

## Concrete candidates

1. Add a paired weekly-ΔIC t-statistic to the ablation report; adopt features/
   families only at t ≥ 3 (or BHY-adjusted p if several are tested at once).
2. Route every ablation through the trials ledger proposed in the DSR note so the
   count of home-grown hypotheses is auditable.
