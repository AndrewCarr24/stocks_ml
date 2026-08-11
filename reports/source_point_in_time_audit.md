# FRED and Wikipedia point-in-time audit

Audited 2026-07-21 before restoring any excluded source to production.

## FRED

The current FRED snapshot was compared with quarterly ALFRED vintages. A series
was restored only when sampled historical vintages matched today's values and
its configured availability delay was conservative for its frequency.

| series | common vintage comparisons | changed | decision |
|---|---:|---:|---|
| T10Y2Y | 204,839 | 0 | restore; daily series delayed one calendar day |
| FEDFUNDS | 12,043 | 0 | restore; monthly observation lag corrected from 1 to 35 days |
| VIXCLS | 234,120 | 272 | exclude |
| DTB3 | 237,332 | 248 | exclude |
| ICSA | 49,465 | 12,706 | exclude |
| CPIAUCSL | 11,316 | 3,463 | exclude |
| UNRATE | 11,316 | 907 | exclude |
| UMCSENT | 11,378 | 16 | exclude |
| DTWEXBGS | 126,859 | 64,045 | exclude |

`T10Y2Y` ALFRED archive snapshots were available from 2014 onward; its source is
the daily Treasury yield spread, and no sampled vintage differed. `FEDFUNDS`
had archive coverage through the full modeling history and no sampled revision.
Its FRED observation is the monthly average dated on the first day of the month,
so the former one-day lag was invalid. Thirty-five days defers availability to
the following month. Raw excluded series remain stored for diagnostics.

## Wikipedia

The historical changes table supplies effective dates for additions/removals,
and the current table supplies the effective addition date of each active stint.
The audit found that the parser had discarded the latter, causing many active
constituents to fall back to 1996. The parser now preserves that date and uses it
as authoritative for the active stint. The current table supplies only today's
GICS sector for a ticker; it does not provide effective-dated historical classifications. Sector
dummies and sector-relative momentum therefore remain excluded until an
effective-dated sector source is available.

## Production decision

Restore `f_macro_T10Y2Y`, `f_macro_T10Y2Y_chg`, `f_macro_FEDFUNDS`, and
`f_macro_FEDFUNDS_chg`. Keep every other `f_macro_` feature and every `f_sec_`
or `_sect` feature out of model matrices.

## Identical-calendar A/B before retuning

Using the previously tuned recipes on the corrected 488-week calendar:

| model | macro variant | mean IC | coverage |
|---|---|---:|---:|
| XGBoost | none | 0.021019 | 488/488 |
| XGBoost | T10Y2Y | 0.008973 | 488/488 |
| XGBoost | FEDFUNDS | 0.009769 | 488/488 |
| XGBoost | both | 0.012989 | 488/488 |
| ElasticNet | none | 0.018402 | 488/488 |
| ElasticNet | T10Y2Y | 0.018389 | 488/488 |
| ElasticNet | FEDFUNDS | 0.018404 | 488/488 |
| ElasticNet | both | 0.018392 | 488/488 |

Point-in-time safety and predictive usefulness are separate questions. The
owner requested that safe data be restored, so both audited series remain in
the production matrix despite the frozen-recipe A/B.

## Final retraining after membership correction

Refreshing Wikipedia membership reduced the panel from 560,535 to 547,944 rows.
Both model families were then retuned from scratch. XGBoost remained champion at
0.019754 mean IC (folds 0.006103, 0.020595, 0.017985, 0.034332); ElasticNet
scored 0.019041 (folds 0.015292, 0.021265, 0.022410, 0.017195). Both covered all
488 evaluation weeks and all fold ICs were positive.