"""The champion's generated feature bundle — RETIRED 2026-09-11, empty.

The 40 formulas adopted 2026-09-06 (git history of this file) were built on
the closeadj-basis raw inputs: r_close and the split-restated SF1 per-share
fields carried future splits (the split leak, reports/nominal_basis_registration.md;
the leaky champion failed ops/leak_audit.py at retention 0.49 on 2006-2015).
Their nominal-basis replacement, screened under the same rule on 2006-2015
alone, failed admission against the clean line (ledger nominal_program_verdict:
5.93 vs 8.14 %/yr at the cascade's book). The champion is the clean line: the
panel's f_ columns and nothing else. Keeping the module (empty) leaves the
plumbing — add_generated on FORMULAS, SPEC["features"] — in place for a bundle
that passes the gate later.
"""
from __future__ import annotations

FORMULAS: dict[str, str] = {}

FEATURES: list[str] = list(FORMULAS)          # the g_ columns, in bundle order
