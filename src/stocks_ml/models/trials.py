"""Trials ledger: the project's complete, append-only record of what was tried.

Bailey & López de Prado's deflated Sharpe needs two inputs the code previously
proxied: an honest count N of every strategy/config ever evaluated, and the
cross-trial variance of their Sharpe ratios (metrics.py used the single
winning path's estimator variance instead — under-deflating when trials were
diverse). Harvey–Liu–Zhu's t >= 3 adoption hurdle needs the same census from
the feature side. One git-tracked ledger serves both.

Producers append here: Optuna/random-search tuning (one row per trial),
the champion tournament (one per candidate), the pipeline league (one per
pipeline), the strategy-zoo backtest (one per strategy), and ablations.
Rows record pre-holdout evidence only; holdout results never enter the ledger.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np

LEDGER_PATH = "models/trials_ledger.json"


def load_ledger(path: str | Path = LEDGER_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def record_trials(entries: list[dict], path: str | Path = LEDGER_PATH) -> int:
    """Upsert entries (each: kind, name, and optional cv_metric /
    pre_holdout_sharpe / notes). Returns the new total count.

    Re-running an identical config is not a new trial: rows are keyed by
    (kind, name) and updated in place, so reruns keep the census honest
    instead of inflating N."""
    p = Path(path)
    ledger = load_ledger(p)
    index = {(r.get("kind"), r.get("name")): i for i, r in enumerate(ledger)}
    today = str(date.today())
    for e in entries:
        row = {"date": today, **e}
        for k, v in list(row.items()):
            if isinstance(v, np.bool_):
                row[k] = bool(v)
            elif isinstance(v, np.integer):
                row[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                row[k] = float(v) if math.isfinite(v) else None
        key = (row.get("kind"), row.get("name"))
        if key in index:
            ledger[index[key]] = row
        else:
            index[key] = len(ledger)
            ledger.append(row)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=1))
    tmp.replace(p)
    return len(ledger)


def ledger_stats(path: str | Path = LEDGER_PATH) -> tuple[int, float | None]:
    """(N, cross-trial variance of ANNUALIZED pre-holdout Sharpes).

    N counts every recorded trial. The variance uses only rows that carry a
    pre_holdout_sharpe (tuning trials record CV metrics, not Sharpes); with
    fewer than 3 such rows it returns None and deflated_sharpe falls back to
    its single-path proxy, stated in the report."""
    ledger = load_ledger(path)
    n = len(ledger)
    sharpes = [r["pre_holdout_sharpe"] for r in ledger
               if isinstance(r.get("pre_holdout_sharpe"), (int, float))]
    if len(sharpes) < 3:
        return n, None
    return n, float(np.var(sharpes, ddof=1))
