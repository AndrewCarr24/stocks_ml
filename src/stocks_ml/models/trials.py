"""Trials ledger: the project's complete, append-only record of what was tried.

Every configuration the selection procedure evaluates lands here (one row per
trial, upserted by name), so the count N of everything ever tried — the
input a deflated Sharpe or a t >= 3 adoption hurdle needs — is a census, not
a guess. Rows record pre-holdout evidence only; holdout results never enter
the ledger. Rows from the retired legacy pipeline (tuning trials, the
champion tournament, the pipeline league, the strategy zoo, ablations) stay
in the file as history.
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
