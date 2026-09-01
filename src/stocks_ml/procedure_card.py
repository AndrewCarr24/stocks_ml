"""Render PROCEDURE.md (the procedure card) from models/champion_spec.json.

The spec file is the single durable source of truth for the champion's
settings. Edit the spec, rerun `stocks-ml procedure-card`; never edit
PROCEDURE.md by hand.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

SPEC_PATH = Path("models/champion_spec.json")
CARD_PATH = Path("PROCEDURE.md")

TEMPLATE = """\
# Procedure card

Blueprint of the production procedure (generated {today} by
`stocks-ml procedure-card` from models/champion_spec.json — edit the spec,
not this file). Rationale and history: AGENTS.md.

## Current champion

| Component | Spec |
|---|---|
| Model | {model_summary} |
| Prediction target | {horizon_label}: stock's {hold_weeks}-week return minus that week's median member's ({purge_days}-day purge) |
| Training | weekly refit on trailing {train_years} years; early stop on validation rank correlation |
| Ensemble | K={k_copies} copies (random_state + whole-week bootstrap), predictions averaged |
| Book | top-{book_size}, equal weight, {sleeves} staggered sleeves rotating weekly, {hold_weeks}-week holds; weekly re-leveling; max {sector_cap}/sector (blocked slots to next-ranked other-sector name); no stop (audited: adds nothing over the ballast) |
| Ballast | {mix}: ballast in SPY, shifted to IEF one-third per breached trailing MA (30/40/52w) |
| Honest expectation | {honest_expectation} |

## Cadences

| Activity | When | Human involvement |
|---|---|---|
| Refit + rotate one sleeve | weekly (Friday decision, Monday open trade, {cost_bps} bps) | none |
| Re-level weights / check stops / ballast state | weekly | none |
| Full re-selection (any layer) | **never on a calendar** — structural triggers only: {triggers} | pre-registered, owner-approved |

## Selection procedure (how each component is chosen)

{procedure_note}

| step | menu | decided by |
|---|---|---|
{procedure_rows}

Metric convention: {metric_convention}. Measured selection inflation of this
procedure: {inflation}.

## Standing rules

- {holdout_start}+ is holdout: {holdout_status}.
- Champion changes are owner-approved and recorded here + in the ledger;
  doubts become pre-registered falsification tests, never quiet overrides.
- Every evaluated config enters models/trials_ledger.json.
- All pre-holdout numbers carry design-iteration shine; treat accordingly.
"""


def render(spec: dict, today: str | None = None) -> str:
    s = spec
    return TEMPLATE.format(
        today=today or str(date.today()),
        model_summary=s["model"]["summary"],
        horizon_label=s["horizon"]["label"],
        purge_days=s["horizon"]["purge_days"],
        train_years=s["training_window_years"],
        k_copies=s["ensemble"]["k_copies"],
        book_size=s["strategy"]["book_size"],
        sleeves=s["strategy"]["sleeves"],
        hold_weeks=s["strategy"]["hold_weeks"],
        sector_cap=s["strategy"]["sector_cap"],
        mix=s["ballast"]["mix"],
        honest_expectation=s["honest_expectation"],
        cost_bps=s["costs_assumed_bps_oneway"],
        triggers="; ".join(s["retune_policy"]["triggers"]),
        holdout_start=s["holdout_start"],
        holdout_status=s["holdout_status"],
        procedure_note=s["selection_procedure"]["note"],
        procedure_rows="\n".join(
            f"| {st['step']} | {st['menu']} | {st['metric']} |"
            for st in s["selection_procedure"]["steps"]),
        metric_convention=s["selection_procedure"]["metric_convention"],
        inflation=s["selection_procedure"]["measured_selection_inflation"],
    )


def write_card(spec_path: Path = SPEC_PATH, card_path: Path = CARD_PATH) -> str:
    spec = json.loads(spec_path.read_text())
    text = render(spec)
    card_path.write_text(text)
    return text
