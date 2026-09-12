"""`stocks-ml procedure`: the backtest procedure writes the deployed strategy.

The strategy layers of models/champion_spec.json — book size, ballast floor,
stop-loss, sector cap — are never typed. This module reads a saved K-copy
walk over the selection window, ranks it exactly as the live job ranks
(mean over copies), runs selection.decide_strategy (book by cost-adjusted
compounded %/yr; floor, stop, cap by Sharpe) and writes the argmax into the
spec together with a `procedure` block recording what it read. tests/
test_procedure.py holds spec.strategy to spec.procedure.decision and the
live job's SPEC to spec.strategy, so a hand edit of either fails the suite.

    stocks-ml procedure --preds data/experiments/<walk>/preds.parquet
    stocks-ml procedure --preds ... --check     recompute; fail if the spec drifted

Rules the command enforces: every rank week of the selection window must be
in the walk (no frozen decision reads a sample); no week at or past the
holdout is read; the walk carries all K copies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

SPEC_PATH = Path("models/champion_spec.json")
SELECT = (pd.Timestamp("2006-01-01"), pd.Timestamp("2015-12-31"))
STORE = "data/sharadar_world2000_nominal_dl"
HORIZON = "4w"
DECISION_KEYS = ("book_size", "floor", "stop_loss", "sector_cap")
MODEL_KEYS = ("label", "train_years")      # the walk's recipe: what the spec's model fields are


def walk_recipe(preds_path: Path) -> dict:
    """The model that made the walk, from the record beside it
    (<walk dir>/spec.json, `recipe`: the label column and training window the
    walk's copies were trained on -- ops/clean_program.py writes it under a
    guard, so a walk cannot be resumed under another recipe). The spec's
    horizon.label and training_window_years come from here and are never
    typed; a walk whose record names no recipe is refused."""
    from stocks_ml.selection import LABELS_4W
    rec_path = Path(preds_path).parent / "spec.json"
    if not rec_path.exists():
        raise RuntimeError(f"{rec_path} is missing: the procedure reads the walk's recipe "
                           "(label, train_years) from the record beside the walk")
    rec = json.loads(rec_path.read_text()).get("recipe")
    if not rec or any(k not in rec for k in MODEL_KEYS):
        raise RuntimeError(f"{rec_path} names no recipe {MODEL_KEYS}: the procedure cannot tell "
                           "what model made the walk, so it refuses to write the spec's model fields")
    if rec["label"] not in LABELS_4W:
        raise RuntimeError(f"the walk was trained on {rec['label']!r}, which live/r5.py cannot "
                           f"train on (selection.LABELS_4W knows {tuple(LABELS_4W)})")
    return {"label": rec["label"], "train_years": int(rec["train_years"]),
            "record": str(rec_path)}


def mix_label(floor: str) -> str:
    """ballast.mix as the spec spells it, from the cascade's floor name."""
    from stocks_ml.ledger import FLOOR_FRACTION
    if floor == "none":
        return "100% book / no ballast"
    if floor == "halfgate":
        return "halfgate: book 100/83/67/50% of NAV by SPY trend gates down (30/40/52w), rest IEF"
    pct = round(FLOOR_FRACTION[floor] * 100)
    return f"{pct}% book / {100 - pct}% ballast"


def live_floor(spec: dict) -> str:
    """The floor live/r5.py hands ledger.floor_split each week, from the spec:
    the menu entry's name. Every entry on the menu runs live (halfgate since
    2026-09-12); anything else is refused rather than deployed ungraded."""
    from stocks_ml.ledger import FLOORS
    floor = spec["procedure"]["decision"]["floor"]
    if floor not in FLOORS:
        raise RuntimeError(f"the procedure decided floor {floor!r}, which live/r5.py cannot "
                           f"express (ledger.floor_split knows {FLOORS}); implement it in "
                           "the live job before deploying")
    assert spec["ballast"]["mix"] == mix_label(floor), \
        f"ballast.mix {spec['ballast']['mix']!r} != procedure floor {floor!r}: run stocks-ml procedure"
    return floor


def live_strategy(spec: dict) -> dict:
    """The strategy fields live/r5.py runs on, read from the spec — never
    typed into the job. Fails when the spec's strategy block and the
    procedure's decision disagree (someone edited one by hand)."""
    dec = spec["procedure"]["decision"]
    st = spec["strategy"]
    for k in DECISION_KEYS:
        if k == "floor":
            continue
        if st[k] != dec[k]:
            raise RuntimeError(f"strategy.{k}={st[k]!r} but procedure.decision.{k}={dec[k]!r}: "
                               "the spec was edited by hand; run stocks-ml procedure")
    if st["stop_loss"] is not None:
        raise RuntimeError(f"the procedure decided stop_loss {st['stop_loss']}, which live/r5.py "
                           "does not implement; implement it before deploying")
    model = spec["procedure"].get("model")
    if model is None:
        raise RuntimeError("the spec's procedure block names no model (label, train_years): "
                           "run stocks-ml procedure on the champion's walk")
    have = {"label": spec["horizon"]["label"], "train_years": spec["training_window_years"]}
    for k in MODEL_KEYS:
        if have[k] != model[k]:
            raise RuntimeError(f"spec {k}={have[k]!r} but procedure.model.{k}={model[k]!r}: "
                               "the spec was edited by hand; run stocks-ml procedure")
    from stocks_ml.selection import LABELS_4W
    if model["label"] not in LABELS_4W:
        raise RuntimeError(f"live/r5.py cannot train on the spec's label {model['label']!r} "
                           f"(selection.LABELS_4W knows {tuple(LABELS_4W)})")
    return {"horizon": HORIZON, "label": model["label"], "train_years": int(model["train_years"]),
            "book": int(st["book_size"]), "cap": st["sector_cap"], "floor": live_floor(spec)}


def load_walk(preds_path: Path, k: int, weeks: list, lo, hi) -> pd.DataFrame:
    """The saved walk on [lo, hi]: all K copies, every rank week of the
    window present, nothing at or past the holdout."""
    from stocks_ml.selection import HOLDOUT_START
    preds = pd.read_parquet(preds_path)
    preds["week"] = pd.to_datetime(preds["week"])
    cols = [f"c{c}" for c in range(1, k + 1)]
    missing = [c for c in cols if c not in preds.columns]
    if missing:
        raise RuntimeError(f"{preds_path} lacks copies {missing}: the procedure reads K={k}")
    if (preds.week >= HOLDOUT_START).any():
        raise RuntimeError(f"{preds_path} carries weeks at or past the holdout "
                           f"{HOLDOUT_START.date()}; the procedure never reads them")
    preds = preds[(preds.week >= lo) & (preds.week <= hi)]
    want = [t for t in weeks if lo <= t <= hi]
    have = set(preds.week.unique())
    absent = [t for t in want if t not in have]
    if absent:
        raise RuntimeError(f"{preds_path} has {len(have)} of the {len(want)} rank weeks of "
                           f"{lo.date()} -> {hi.date()} (first missing {absent[0].date()}); "
                           "no frozen decision reads a sample — finish the walk")
    return preds[["week", "ticker", *cols]].sort_values(["week", "ticker"]).reset_index(drop=True)


def decide(preds_path, store=STORE, k=None, lo=SELECT[0], hi=SELECT[1], log=print) -> dict:
    """Rank the walk as live ranks and run the cascade's strategy layers on
    the selection window. Returns the decision, its evidence and the record
    of what was read (the spec's `procedure` block)."""
    import stocks_ml.selection as sel
    k = k or sel.K_COPIES
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    ctx = sel.Ctx(store)
    ctx.extra = []
    preds_path = Path(preds_path)
    model = walk_recipe(preds_path)
    preds = load_walk(preds_path, k, ctx.weeks, lo, hi)
    log(f"procedure: {preds_path}, K={k}, {preds.week.nunique()} rank weeks "
        f"{preds.week.min().date()} -> {preds.week.max().date()} on {store} "
        f"(price_basis {ctx.cfg.price_basis}, delist_labels {ctx.delist_labels}); "
        f"walk recipe {model['label']} / {model['train_years']}y from {model['record']}")
    hold, _ = sel.ensemble_holdings(ctx, preds, range(1, k + 1), HORIZON)
    hold = hold.sort_values("week").reset_index(drop=True)
    layers = sel.decide_strategy(ctx, hold, HORIZON, lo, hi)
    series = sel.simulate(ctx, hold, HORIZON, layers["book"], layers["cap"], layers["stop"],
                          layers["floor"])
    grade_hi = hi + pd.Timedelta(days=1)
    spy = ctx.wret["SPY"].reindex(series.index)
    return {
        "code": "stocks_ml.procedure.decide -> selection.decide_strategy (stocks-ml procedure)",
        "registration": "selection_procedure (this file): book by cost-adjusted compounded %/yr; "
                        "floor, stop, cap by Sharpe of the simulated weekly series, each at the "
                        "picks above it; every layer reads every rank week of the selection window",
        "selection_window": [str(lo.date()), str(hi.date())],
        "world": store,
        "price_basis": ctx.cfg.price_basis,
        "delist_labels": ctx.delist_labels,
        "horizon": HORIZON,
        "model": model,
        "k_copies": k,
        "preds": {"path": str(preds_path),
                  "sha256": hashlib.sha256(preds_path.read_bytes()).hexdigest(),
                  "rank_weeks": int(preds.week.nunique()),
                  "first": str(preds.week.min().date()), "last": str(preds.week.max().date()),
                  "ranked_weeks": int(len(hold))},
        "decided_at": str(pd.Timestamp.now().floor("s")),
        "decision": {"book_size": layers["book"], "floor": layers["floor"],
                     "stop_loss": layers["stop"], "sector_cap": layers["cap"]},
        "evidence": layers["evidence"],
        "selection_window_record": {"config": sel.metrics(series, lo, grade_hi),
                                    "sp500": sel.metrics(spy, lo, grade_hi)},
    }


def apply(spec: dict, proc: dict) -> dict:
    """Write the decision into the spec's strategy and ballast blocks, the
    walk's recipe into its model fields (horizon.label, purge_days,
    training_window_years), and attach the procedure record. Prose fields are
    left alone."""
    from stocks_ml.selection import HORIZONS
    dec = proc["decision"]
    spec["horizon"]["label"] = proc["model"]["label"]
    spec["horizon"]["purge_days"] = HORIZONS[HORIZON]["purge"]
    spec["training_window_years"] = proc["model"]["train_years"]
    spec["strategy"]["book_size"] = dec["book_size"]
    spec["strategy"]["stop_loss"] = dec["stop_loss"]
    spec["strategy"]["sector_cap"] = dec["sector_cap"]
    spec["ballast"]["mix"] = mix_label(dec["floor"])
    spec["name"] = f"r5 monthly system — {dec['floor']} trend-ballast"
    spec["procedure"] = proc
    return spec


def drift(spec: dict, proc: dict) -> dict:
    """Decision and model fields where the spec disagrees with a fresh run
    (the model fields are read from the spec itself, so a hand edit of
    horizon.label or training_window_years shows up as drift too)."""
    have = spec.get("procedure", {}).get("decision", {})
    out = {k: (have.get(k), v) for k, v in proc["decision"].items() if have.get(k) != v}
    model = {"label": spec.get("horizon", {}).get("label"),
             "train_years": spec.get("training_window_years")}
    out.update({k: (model[k], proc["model"][k]) for k in MODEL_KEYS if model[k] != proc["model"][k]})
    return out


def run(preds_path, store=STORE, k=None, lo=SELECT[0], hi=SELECT[1], check=False,
        spec_path: Path = SPEC_PATH, card_path: Path | None = None, log=print) -> dict:
    from stocks_ml.models.trials import record_trials
    from stocks_ml.procedure_card import CARD_PATH, write_card
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    proc = decide(preds_path, store, k, lo, hi, log=log)
    dec = proc["decision"]
    log(f"procedure: decision {dec}; evidence {json.dumps(proc['evidence'])}")
    spec = json.loads(Path(spec_path).read_text())
    changed = drift(spec, proc)
    if check:
        if changed:
            raise SystemExit(f"spec drifted from the procedure: {changed}")
        log("procedure: the spec matches the procedure")
        return proc
    spec = apply(spec, proc)
    Path(spec_path).write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    write_card(spec_path, card_path or CARD_PATH)
    walk = Path(preds_path).parent
    try:                                   # the walk's path under data/experiments names the row
        walk_name = "_".join(walk.relative_to("data/experiments").parts)
    except ValueError:
        walk_name = walk.name
    record_trials([{"kind": "procedure",
                    "name": f"procedure_{lo.date()}_{hi.date()}_{walk_name}",
                    "pre_holdout_sharpe": proc["selection_window_record"]["config"]["sharpe"],
                    "notes": json.dumps({"decision": dec, "evidence": proc["evidence"],
                                         "preds_sha256": proc["preds"]["sha256"][:12],
                                         "changed": changed})}])
    log(f"procedure: wrote {spec_path} and PROCEDURE.md"
        + (f"; changed {changed}" if changed else "; no change"))
    return proc
