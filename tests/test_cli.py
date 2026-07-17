import json
import subprocess
import sys
from argparse import Namespace

from stocks_ml.cli import cmd_ledger


def test_cli_help_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "stocks_ml.cli", "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    for cmd in ("ingest", "train", "backtest", "signals", "ledger", "torture"):
        assert cmd in proc.stdout


def test_ledger_apply_second_call_is_idempotent_noop(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "signals").mkdir()
    (tmp_path / "signals" / "2026-07-11-trades.json").write_text(
        '[["AAA", 1.0, 10.0]]')

    cmd_ledger(Namespace(action="init", file=None, force=False, cash=100.0), cfg=None)
    capsys.readouterr()

    cmd_ledger(Namespace(action="apply", file=None, force=False, cash=100.0), cfg=None)
    first_out = capsys.readouterr().out
    assert "applied 1 trades" in first_out

    # second apply of the same file is a benign no-op: exits normally (no
    # SystemExit/raise) and prints a skip message rather than re-applying.
    cmd_ledger(Namespace(action="apply", file=None, force=False, cash=100.0), cfg=None)
    second_out = capsys.readouterr().out
    assert "already applied 2026-07-11-trades.json; skipping" in second_out

    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["positions"] == {"AAA": 1.0}  # not double-applied
    assert ledger["cash"] == 90.0
