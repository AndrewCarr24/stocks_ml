import subprocess
import sys


def test_cli_help_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "stocks_ml.cli", "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    for cmd in ("ingest", "train", "backtest", "signals", "ledger"):
        assert cmd in proc.stdout
