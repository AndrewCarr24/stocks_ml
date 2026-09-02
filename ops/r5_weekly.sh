#!/bin/zsh
# The r5 champion's weekly signal, run by launchd (com.stocks-ml.r5-weekly)
# every Saturday 09:00 from the owner's Mac. Needs data/.sharadar_key.
# Logs to logs/ (git-ignored). Not for Actions: needs the key; ~20 min (data refresh).
set -euo pipefail
REPO="/Users/andrewcarr/Documents/projects/stocks_ml.nosync"
UV="/opt/homebrew/Caskroom/miniconda/base/bin/uv"
cd "$REPO"
mkdir -p logs
LOG="logs/r5_weekly_$(date +%Y-%m-%d).log"
{
  echo "=== r5-weekly start $(date) ==="
  # AGENTS.md: synchronize before local work
  git pull -q --ff-only || echo "git pull failed; continuing on the local tree"
  "$UV" run stocks-ml r5-weekly "$@"
  echo "=== r5-weekly done $(date) ==="
} >> "$LOG" 2>&1
