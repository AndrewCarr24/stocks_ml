#!/bin/zsh
# The r5 champion's weekly cycle on the owner's Mac, by hand:
#   ops/r5_weekly.sh [--dry-run] [--as-of F] ...
# The scheduled run is GitHub Actions (.github/workflows/champion.yml); the
# launchd schedule for this script (com.stocks-ml.r5-weekly.plist) was
# retired 2026-09-02 so two champions never race. Needs data/.sharadar_key.
# Logs to logs/ (git-ignored); ~20 min (data refresh).
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
