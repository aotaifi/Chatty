#!/bin/zsh
set -euo pipefail
ROOT="/Users/aliotaifi/Chatty/.chatty/slurm_wake_release_20261003_1137_safeguard"
mkdir -p "$ROOT"
nohup /opt/homebrew/bin/python3 "/Users/aliotaifi/Chatty/tools/local_loop/slurm_wake_release_20261003_1137_safeguard/ui_broker.py" >> "$ROOT/ui-broker.stdout.log" 2>&1 < /dev/null &
exit 0
