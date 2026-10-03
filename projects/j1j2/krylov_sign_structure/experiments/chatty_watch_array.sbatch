#!/bin/bash
#SBATCH --job-name=chatty-mailwatch
#SBATCH --partition=small
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:05:00
#SBATCH --mail-type=NONE
set -u
JOB_TO_WATCH="$1"
BASE=/home/a/A.Otaifi/ChattyRun/j1j2/krylov_sign_structure
states=$(sacct -j "$JOB_TO_WATCH" -X -n -P -o State | sed '/^$/d')
if echo "$states" | grep -Eq 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED'; then
  bad=$(echo "$states" | grep -E 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED' | head -1)
  "$BASE/experiments/chatty_notify.sh" FAILED "$JOB_TO_WATCH" "$bad"
else
  n=$(echo "$states" | grep -c '^COMPLETED' || true)
  "$BASE/experiments/chatty_notify.sh" FINISHED "$JOB_TO_WATCH" "$n task(s) completed"
fi
