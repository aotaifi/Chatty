#!/usr/bin/env bash
# durable watcher for one B1 job.  usage: b1_watch.sh JOB RUN_DIR STAGE STALL_MIN
# fires on: COMPLETED with the run's own "B1_VERDICT <STAGE> COMPLETE" line, any non-COMPLETED end state, disappearance, stall (newest slurm-*.out mtime older than STALL_MIN).
# test hooks (env): STATE_FILE (fake sacct state), JOB_FINISH (stub), DELTA, POLL (seconds)
JOB=$1; RUN=$2; STAGE=$3; STALL=${4:-30}
DELTA=${DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
JF=${JOB_FINISH:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
POLL=${POLL:-60}
state() { if [ -n "${STATE_FILE:-}" ]; then cat "$STATE_FILE"; else sacct -j "$JOB" -X -n -P -o State 2>/dev/null | head -1 | awk '{print $1}'; fi; }
fire() { echo "$(date '+%F %T') watcher job $JOB stage $STAGE final state $1" >> "$RUN/WATCHER_VERDICT.txt"
         "$JF" --run "$RUN" --job "$JOB" --state "$1" --delta "$DELTA" --wake none >> "$RUN/watcher.log" 2>&1; exit 0; }
empty=0
while true; do
  s=$(state)
  case "$s" in
    COMPLETED) if grep -q "B1_VERDICT $STAGE COMPLETE" "$RUN"/slurm-"$JOB".out 2>/dev/null; then fire COMPLETED; else fire FAILED; fi;;
    FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED) fire "${s%% *}";;
    "") empty=$((empty+1)); [ $empty -ge 5 ] && fire DISAPPEARED;;
    RUNNING)
      empty=0
      f=$(ls -t "$RUN"/slurm-"$JOB".out 2>/dev/null | head -1)
      if [ -n "$f" ]; then age=$(( ( $(date +%s) - $(stat -c %Y "$f") ) / 60 )); [ $age -ge "$STALL" ] && fire STALLED; fi;;
    *) empty=0;;
  esac
  sleep "$POLL"
done
