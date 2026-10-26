#!/bin/bash
# Durable watcher (cluster-jobs skill): usage watch_job.sh JOBID RUNDIR STALL_SECONDS DELTA
# Fires on: completion (RUN_VERDICT line), non-zero exit / failed states, disappearance from sacct, stall (progress.txt
# not modified for STALL_SECONDS while RUNNING).  Last step: tools/job_finish.sh --wake none.  SACCT_CMD overridable for tests.
JOB=$1; RUN=$2; STALL=$3; DELTA=$4
SACCT=${SACCT_CMD:-"sacct -n -X -P -o State -j"}
FIN=${FINISH_CMD:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
POLL=${POLL:-60}
W="$RUN/watcher_$JOB.log"; miss=0
while true; do
  st=$($SACCT "$JOB" 2>/dev/null | head -1 | awk '{print $1}')
  now=$(date +%s)
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { state=DISAPPEARED; break; }; else miss=0; fi
  case "$st" in
    COMPLETED) if grep -q "RUN_VERDICT OK" "$RUN/progress.txt" 2>/dev/null; then state=COMPLETED; else state=FAILED; fi; break;;
    FAILED|CANCELLED*|TIMEOUT|OUT_OF_ME*|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE) state=$st; [ "$st" = TIMEOUT ] || state=FAILED; break;;
    RUNNING)
      m=$(stat -c %Y "$RUN/progress.txt" 2>/dev/null || echo 0); [ "$m" = 0 ] && m=${START:-$now}; START=${START:-$now}
      [ $((now-m)) -gt "$STALL" ] && { state=STALLED; break; };;
  esac
  sleep "$POLL"
done
echo "$(date -Is) job $JOB final $state (sacct '$st')" >> "$W"
"$FIN" --run "$RUN" --job "$JOB" --state "$state" --delta "$DELTA" --wake none >> "$W" 2>&1
