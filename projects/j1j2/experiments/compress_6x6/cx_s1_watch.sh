#!/bin/bash
# Durable watcher for one S1 job: usage cx_s1_watch.sh JOBID RUNDIR   (started by launch_s1.sh under setsid)
# Fires on completion, non-zero exit, disappearance, stall (no new PROGRESS line in progress.txt for STALL s; a round takes ~2 min). Last step: job_finish.sh --wake none.
JOB=$1; RUN=$2; STALL=${STALL:-1200}; POLL=${POLL:-60}
SACCT=${S1_SACCT:-sacct}; FIN=${S1_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
DELTA=${S1_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
t_sub=$(date +%s); miss=0
while true; do
  st=$($SACCT -j "$JOB" -X -n -P -o State 2>/dev/null | head -1 | sed 's/ .*//')
  now=$(date +%s)
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { state=DISAPPEARED; break; }
  else miss=0
    case "$st" in
      COMPLETED) if grep -q '"done": true' "$RUN/cx_s1.json" 2>/dev/null; then state=COMPLETED; else state=FAILED; fi; break;;
      FAILED|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) state=FAILED; break;;
      TIMEOUT) state=TIMEOUT; break;;
      RUNNING) ref=$(stat -c %Y "$RUN/progress.txt" 2>/dev/null || stat -c %Y "$RUN/cx_s1.json" 2>/dev/null || stat -c %Y "$RUN/run_metadata.txt" 2>/dev/null || echo $t_sub)
               [ $((now-ref)) -gt $STALL ] && { state=STALLED; break; };;
    esac
  fi
  sleep "$POLL"
done
echo "$state $(date -Is)" > "$RUN/watcher_verdict.txt"
exec "$FIN" --run "$RUN" --job "$JOB" --state "$state" --delta "$DELTA" --wake none
