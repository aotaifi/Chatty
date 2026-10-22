#!/bin/bash
# Durable watcher for one O1 job: usage cx_o1_watch.sh JOBID RUNDIR   (started by launch_g0.sh under setsid)
# Fires on completion, non-zero exit, disappearance, stall (cx_o1.json not touched for STALL s). Last step: job_finish.sh --wake none.
JOB=$1; RUN=$2; STALL=${STALL:-1800}; POLL=${POLL:-60}
SACCT=${O1_SACCT:-sacct}; FIN=${O1_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
DELTA=${O1_DELTA:-/project/theorie/a/A.Otaifi/Chatty-organize/projects/j1j2/delta_today.md}
t_sub=$(date +%s); miss=0
while true; do
  st=$($SACCT -j "$JOB" -X -n -P -o State 2>/dev/null | head -1 | sed 's/ .*//')
  now=$(date +%s)
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { state=DISAPPEARED; break; }
  else miss=0
    case "$st" in
      COMPLETED) if grep -q '"done": true' "$RUN/cx_o1.json" 2>/dev/null; then state=COMPLETED; else state=FAILED; fi; break;;
      FAILED|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) state=FAILED; break;;
      TIMEOUT) state=TIMEOUT; break;;
      RUNNING) ref=$(stat -c %Y "$RUN/progress.txt" 2>/dev/null || stat -c %Y "$RUN/cx_o1.json" 2>/dev/null || stat -c %Y "$RUN/run_metadata.txt" 2>/dev/null || echo $t_sub)
               [ $((now-ref)) -gt $STALL ] && { state=STALLED; break; };;
    esac
  fi
  sleep "$POLL"
done
echo "$state $(date -Is)" > "$RUN/watcher_verdict.txt"
exec "$FIN" --run "$RUN" --job "$JOB" --state "$state" --delta "$DELTA" --wake none
