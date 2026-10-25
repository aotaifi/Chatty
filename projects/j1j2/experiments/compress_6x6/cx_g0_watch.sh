#!/bin/bash
# Durable watcher for one G0 job: usage cx_g0_watch.sh JOBID RUNDIR   (started by launch_g0.sh under setsid)
# Fires on completion, non-zero exit, disappearance, stall (cx_g0.json not touched for STALL s). Last step: job_finish.sh --wake none.
JOB=$1; RUN=$2; STALL=${STALL:-1500}; POLL=${POLL:-60}
SACCT=${G0_SACCT:-sacct}; FIN=${G0_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
DELTA=${G0_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
t_sub=$(date +%s); miss=0
while true; do
  st=$($SACCT -j "$JOB" -X -n -P -o State 2>/dev/null | head -1 | sed 's/ .*//')
  now=$(date +%s)
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { state=DISAPPEARED; break; }
  else miss=0
    case "$st" in
      COMPLETED) if grep -q '"sec"' "$RUN/cx_g0.json" 2>/dev/null && ! grep -q '"all_ok": false' "$RUN/cx_g0.json"; then state=COMPLETED; else state=FAILED; fi; break;;
      FAILED|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) state=FAILED; break;;
      TIMEOUT) state=TIMEOUT; break;;
      RUNNING) ref=$(stat -c %Y "$RUN/cx_g0.json" 2>/dev/null || stat -c %Y "$RUN/run_metadata.txt" 2>/dev/null || echo $t_sub)
               [ $((now-ref)) -gt $STALL ] && { state=STALLED; break; };;
    esac
  fi
  sleep "$POLL"
done
# a controls-failed exit (exit 2) is a verdict of the run itself: label it
grep -q '"all_ok": false' "$RUN/cx_g0.json" 2>/dev/null && state=FAILED
echo "$state $(date -Is)" > "$RUN/watcher_verdict.txt"
exec "$FIN" --run "$RUN" --job "$JOB" --state "$state" --delta "$DELTA" --wake none
