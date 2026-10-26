#!/bin/bash
# Durable watcher for the lp6x6 job.  usage: lp6x6_watch.sh JOBID RUNDIR
# COMPLETED only if progress.txt has the 'VERDICT 6x6' line; FAILED/CANCELLED/OOM/NODE_FAIL -> FAILED; TIMEOUT; DISAPPEARED (sacct
# lists no such job 5 polls running); STALLED (RUNNING and progress.txt untouched for STALL s).  Last step: job_finish.sh --wake none.
# Test hooks: W_SACCT (command printing the state), W_FIN, W_DELTA, POLL, STALL.
JOB=$1; RUN=$2; POLL=${POLL:-120}; STALL=${STALL:-3600}
FIN=${W_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
DELTA=${W_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
st_of() { if [ -n "${W_SACCT:-}" ]; then $W_SACCT; else sacct -j $JOB -X -n -P -o State 2>/dev/null | head -1 | sed 's/ .*//'; fi; }
miss=0; t_start=$(date +%s)
while true; do
  st=$(st_of)
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { state=DISAPPEARED; break; }
  else miss=0
    case "$st" in
      COMPLETED) if grep -q "VERDICT 6x6" $RUN/progress.txt 2>/dev/null; then state=COMPLETED; else state=FAILED; fi; break;;
      FAILED|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) state=FAILED; break;;
      TIMEOUT) state=TIMEOUT; break;;
      RUNNING) ref=$(stat -c %Y $RUN/progress.txt 2>/dev/null || echo $t_start); [ $(( $(date +%s) - ref )) -gt $STALL ] && { state=STALLED; break; };;
    esac
  fi
  sleep $POLL
done
echo "$state $(date -Is)" > $RUN/watcher_verdict.txt
exec $FIN --run $RUN --job $JOB --state $state --delta $DELTA --wake none
