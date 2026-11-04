#!/bin/bash
# Durable watcher for pool_fit8_job.sbatch jobs.  usage: pf8_watch.sh JOBID RUNDIR   env STALL (3600 s), POLL (120 s); test hooks R1_SACCT, R1_FIN, R1_DELTA
# COMPLETED needs RUNDIR/VERDICT_LINE to hold a 'VERDICT pf8' line (the run's own verdict, written last by the job).  Stall reference: newest mtime of RUNDIR/run.log, RUNDIR/test.log, RUNDIR/verdict.txt.
JOB=$1; RUN=$2; STALL=${STALL:-3600}; POLL=${POLL:-120}
FIN=${R1_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}; DELTA=${R1_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}; SACCT=${R1_SACCT:-sacct}
miss=0
while true; do
  st=$($SACCT -j "$JOB" -X -n -P -o State 2>/dev/null | head -n 1 | sed 's/ .*//'); now=$(date +%s)
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { state=DISAPPEARED; break; }
  else miss=0
    case "$st" in
      COMPLETED) if grep -q 'VERDICT pf8' "$RUN/VERDICT_LINE" 2>/dev/null; then state=COMPLETED; else state=FAILED; fi; break;;
      FAILED|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) state=FAILED; break;;
      TIMEOUT) state=TIMEOUT; break;;
      RUNNING) ref=$(stat -c %Y "$RUN/run.log" "$RUN/test.log" "$RUN/verdict.txt" 2>/dev/null | sort -n | tail -1); [ -n "$ref" ] && [ $((now-ref)) -gt "$STALL" ] && { state=STALLED; break; };;
    esac
  fi
  sleep "$POLL"
done
echo "$state $(date -Is)" > "$RUN/watcher_verdict.txt"
exec "$FIN" --run "$RUN" --job "$JOB" --state "$state" --delta "$DELTA" --wake none
