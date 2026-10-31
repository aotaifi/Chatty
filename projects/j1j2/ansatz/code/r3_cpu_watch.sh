#!/bin/bash
# Durable watcher for the round-3 CPU readout job.  usage: r3_cpu_watch.sh JOBID RUNDIR(readout/r3_1500)   env STALL (3600), POLL (120); test hooks R1_SACCT, R1_FIN, R1_DELTA
# COMPLETED needs readout.json "done": true.  Stall reference: mtime of RUNDIR/readout.log (build/solve lines every <= ~10 min).
JOB=$1; RUN=$2; STALL=${STALL:-3600}; POLL=${POLL:-120}
FIN=${R1_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}; DELTA=${R1_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}; SACCT=${R1_SACCT:-sacct}
miss=0
while true; do
  st=$($SACCT -j "$JOB" -X -n -P -o State 2>/dev/null | head -n 1 | sed 's/ .*//'); now=$(date +%s)
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { state=DISAPPEARED; break; }
  else miss=0
    case "$st" in
      COMPLETED) if grep -q '"done": true' "$RUN/readout.json" 2>/dev/null; then state=COMPLETED; else state=FAILED; fi; break;;
      FAILED|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) state=FAILED; break;;
      TIMEOUT) state=TIMEOUT; break;;
      RUNNING) ref=$(stat -c %Y "$RUN/readout.log" 2>/dev/null); [ -n "$ref" ] && [ $((now-ref)) -gt "$STALL" ] && { state=STALLED; break; };;
    esac
  fi
  sleep "$POLL"
done
echo "$state $(date -Is)" > "$RUN/watcher_verdict.txt"
exec "$FIN" --run "$RUN" --job "$JOB" --state "$state" --delta "$DELTA" --wake none
