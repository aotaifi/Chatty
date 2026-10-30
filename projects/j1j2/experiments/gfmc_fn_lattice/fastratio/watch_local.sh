#!/bin/bash
# Durable watcher for a LOCAL (theorie Slurm) job.  usage: watch_local.sh JOBID RUNDIR   (start with setsid nohup ... &)
# Polls sacct.  COMPLETED needs the run's own verdict line (FASTRATIO_DONE) in RUNDIR/progress.txt, else FAILED.  STALLED = RUNNING while progress.txt
# / run_metadata.txt untouched for STALL s.  PENDING is waited for (no stall check).  No such job NMISS polls in a row = DISAPPEARED.
# Last step: job_finish.sh --wake none (VERDICT line in delta_today.md).  Test hooks: GF_SACCT (command printing the state), GF_FIN, GF_DELTA, POLL.
JOB=$1; RUN=$2; STALL=${STALL:-600}; POLL=${POLL:-60}; NMISS=${NMISS:-5}; DONEPAT=${GF_DONE_PAT:-FASTRATIO_DONE}
FIN=${GF_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
DELTA=${GF_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
SACCT=${GF_SACCT:-"sacct -j $JOB -X -n -P -o State"}
miss=0
while true; do
  st=$($SACCT 2>/dev/null | head -1 | sed 's/ .*//'); now=$(date +%s)
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge $NMISS ] && { state=DISAPPEARED; break; }
  else miss=0
    case "$st" in
      COMPLETED) if [ "$(grep -c "$DONEPAT" $RUN/progress.txt 2>/dev/null)" -ge 1 ]; then state=COMPLETED; else state=FAILED; fi; break;;
      FAILED|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) state=FAILED; break;;
      TIMEOUT) state=TIMEOUT; break;;
      RUNNING) ref=$(stat -c %Y $RUN/progress.txt $RUN/run_metadata.txt 2>/dev/null | sort -n | tail -1)
               [ -n "$ref" ] && [ $((now-ref)) -gt $STALL ] && { state=STALLED; break; };;
    esac
  fi
  sleep "$POLL"
done
echo "$state $(date -Is)" > "$RUN/watcher_verdict.txt"
exec "$FIN" --run "$RUN" --job "$JOB" --state "$state" --delta "$DELTA" --wake none
