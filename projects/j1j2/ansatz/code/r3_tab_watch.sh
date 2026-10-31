#!/bin/bash
# Durable watcher for the table job.  usage: r3_tab_watch.sh JOBID ROOT/tabs   (env STALL s, POLL s; test hooks R1_SACCT, R1_FIN, R1_DELTA)
# COMPLETED needs both tab_900/cx_r1.json and tab_1200/cx_r1.json "done": true without "stop".  Stall reference: newest mtime of run_metadata.txt and tab_*/round_1/progress.txt (a table pass has no lines for ~35 min: STALL 5400).
JOB=$1; RUN=$2; STALL=${STALL:-5400}; POLL=${POLL:-120}
FIN=${R1_FIN:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
DELTA=${R1_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
SACCT=${R1_SACCT:-sacct}
miss=0
while true; do
  st=$($SACCT -j "$JOB" -X -n -P -o State 2>/dev/null | head -1 | sed 's/ .*//')
  now=$(date +%s)
  if [ -z "$st" ]; then miss=$((miss+1)); [ $miss -ge 5 ] && { state=DISAPPEARED; break; }
  else miss=0
    case "$st" in
      COMPLETED) ok=1; for N in 900 1200; do f="$RUN/tab_$N/cx_r1.json"; grep -q '"done": true' "$f" 2>/dev/null && ! grep -q '"stop"' "$f" || ok=0; done
                 if [ $ok = 1 ]; then state=COMPLETED; else state=FAILED; fi; break;;
      FAILED|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL) state=FAILED; break;;
      TIMEOUT) state=TIMEOUT; break;;
      RUNNING) ref=$(stat -c %Y "$RUN"/tab_*/round_1/progress.txt "$RUN/run_metadata.txt" 2>/dev/null | sort -n | tail -1)
               [ -n "$ref" ] && [ $((now-ref)) -gt "$STALL" ] && { state=STALLED; break; };;
    esac
  fi
  sleep "$POLL"
done
echo "$state $(date -Is)" > "$RUN/watcher_verdict.txt"
exec "$FIN" --run "$RUN" --job "$JOB" --state "$state" --delta "$DELTA" --wake none
