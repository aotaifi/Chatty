#!/bin/bash
# usage: watch2.sh JOBID DONE_MARKER OUTFILE PROGFILE STALL_SEC RUNDIR   (setsid orphan; last step tools/job_finish.sh --wake none)
# states: sacct final state + the run's own marker; STALLED if RUNNING and PROGFILE older than STALL_SEC (scancel own job); DEPFAIL if PENDING with reason DependencyNeverSatisfied (an upstream stage failed); DISAPPEARED after 5 empty polls.
J=$1; MK=$2; F=$3; PF=$4; ST=$5; D=$6; DELTA=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md
decide() { case "$1" in COMPLETED) [ "$3" = 1 ] && echo COMPLETED || echo FAILED;; FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL) echo ${1%%+*};;
  RUNNING) [ "$2" -gt "$ST" ] && echo STALLED;; PENDING) [ "$4" = DependencyNeverSatisfied ] && echo DEPFAIL;; "") echo DISAPPEARED;; esac; }
if [ "$J" = selftest ]; then ST=1200; for t in "COMPLETED 0 1 x" "COMPLETED 0 0 x" "FAILED 0 0 x" "OUT_OF_MEMORY 0 0 x" "RUNNING 99999 0 x" "RUNNING 5 0 x" "PENDING 0 0 Priority" "PENDING 0 0 DependencyNeverSatisfied" "TIMEOUT 0 0 x"; do set -- $t; echo "[$t] -> $(decide "$@")"; done; echo "[<empty>] -> $(decide "" 0 0 x)"; exit; fi
miss=0
while :; do sleep 60
  S=$(sacct -j $J -n -X -o State | head -1 | tr -d ' ')
  if [ -z "$S" ]; then miss=$((miss+1)); [ $miss -lt 5 ] && continue; else miss=0; fi
  age=0; [ -f "$PF" ] && age=$(( $(date +%s) - $(stat -c %Y "$PF") )); dn=0; grep -q "$MK" "$F" 2>/dev/null && dn=1
  RS=$(squeue -h -j $J -o %r 2>/dev/null | head -1)
  R=$(decide "$S" $age $dn "$RS"); [ -n "$R" ] && break
done
[ "$R" = STALLED ] && scancel $J
echo "$(date '+%F %T') job $J final $R (sacct $S, marker $dn, progress age $age s)" >> $D/watch_$J.log
/project/theorie/a/A.Otaifi/tools/job_finish.sh --run $D --job $J --state $R --delta $DELTA --wake none >> $D/watch_$J.log 2>&1
