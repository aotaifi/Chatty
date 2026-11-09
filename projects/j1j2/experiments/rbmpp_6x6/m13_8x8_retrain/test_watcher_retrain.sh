#!/bin/bash
# synthetic test of every watch_retrain.sh branch (no ssh), incl. the registered kill
H=$(cd "$(dirname "$0")" && pwd); T=$(mktemp -d); export POLL=1 DELTA=$T/delta.md CLAUDE_BIN=/bin/false
for c in "COMPLETED 10 1 WAIT COMPLETED" "COMPLETED 10 0 WAIT FAILED" "FAILED 10 0 WAIT FAILED" "TIMEOUT 10 0 WAIT TIMEOUT" "RUNNING 99999 0 WAIT STALLED" "GONE 0 0 WAIT DISAPPEARED" "CANCELLED+ 0 0 WAIT CANCELLED" "RUNNING 10 0 KILL KILLED"; do
  set -- $c; mkdir -p $T/f; echo $1 > $T/f/state; echo $2 > $T/f/age; echo $3 > $T/f/done; echo $4 > $T/f/kill; L=$T/loc_$1_$3_$4; mkdir -p $L; echo brief > $L/ANALYSIS_BRIEF.md
  WATCH_FAKE=$T/f timeout 60 bash $H/watch_retrain.sh 99$RANDOM RUN $L 3600 > /dev/null 2>&1
  v=$(awk '{print $1}' $L/WATCHER_VERDICT 2>/dev/null); [ "$v" = "$5" ] && echo "PASS $1/$4 -> $v" || echo "FAIL $1/$4 -> '$v' (want $5)"
done
for k in PASS WAIT; do echo RUNNING > $T/f/state; echo 10 > $T/f/age; echo 0 > $T/f/done; echo $k > $T/f/kill; L=$T/loc_run_$k; mkdir -p $L
  WATCH_FAKE=$T/f timeout 5 bash $H/watch_retrain.sh 777 RUN $L 3600 >/dev/null 2>&1; [ -f $L/WATCHER_VERDICT ] && echo "FAIL running/$k finished" || echo "PASS running/$k keeps waiting"; done
grep -c "kill check PASS" $T/loc_run_PASS/watcher.log | sed 's/^/kill-PASS logged once: /'
echo "delta lines:"; cat $T/delta.md
