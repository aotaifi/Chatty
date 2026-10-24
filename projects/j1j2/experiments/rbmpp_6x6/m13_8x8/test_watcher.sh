#!/bin/bash
# synthetic test of every watcher branch (no ssh): each case must end with the right line in a scratch delta file
H=$(cd "$(dirname "$0")" && pwd); T=$(mktemp -d); export POLL=1 DELTA=$T/delta.md CLAUDE_BIN=/bin/false
for c in "COMPLETED 10 1 COMPLETED" "COMPLETED 10 0 FAILED" "FAILED 10 0 FAILED" "TIMEOUT 10 0 TIMEOUT" "RUNNING 99999 0 STALLED" "GONE 0 0 DISAPPEARED" "CANCELLED+ 0 0 CANCELLED"; do
  set -- $c; mkdir -p $T/f; echo $1 > $T/f/state; echo $2 > $T/f/age; echo $3 > $T/f/done; L=$T/loc_$1_$3; mkdir -p $L; echo "brief" > $L/ANALYSIS_BRIEF.md
  WATCH_FAKE=$T/f timeout 60 bash $H/watch_rbmpp8.sh 99$RANDOM RUN $L 3600 > /dev/null 2>&1
  v=$(cat $L/WATCHER_VERDICT 2>/dev/null | awk '{print $1}')
  [ "$v" = "$4" ] && echo "PASS $1 -> $v" || echo "FAIL $1 -> '$v' (want $4)"
done
# a RUNNING job with a fresh age must NOT finish within 5 s
echo RUNNING > $T/f/state; echo 10 > $T/f/age; echo 0 > $T/f/done; L=$T/loc_run; mkdir -p $L
WATCH_FAKE=$T/f timeout 5 bash $H/watch_rbmpp8.sh 777 RUN $L 3600 >/dev/null 2>&1; [ -f $L/WATCHER_VERDICT ] && echo "FAIL running finished" || echo "PASS running keeps waiting"
echo "delta lines:"; cat $T/delta.md
