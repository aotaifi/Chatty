#!/bin/bash
# durable watcher (setsid orphan) for one loop4x4 driver on this host.  usage: loop4x4_watch.sh RUNDIR PID JOBID DELTA [STALL_SEC] [POLL_SEC]
D=$1; PID=$2; JOB=$3; DELTA=$4; STALL=${5:-1800}; POLL=${6:-60}; JF=${JOB_FINISH:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
echo "$(date -Is) watcher pid $$ for $PID ($JOB) stall ${STALL}s" >> $D/watcher.log
while true; do
  if [ -f $D/DRIVER_DONE ]; then st=$(cat $D/DRIVER_DONE); break; fi
  if ! kill -0 $PID 2>/dev/null; then sleep 5; [ -f $D/DRIVER_DONE ] && { st=$(cat $D/DRIVER_DONE); break; }; st=DISAPPEARED; break; fi
  age=$(( $(date +%s) - $(stat -c %Y $D/progress.txt 2>/dev/null || date +%s) ))
  if [ $age -gt $STALL ]; then st=STALLED; break; fi
  sleep $POLL
done
echo "$(date -Is) final state $st" >> $D/watcher.log
$JF --run $D --job $JOB --state $st --delta $DELTA --wake none >> $D/watcher.log 2>&1
