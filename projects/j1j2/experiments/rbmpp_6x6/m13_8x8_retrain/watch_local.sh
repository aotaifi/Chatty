#!/bin/bash
# Durable watcher for a theorie-cluster rbmpp8 evaluation job. Usage (setsid orphan):
#   setsid nohup bash watch_local.sh JOBID RUNDIR STALL_S > RUNDIR/watcher.out 2>&1 < /dev/null &
# Branches: COMPLETED (sacct COMPLETED and VERDICT_RUN has eval_rc=0), FAILED (COMPLETED without it, or sacct FAILED),
# TIMEOUT/CANCELLED/OUT_OF_MEMORY/NODE_FAIL, STALLED (RUNNING but eval.log has no new line for STALL_S; PROGRESS is every
# 20 sweeps = ~5 min on an A40), DISAPPEARED (absent from squeue and sacct 3 polls). Test hook: WATCH_FAKE=DIR with files
# state, age, done (replaces squeue/sacct/stat). Last step: job_finish.sh --wake none.
JOB=$1; RUN=$2; STALL=${3:-900}; POLL=${POLL:-120}
DELTA=${DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
JF=${JF:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
say() { echo "$(date -Is) $*" | tee -a "$RUN/watcher.log"; }
getstate() {
  if [ -n "${WATCH_FAKE:-}" ]; then echo "$(cat $WATCH_FAKE/state) $(cat $WATCH_FAKE/age) $(cat $WATCH_FAKE/done)"; return; fi
  st=$(squeue -h -j $JOB -o %T 2>/dev/null | head -1)
  [ -z "$st" ] && st=$(sacct -n -X -j $JOB -o State%30 2>/dev/null | head -1 | awk '{print $1}')
  [ -z "$st" ] && st=GONE
  now=$(date +%s); m=$(stat -c %Y $RUN/eval.log 2>/dev/null || stat -c %Y $RUN/slurm-$JOB.out 2>/dev/null || echo $now)
  d=0; grep -q "eval_rc=0" $RUN/VERDICT_RUN 2>/dev/null && d=1
  echo "$st $((now-m)) $d"
}
finish() {
  say "FINAL $1"; echo "$1 $(date -Is) job $JOB" > "$RUN/WATCHER_VERDICT"
  bash "$JF" --run "$RUN" --job "$JOB" --state "$1" --delta "$DELTA" --wake none >> "$RUN/watcher.log" 2>&1
  exit 0
}
gone=0
while true; do
  read st age done_ <<< "$(getstate)"
  case "$st" in
    GONE) gone=$((gone+1)); [ $gone -ge 3 ] && finish DISAPPEARED;;
    PENDING|CONFIGURING) gone=0;;
    RUNNING|COMPLETING) gone=0; if [ "$st" = RUNNING ] && [ "$age" -gt "$STALL" ]; then finish STALLED; fi;;
    COMPLETED) if [ "$done_" = 1 ]; then finish COMPLETED; else finish FAILED; fi;;
    FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED) finish "${st%%+*}";;
    *) say "unhandled state $st";;
  esac
  sleep "$POLL"
done
