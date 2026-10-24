#!/bin/bash
# Durable watcher for a Paderborn rbmpp8 job (cluster-jobs skill). Poll interval POLL s. Run it as a setsid orphan:
#   setsid nohup bash watch_rbmpp8.sh JOBID RUNNAME LOCALDIR STALL_S > LOCALDIR/watcher.log 2>&1 < /dev/null &
# Branches: COMPLETED (job COMPLETED and the run wrote DONE_RBMPP8 with exit 0), FAILED / TIMEOUT / CANCELLED /
# OUT_OF_MEMORY etc (sacct state), STALLED (job RUNNING but the newest PROGRESS line / checkpoint is older than STALL_S;
# set STALL_S = 2 x checkpoint interval), DISAPPEARED (neither in squeue nor in sacct for 3 polls in a row).
# Test hook: WATCH_FAKE=DIR replaces ssh by files DIR/{state,age,done} (state word; age seconds; done: 0/1).
# Last step: tools/job_finish.sh --wake none (writes the VERDICT line into DELTA, runs the analysis from ANALYSIS_BRIEF.md).
JOB=$1; RUN=$2; LOCAL=$3; STALL=${4:-3600}
POLL=${POLL:-300}
REMOTE=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8/runs/$RUN
DELTA=${DELTA:-/project/theorie/a/A.Otaifi/Chatty-organize/projects/j1j2/delta_today.md}
JF=${JF:-/project/theorie/a/A.Otaifi/tools/job_finish.sh}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn"
SL='export PATH=/opt/software/slurm/25.11.8-1/bin:$PATH;'
mkdir -p "$LOCAL"
say() { echo "$(date -Is) $*" | tee -a "$LOCAL/watcher.log"; }
getstate() {   # prints: STATE AGE DONE   (STATE=UNKNOWN on ssh failure, GONE if absent from squeue and sacct)
  if [ -n "${WATCH_FAKE:-}" ]; then echo "$(cat $WATCH_FAKE/state) $(cat $WATCH_FAKE/age) $(cat $WATCH_FAKE/done)"; return; fi
  out=$($SSH "$SL st=\$(squeue -h -j $JOB -o %T | head -1); [ -z \"\$st\" ] && st=\$(sacct -n -X -j $JOB -o State%30 | head -1 | awk '{print \$1}'); [ -z \"\$st\" ] && st=GONE;
     now=\$(date +%s); m=\$(stat -c %Y $REMOTE/state.npz $REMOTE/train.json 2>/dev/null | sort -n | tail -1); [ -z \"\$m\" ] && m=\$now;
     l=\$(stat -c %Y /scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8/runs/*-$JOB.out 2>/dev/null | tail -1); [ -z \"\$l\" ] && l=\$m; [ \$l -gt \$m ] && m=\$l;
     d=0; grep -q '^DONE_RBMPP8 exit=0' $REMOTE/VERDICT_RUN 2>/dev/null && d=1; echo \$st \$((now-m)) \$d" 2>/dev/null) || { echo "UNKNOWN 0 0"; return; }
  echo "$out"
}
finish() {   # $1 = state
  say "FINAL $1"
  if [ -z "${WATCH_FAKE:-}" ]; then rsync -a -e "ssh -o BatchMode=yes -o HostName=login4" --exclude 'code/' --exclude 'state*.npz' paderborn:$REMOTE/ "$LOCAL/" >> "$LOCAL/watcher.log" 2>&1; fi
  echo "$1 $(date -Is) job $JOB" > "$LOCAL/WATCHER_VERDICT"
  bash "$JF" --run "$LOCAL" --job "$JOB" --state "$1" --delta "$DELTA" --wake none >> "$LOCAL/watcher.log" 2>&1
  exit 0
}
gone=0; unk=0
while true; do
  read st age done_ <<< "$(getstate)"
  case "$st" in
    UNKNOWN) unk=$((unk+1)); say "ssh failed ($unk)";;
    GONE) gone=$((gone+1)); unk=0; [ $gone -ge 3 ] && finish DISAPPEARED;;
    PENDING|CONFIGURING) gone=0; unk=0;;
    RUNNING|COMPLETING) gone=0; unk=0
        if [ "$st" = RUNNING ] && [ "$age" -gt "$STALL" ]; then finish STALLED; fi;;
    COMPLETED) if [ "$done_" = 1 ]; then finish COMPLETED; else finish FAILED; fi;;
    FAILED|TIMEOUT|CANCELLED*|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED) finish "${st%%+*}";;
    *) say "unhandled state $st";;
  esac
  sleep "$POLL"
done
