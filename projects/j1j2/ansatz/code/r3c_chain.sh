#!/bin/bash
# R3-cont chain gate (durable, setsid).  Slurm dependencies do not cross clusters: this waits until the four Paderborn watchers have written their verdicts and
# the ten la_tables are pulled back, then submits r3c_readout.sbatch (theorie cluster) and arms r3_cpu_watch.sh on it.  If a fit watcher's verdict is not COMPLETED, the
# arm's dependent table job (never runs: afterok) is cancelled on Paderborn and nothing is submitted.  Gives up after GIVEUP s (default 8 h).  Test hooks: R1_SBATCH, R1_CHAIN_Q, R1_NOCANCEL.
Q=${R1_CHAIN_Q:-/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r3c}; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code
POLL=${POLL:-120}; GIVEUP=${GIVEUP:-28800}; SB=${R1_SBATCH:-sbatch}; t0=$(date +%s)
v() { head -c 9 "$1" 2>/dev/null | awk '{print $1}'; }
say() { echo "$(date -Is) $*" | tee -a $Q/readout/chain_verdict.txt; }
cancelled=""
while true; do
  now=$(date +%s); [ $((now-t0)) -gt $GIVEUP ] && { say "GIVEUP: no readout submitted (verdicts: AF=$(v $Q/A_fit/watcher_verdict.txt) AT=$(v $Q/A_tabs/watcher_verdict.txt) BF=$(v $Q/B_fit/watcher_verdict.txt) BT=$(v $Q/B_tabs/watcher_verdict.txt))"; exit 1; }
  for A in A B; do
    f=$(v $Q/${A}_fit/watcher_verdict.txt)
    if [ -n "$f" ] && [ "$f" != COMPLETED ] && [[ "$cancelled" != *$A* ]]; then
      cancelled="$cancelled$A"
      [ -z "${R1_NOCANCEL:-}" ] && ssh -o BatchMode=yes -o HostName=login4 paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; scancel $(cat $Q/${A}_tabs/job_id.txt)" >/dev/null 2>&1
      say "arm $A fit verdict $f: dependent table job $(cat $Q/${A}_tabs/job_id.txt) cancelled (a failed fit never satisfies afterok)"
    fi
  done
  AT=$(v $Q/A_tabs/watcher_verdict.txt); BT=$(v $Q/B_tabs/watcher_verdict.txt); AF=$(v $Q/A_fit/watcher_verdict.txt); BF=$(v $Q/B_fit/watcher_verdict.txt)
  if [ -n "$AT" ] && [ -n "$BT" ] && [ -n "$AF" ] && [ -n "$BF" ]; then
    if [ "$AT" = COMPLETED ] && [ "$BT" = COMPLETED ] && [ "$AF" = COMPLETED ] && [ "$BF" = COMPLETED ]; then break; fi
    say "NOT SUBMITTED: verdicts AF=$AF AT=$AT BF=$BF BT=$BT"; exit 1
  fi
  sleep $POLL
done
miss=""; for A in A B; do for N in $([ $A = A ] && echo "150 300 450" || echo "150 300 450 600 750"); do [ -s $Q/${A}_tabs/tab_$N/round_1/la_table.npy ] || miss="$miss ${A}_$N"; done; [ -s $Q/${A}_fit/round_1/la_table.npy ] || miss="$miss ${A}_final"; done
[ -n "$miss" ] && { say "NOT SUBMITTED: la_table missing locally:$miss"; exit 1; }
OUTL=$($SB --parsable --no-requeue --output=$Q/readout/slurm-%j.out $C/r3c_readout.sbatch); JOB=${OUTL%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { say "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > $Q/readout/job_id.txt; echo "r3c_readout $JOB $(date -Is)" >> $Q/job_ids.txt
setsid nohup $C/r3_cpu_watch.sh "$JOB" $Q/readout > $Q/readout/watcher.log 2>&1 < /dev/null &
echo $! > $Q/readout/watcher_pid.txt
say "SUBMITTED readout job $JOB; r3_cpu_watch pid $(cat $Q/readout/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/readout/watcher_pid.txt))"
