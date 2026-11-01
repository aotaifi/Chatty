#!/bin/bash
# Round-4 chain gate (durable, setsid).  Slurm dependencies do not cross clusters: this waits until ONE cluster (Paderborn: pdb_fit + pdb_tabs; theorie A40: fit + tabs) has fit AND table verdicts COMPLETED with
# the la_tables pulled back, then writes readout/winner.txt, submits r4_readout.sbatch (theorie cluster) and arms r3_cpu_watch.sh on it.  If a cluster's fit verdict is not COMPLETED, its dependent table job is cancelled (a failed fit
# never satisfies afterok).  If neither cluster can complete (every cluster dead: verdict != COMPLETED) nothing is submitted.  Gives up after GIVEUP s (default 36 h: A40 worst case 13 h fit + 10 h tabs + queue).
# Ssh uses -o HostName=login4.  Test hooks: R1_SBATCH, R1_CHAIN_Q, R1_NOCANCEL, POLL, GIVEUP.
Q=${R1_CHAIN_Q:-/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r4}; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code
POLL=${POLL:-120}; GIVEUP=${GIVEUP:-129600}; SB=${R1_SBATCH:-sbatch}; t0=$(date +%s)
SUBS="t4 600 750 900 1050 1200 1350"
v() { head -c 9 "$1" 2>/dev/null | awk '{print $1}'; }
say() { echo "$(date -Is) $*" | tee -a $Q/readout/chain_verdict.txt; }
cancelled=""
cstate() { # FITDIR TABDIR -> OK | DEAD | PENDING
  f=$(v $1/watcher_verdict.txt); t=$(v $2/watcher_verdict.txt)
  if [ "$f" = COMPLETED ] && [ "$t" = COMPLETED ]; then echo OK
  elif { [ -n "$f" ] && [ "$f" != COMPLETED ]; } || { [ -n "$t" ] && [ "$t" != COMPLETED ]; }; then echo DEAD
  else echo PENDING; fi; }
while true; do
  now=$(date +%s)
  [ $((now-t0)) -gt $GIVEUP ] && { say "GIVEUP: no readout submitted (verdicts: PF=$(v $Q/pdb_fit/watcher_verdict.txt) PT=$(v $Q/pdb_tabs/watcher_verdict.txt) AF=$(v $Q/fit/watcher_verdict.txt) AT=$(v $Q/tabs/watcher_verdict.txt))"; exit 1; }
  for A in pdb a40; do
    [ $A = pdb ] && { FD=$Q/pdb_fit; TD=$Q/pdb_tabs; } || { FD=$Q/fit; TD=$Q/tabs; }
    f=$(v $FD/watcher_verdict.txt)
    if [ -n "$f" ] && [ "$f" != COMPLETED ] && [[ "$cancelled" != *$A* ]]; then
      cancelled="$cancelled$A"; tj=$(cat $TD/job_id.txt 2>/dev/null)
      if [ -z "${R1_NOCANCEL:-}" ] && [ -n "$tj" ]; then
        if [ $A = pdb ]; then ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; scancel $tj" >/dev/null 2>&1
        else scancel $tj >/dev/null 2>&1; fi
      fi
      say "$A fit verdict $f: dependent table job $tj cancelled (a failed fit never satisfies afterok)"
    fi
  done
  SP=$(cstate $Q/pdb_fit $Q/pdb_tabs); SA=$(cstate $Q/fit $Q/tabs)
  W=""; [ "$SP" = OK ] && W=pdb; [ -z "$W" ] && [ "$SA" = OK ] && W=a40
  if [ -n "$W" ]; then
    [ $W = pdb ] && { FD=$Q/pdb_fit; TD=$Q/pdb_tabs; } || { FD=$Q/fit; TD=$Q/tabs; }
    miss=""; for N in $SUBS; do [ -s $TD/tab_$N/round_1/la_table.npy ] || miss="$miss tab_$N"; done
    [ -s $FD/round_1/la_table.npy ] || miss="$miss final"
    [ -s $TD/theta_4.npy ] || miss="$miss theta_4"; [ -s $TD/theta_shas.json ] || miss="$miss theta_shas"
    for N in 600 750 900 1050 1200 1350; do [ -s $FD/round_1/theta_step$N.npy ] || miss="$miss theta_step$N"; done; [ -s $FD/round_1/theta.npy ] || miss="$miss theta_final"
    [ -n "$miss" ] && { say "NOT SUBMITTED (winner $W): missing locally:$miss"; exit 1; }
    echo $W > $Q/readout/winner.txt
    break
  fi
  if [ "$SP" = DEAD ] && [ "$SA" = DEAD ]; then say "NOT SUBMITTED: both clusters dead (PF=$(v $Q/pdb_fit/watcher_verdict.txt) PT=$(v $Q/pdb_tabs/watcher_verdict.txt) AF=$(v $Q/fit/watcher_verdict.txt) AT=$(v $Q/tabs/watcher_verdict.txt))"; exit 1; fi
  sleep $POLL
done
OUTL=$($SB --parsable --no-requeue --output=$Q/readout/slurm-%j.out $C/r4_readout.sbatch); JOB=${OUTL%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { say "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > $Q/readout/job_id.txt; echo "r4_readout $JOB (winner $W) $(date -Is)" >> $Q/job_ids.txt
setsid nohup $C/r3_cpu_watch.sh "$JOB" $Q/readout > $Q/readout/watcher.log 2>&1 < /dev/null &
echo $! > $Q/readout/watcher_pid.txt
say "SUBMITTED readout job $JOB (winner $W); r3_cpu_watch pid $(cat $Q/readout/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/readout/watcher_pid.txt))"
