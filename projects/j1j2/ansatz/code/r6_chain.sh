#!/bin/bash
# Round-6 chain gate (durable, setsid; = r5_chain.sh with pfaffian_r6, arms r6 m5, per-arm table tag t6/t5, r6_readout.sbatch).  Slurm dependencies do not cross clusters.  Per arm (r6 m5) two copies race:
# Paderborn (ARM/fit + ARM/tab) and theorie (ARM/fit_th + ARM/tab_th).  A copy is OK when its fit and table watcher verdicts are COMPLETED, DEAD when either verdict is
# not COMPLETED (or, for Paderborn, ARM/pdb_skip exists or no Paderborn job was ever submitted while the theorie copy is OK).  Arm OK = some copy OK (Paderborn preferred when
# both); arm DEAD = both copies DEAD.  A dead fit's dependent table job on the same cluster is cancelled (a failed fit never satisfies afterok).
# When no arm is pending: if >= 1 arm is OK and every OK arm's files are local, writes readout/arms.txt (ARM:pdb | ARM:th), submits r6_readout.sbatch (theorie) and arms r3_cpu_watch.sh.
# All arms dead: nothing submitted.  Gives up after GIVEUP s (default 60 h: theorie pp2 copy worst case 26 h fit + 2 h table + queue).
# Ssh uses -o HostName=login4.  Test hooks: R1_SBATCH, R1_CPUWATCH, R1_SCANCEL, R1_CHAIN_Q, R1_NOCANCEL, POLL, GIVEUP.
Q=${R1_CHAIN_Q:-/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r6}; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code
POLL=${POLL:-120}; GIVEUP=${GIVEUP:-216000}; SB=${R1_SBATCH:-sbatch}; CW=${R1_CPUWATCH:-$C/r3_cpu_watch.sh}; SC=${R1_SCANCEL:-scancel}; t0=$(date +%s)
ARMS="r6 m5"
tg() { case $1 in r6) echo 6;; m5) echo 5;; esac; }
v() { head -c 9 "$1" 2>/dev/null | awk '{print $1}'; }
say() { echo "$(date -Is) $*" | tee -a $Q/readout/chain_verdict.txt; }
cancelled=""
cstate() { # FITDIR TABDIR -> OK | DEAD | PENDING
  f=$(v $1/watcher_verdict.txt); t=$(v $2/watcher_verdict.txt)
  if [ "$f" = COMPLETED ] && [ "$t" = COMPLETED ]; then echo OK
  elif { [ -n "$f" ] && [ "$f" != COMPLETED ]; } || { [ -n "$t" ] && [ "$t" != COMPLETED ]; }; then echo DEAD
  else echo PENDING; fi; }
pstate() { [ -e $Q/$1/pdb_skip ] && { echo DEAD; return; }; cstate $Q/$1/fit $Q/$1/tab; }
tstate() { [ -e $Q/$1/fit_th/job_id.txt ] || { echo DEAD; return; }; cstate $Q/$1/fit_th $Q/$1/tab_th; }
allv() { for A in $ARMS; do echo -n "$A=pdb:$(v $Q/$A/fit/watcher_verdict.txt)/$(v $Q/$A/tab/watcher_verdict.txt),th:$(v $Q/$A/fit_th/watcher_verdict.txt)/$(v $Q/$A/tab_th/watcher_verdict.txt) "; done; }
while true; do
  now=$(date +%s)
  [ $((now-t0)) -gt $GIVEUP ] && { say "GIVEUP: no readout submitted (verdicts: $(allv))"; exit 1; }
  for A in $ARMS; do for K in pdb th; do
    [ $K = pdb ] && { FD=$Q/$A/fit; TD=$Q/$A/tab; } || { FD=$Q/$A/fit_th; TD=$Q/$A/tab_th; }
    f=$(v $FD/watcher_verdict.txt)
    if [ -n "$f" ] && [ "$f" != COMPLETED ] && [[ "$cancelled" != *"<$A$K>"* ]]; then
      cancelled="$cancelled<$A$K>"; tj=$(cat $TD/job_id.txt 2>/dev/null)
      if [ -z "${R1_NOCANCEL:-}" ] && [ -n "$tj" ]; then
        if [ $K = pdb ]; then ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; scancel $tj" >/dev/null 2>&1
        else $SC $tj >/dev/null 2>&1; fi
      fi
      say "$A $K fit verdict $f: dependent table job $tj cancelled (a failed fit never satisfies afterok)"
    fi
  done; done
  ok=""; pend=0
  for A in $ARMS; do
    sp=$(pstate $A); st=$(tstate $A)
    # no Paderborn job ever submitted for this arm: counts as dead once the theorie copy is settled
    [ ! -e $Q/$A/fit/job_id.txt ] && [ "$st" != PENDING ] && sp=DEAD
    if [ $sp = OK ]; then ok="$ok $A:pdb"; elif [ $st = OK ]; then ok="$ok $A:th"; elif [ $sp = DEAD ] && [ $st = DEAD ]; then :; else pend=1; fi
  done
  if [ $pend = 0 ]; then
    [ -z "$ok" ] && { say "NOT SUBMITTED: every arm dead ($(allv))"; exit 1; }
    miss=""
    for T in $ok; do A=${T%%:*}; K=${T##*:}
      [ $K = pdb ] && { FD=$Q/$A/fit; TD=$Q/$A/tab; } || { FD=$Q/$A/fit_th; TD=$Q/$A/tab_th; }
      G=$(tg $A); [ -s $TD/tab_t$G/round_1/la_table.npy ] || miss="$miss $T:tab_t$G"; [ -s $TD/theta_$G.npy ] || miss="$miss $T:theta_$G"; [ -s $TD/theta_shas.json ] || miss="$miss $T:theta_shas"; [ $A = m5 ] && { [ -s $TD/tab_t0/round_1/la_table.npy ] || miss="$miss $T:tab_t0"; }
      for N in 600 750 900 1050 1200 1350; do [ -s $FD/round_1/theta_step$N.npy ] || miss="$miss $T:theta_step$N"; done; [ -s $FD/round_1/theta.npy ] || miss="$miss $T:theta_final"
    done
    [ -n "$miss" ] && { say "NOT SUBMITTED: missing locally:$miss"; exit 1; }
    echo $ok > $Q/readout/arms.txt
    break
  fi
  sleep $POLL
done
OUTL=$($SB --parsable --no-requeue --output=$Q/readout/slurm-%j.out $C/r6_readout.sbatch); JOB=${OUTL%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { say "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > $Q/readout/job_id.txt; echo "r6_readout $JOB (arms$ok) $(date -Is)" >> $Q/job_ids.txt
setsid nohup $CW "$JOB" $Q/readout > $Q/readout/watcher.log 2>&1 < /dev/null &
echo $! > $Q/readout/watcher_pid.txt
say "SUBMITTED readout job $JOB (arms$ok; verdicts $(allv)); r3_cpu_watch pid $(cat $Q/readout/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/readout/watcher_pid.txt))"
