#!/bin/bash
# M5' RERUN chain gate (= m5p_chain.sh with Q=pfaffian_m5p2, m5p2_read.sbatch, and the plateau tables tab_s1200/tab_s1500 required; theorie copy only: create Q/round/pdb_skip) for ONE phase (usage: m5p_chain.sh PHASE; durable, setsid; = r6_chain.sh with a single "arm" = PHASE).  Slurm dependencies do not cross clusters.  Two copies race:
# Paderborn (PHASE/fit + PHASE/tab) and theorie (PHASE/fit_th + PHASE/tab_th).  Copy OK = fit and table watcher verdicts COMPLETED; DEAD = either not COMPLETED (Paderborn also when
# PHASE/pdb_skip exists or never submitted while the theorie copy is OK).  A dead fit's dependent table job is cancelled.  When one copy is OK and its files are local: writes
# readout_PHASE/arms.txt ("PHASE:pdb|th") and submits the theorie CPU job: rel* -> m5p_read.sbatch OUT rel PHASE;  round -> m5p_read.sbatch OUT final  (OUT = readout_PHASE),
# arms r3_cpu_watch.sh on it.  Both copies dead: nothing submitted.  GIVEUP s (default 60 h).  Ssh uses -o HostName=login4.  Test hooks: R1_SBATCH, R1_CPUWATCH, R1_SCANCEL, R1_CHAIN_Q, R1_NOCANCEL, POLL, GIVEUP.
PH=${1:?phase}; Q=${R1_CHAIN_Q:-/scratch/a/A.Otaifi/ansatz_runs/pfaffian_m5p2}; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=$Q/readout_$PH
POLL=${POLL:-120}; GIVEUP=${GIVEUP:-216000}; SB=${R1_SBATCH:-sbatch}; CW=${R1_CPUWATCH:-$C/r3_cpu_watch.sh}; SC=${R1_SCANCEL:-scancel}; t0=$(date +%s)
case $PH in rel*) TAG=rel; RMODE="rel $PH";; round) TAG=t5; RMODE=final;; *) echo bad phase; exit 2;; esac
mkdir -p $O
v() { head -c 9 "$1" 2>/dev/null | awk '{print $1}'; }
say() { echo "$(date -Is) $*" | tee -a $O/chain_verdict.txt; }
cstate() { f=$(v $1/watcher_verdict.txt); t=$(v $2/watcher_verdict.txt)
  if [ "$f" = COMPLETED ] && [ "$t" = COMPLETED ]; then echo OK
  elif { [ -n "$f" ] && [ "$f" != COMPLETED ]; } || { [ -n "$t" ] && [ "$t" != COMPLETED ]; }; then echo DEAD; else echo PENDING; fi; }
allv() { echo "pdb:$(v $Q/$PH/fit/watcher_verdict.txt)/$(v $Q/$PH/tab/watcher_verdict.txt),th:$(v $Q/$PH/fit_th/watcher_verdict.txt)/$(v $Q/$PH/tab_th/watcher_verdict.txt)"; }
cancelled=""
while true; do
  [ $(( $(date +%s)-t0 )) -gt $GIVEUP ] && { say "GIVEUP: no readout submitted ($(allv))"; exit 1; }
  for K in pdb th; do
    [ $K = pdb ] && { FD=$Q/$PH/fit; TD=$Q/$PH/tab; } || { FD=$Q/$PH/fit_th; TD=$Q/$PH/tab_th; }
    f=$(v $FD/watcher_verdict.txt)
    if [ -n "$f" ] && [ "$f" != COMPLETED ] && [[ "$cancelled" != *"<$K>"* ]]; then
      cancelled="$cancelled<$K>"; tj=$(cat $TD/job_id.txt 2>/dev/null)
      if [ -z "${R1_NOCANCEL:-}" ] && [ -n "$tj" ]; then
        if [ $K = pdb ]; then ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; scancel $tj" >/dev/null 2>&1; else $SC $tj >/dev/null 2>&1; fi
      fi
      say "$PH $K fit verdict $f: dependent table job $tj cancelled (a failed fit never satisfies afterok)"
    fi
  done
  if [ -e $Q/$PH/pdb_skip ]; then sp=DEAD; else sp=$(cstate $Q/$PH/fit $Q/$PH/tab); fi
  if [ -e $Q/$PH/fit_th/job_id.txt ]; then st=$(cstate $Q/$PH/fit_th $Q/$PH/tab_th); else st=DEAD; fi
  [ ! -e $Q/$PH/fit/job_id.txt ] && [ "$st" != PENDING ] && sp=DEAD
  if [ $sp = OK ]; then WIN=pdb; break; elif [ $st = OK ]; then WIN=th; break; elif [ $sp = DEAD ] && [ $st = DEAD ]; then say "NOT SUBMITTED: both copies dead ($(allv))"; exit 1; fi
  sleep $POLL
done
[ $WIN = pdb ] && { FD=$Q/$PH/fit; TD=$Q/$PH/tab; } || { FD=$Q/$PH/fit_th; TD=$Q/$PH/tab_th; }
miss=""; [ -s $TD/tab_$TAG/round_1/la_table.npy ] || miss="$miss tab_$TAG"; [ -s $TD/theta_${TAG#t}.npy ] || miss="$miss theta"; [ -s $TD/theta_shas.json ] || miss="$miss theta_shas"
[ -s $FD/round_1/theta.npy ] || miss="$miss theta_final"; [ -s $FD/cx_r1.json ] || miss="$miss cx_r1.json"
if [ $PH = round ]; then for N in 600 750 900 1050 1200 1350; do [ -s $FD/round_1/theta_step$N.npy ] || miss="$miss theta_step$N"; done; for T in s1200 s1500; do [ -s $TD/tab_$T/round_1/la_table.npy ] || miss="$miss tab_$T"; done; fi
[ -n "$miss" ] && { say "NOT SUBMITTED: missing locally:$miss"; exit 1; }
echo "$PH:$WIN" > $O/arms.txt
OUTL=$($SB --parsable --no-requeue --output=$O/slurm-%j.out $C/m5p2_read.sbatch $O $RMODE); JOB=${OUTL%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { say "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > $O/job_id.txt; echo "m5p_read_$PH $JOB ($WIN; verdicts $(allv)) $(date -Is)" >> $Q/job_ids.txt
setsid nohup $CW "$JOB" $O > $O/watcher.log 2>&1 < /dev/null &
echo $! > $O/watcher_pid.txt
say "SUBMITTED readout job $JOB (phase $PH, copy $WIN; verdicts $(allv)); r3_cpu_watch pid $(cat $O/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $O/watcher_pid.txt))"
