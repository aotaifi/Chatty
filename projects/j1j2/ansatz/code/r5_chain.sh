#!/bin/bash
# Round-5 chain gate (durable, setsid; adapted from r4_chain.sh).  Slurm dependencies do not cross clusters: this waits until every arm (pp2_s1 pp2_s2 ctl_s1 ctl_s2, all Paderborn)
# is settled: OK = fit AND tab watcher verdicts COMPLETED; DEAD = any verdict != COMPLETED (a dead fit's dependent tab job is cancelled: a failed fit never satisfies afterok).
# Then, if >= 1 arm is OK and every OK arm's files are local, writes readout/arms.txt (OK arms, pp2 first), submits r5_readout.sbatch (theorie cluster) and arms r3_cpu_watch.sh on it.
# All arms dead: nothing submitted.  Gives up after GIVEUP s (default 48 h).  Ssh uses -o HostName=login4.  Test hooks: R1_SBATCH, R1_CPUWATCH, R1_CHAIN_Q, R1_NOCANCEL, POLL, GIVEUP.
Q=${R1_CHAIN_Q:-/scratch/a/A.Otaifi/ansatz_runs/pfaffian_r5}; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code
POLL=${POLL:-120}; GIVEUP=${GIVEUP:-172800}; SB=${R1_SBATCH:-sbatch}; CW=${R1_CPUWATCH:-$C/r3_cpu_watch.sh}; t0=$(date +%s)
ARMS="pp2_s1 pp2_s2 ctl_s1 ctl_s2"
v() { head -c 9 "$1" 2>/dev/null | awk '{print $1}'; }
say() { echo "$(date -Is) $*" | tee -a $Q/readout/chain_verdict.txt; }
cancelled=""
cstate() { # ARM -> OK | DEAD | PENDING
  f=$(v $Q/$1/fit/watcher_verdict.txt); t=$(v $Q/$1/tab/watcher_verdict.txt)
  if [ "$f" = COMPLETED ] && [ "$t" = COMPLETED ]; then echo OK
  elif { [ -n "$f" ] && [ "$f" != COMPLETED ]; } || { [ -n "$t" ] && [ "$t" != COMPLETED ]; }; then echo DEAD
  else echo PENDING; fi; }
allv() { for A in $ARMS; do echo -n "$A=$(v $Q/$A/fit/watcher_verdict.txt)/$(v $Q/$A/tab/watcher_verdict.txt) "; done; }
while true; do
  now=$(date +%s)
  [ $((now-t0)) -gt $GIVEUP ] && { say "GIVEUP: no readout submitted (verdicts: $(allv))"; exit 1; }
  for A in $ARMS; do
    f=$(v $Q/$A/fit/watcher_verdict.txt)
    if [ -n "$f" ] && [ "$f" != COMPLETED ] && [[ "$cancelled" != *"<$A>"* ]]; then
      cancelled="$cancelled<$A>"; tj=$(cat $Q/$A/tab/job_id.txt 2>/dev/null)
      if [ -z "${R1_NOCANCEL:-}" ] && [ -n "$tj" ]; then
        ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; scancel $tj" >/dev/null 2>&1
      fi
      say "$A fit verdict $f: dependent table job $tj cancelled (a failed fit never satisfies afterok)"
    fi
  done
  ok=""; pend=0; for A in $ARMS; do s=$(cstate $A); [ $s = OK ] && ok="$ok $A"; [ $s = PENDING ] && pend=1; done
  if [ $pend = 0 ]; then
    [ -z "$ok" ] && { say "NOT SUBMITTED: every arm dead ($(allv))"; exit 1; }
    miss=""
    for A in $ok; do FD=$Q/$A/fit; TD=$Q/$A/tab
      [ -s $TD/tab_t5/round_1/la_table.npy ] || miss="$miss $A:tab_t5"; [ -s $TD/theta_5.npy ] || miss="$miss $A:theta_5"; [ -s $TD/theta_shas.json ] || miss="$miss $A:theta_shas"
      for N in 600 750 900 1050 1200 1350; do [ -s $FD/round_1/theta_step$N.npy ] || miss="$miss $A:theta_step$N"; done; [ -s $FD/round_1/theta.npy ] || miss="$miss $A:theta_final"
    done
    [ -n "$miss" ] && { say "NOT SUBMITTED: missing locally:$miss"; exit 1; }
    echo $ok > $Q/readout/arms.txt
    break
  fi
  sleep $POLL
done
OUTL=$($SB --parsable --no-requeue --output=$Q/readout/slurm-%j.out $C/r5_readout.sbatch); JOB=${OUTL%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { say "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > $Q/readout/job_id.txt; echo "r5_readout $JOB (arms$ok) $(date -Is)" >> $Q/job_ids.txt
setsid nohup $CW "$JOB" $Q/readout > $Q/readout/watcher.log 2>&1 < /dev/null &
echo $! > $Q/readout/watcher_pid.txt
say "SUBMITTED readout job $JOB (arms$ok; verdicts $(allv)); r3_cpu_watch pid $(cat $Q/readout/watcher_pid.txt) ppid=$(ps -o ppid= -p $(cat $Q/readout/watcher_pid.txt))"
