#!/bin/bash
# Gate V chain (durable, setsid; one-arm reduction of r5_chain.sh with the theorie race copy).  Copies: Paderborn (Q/fit + Q/tab) and theorie A40 (Q/fit_th + Q/tab_th).
# A copy is OK when its fit and table watcher verdicts are COMPLETED; DEAD when either verdict is not COMPLETED, or (Paderborn) Q/pdb_skip exists, or (theorie) no job was submitted
# while the other copy is settled.  A dead fit's dependent table job on the same cluster is cancelled (afterok never fires).  First OK copy (Paderborn preferred if both):
# writes Q/readout/winner.txt (pdb|th), checks the files are local, submits v_readout.sbatch (theorie) and arms r3_cpu_watch.sh.  Both dead: nothing submitted.  GIVEUP s (default 60 h).
# Test hooks: R1_SBATCH, R1_CPUWATCH, R1_CHAIN_Q, R1_NOCANCEL, R1_SCANCEL, POLL, GIVEUP.
Q=${R1_CHAIN_Q:-/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v}; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code
POLL=${POLL:-120}; GIVEUP=${GIVEUP:-216000}; SB=${R1_SBATCH:-sbatch}; CW=${R1_CPUWATCH:-$C/r3_cpu_watch.sh}; SC=${R1_SCANCEL:-scancel}; t0=$(date +%s)
v() { head -c 12 "$1" 2>/dev/null | awk '{print $1}'; }
say() { echo "$(date -Is) $*" | tee -a $Q/readout/chain_verdict.txt; }
mkdir -p $Q/readout; cancelled=""
cst() { f=$(v $1/watcher_verdict.txt); t=$(v $2/watcher_verdict.txt)
  if [ "$f" = COMPLETED ] && [ "$t" = COMPLETED ]; then echo OK
  elif { [ -n "$f" ] && [ "$f" != COMPLETED ]; } || { [ -n "$t" ] && [ "$t" != COMPLETED ]; }; then echo DEAD; else echo PENDING; fi; }
allv() { echo "pdb $(v $Q/fit/watcher_verdict.txt)/$(v $Q/tab/watcher_verdict.txt) th $(v $Q/fit_th/watcher_verdict.txt)/$(v $Q/tab_th/watcher_verdict.txt)"; }
while true; do
  [ $(( $(date +%s) - t0 )) -gt $GIVEUP ] && { say "GIVEUP: $(allv)"; exit 1; }
  for K in pdb th; do
    [ $K = pdb ] && { FD=$Q/fit; TD=$Q/tab; } || { FD=$Q/fit_th; TD=$Q/tab_th; }
    f=$(v $FD/watcher_verdict.txt)
    if [ -n "$f" ] && [ "$f" != COMPLETED ] && [[ "$cancelled" != *"<$K>"* ]]; then
      cancelled="$cancelled<$K>"; tj=$(cat $TD/job_id.txt 2>/dev/null)
      if [ -z "${R1_NOCANCEL:-}" ] && [ -n "$tj" ]; then
        if [ $K = pdb ]; then ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; scancel $tj" >/dev/null 2>&1
        else $SC $tj >/dev/null 2>&1; fi
      fi
      say "$K fit verdict $f: dependent table job $tj cancelled"
    fi
  done
  sp=$( [ -e $Q/pdb_skip ] && echo DEAD || cst $Q/fit $Q/tab ); st=$(cst $Q/fit_th $Q/tab_th)
  [ ! -e $Q/fit/job_id.txt ] && [ "$st" != PENDING ] && sp=DEAD
  [ ! -e $Q/fit_th/job_id.txt ] && [ "$sp" != PENDING ] && st=DEAD
  if [ $sp = OK ]; then W=pdb; break; elif [ $st = OK ]; then W=th; break; elif [ $sp = DEAD ] && [ $st = DEAD ]; then say "NOT SUBMITTED: both copies dead ($(allv))"; exit 1; fi
  sleep $POLL
done
[ $W = pdb ] && { FD=$Q/fit; TD=$Q/tab; } || { FD=$Q/fit_th; TD=$Q/tab_th; }
miss=""; [ -s $TD/tab_t5/round_1/la_table.npy ] || miss="$miss tab_t5"; [ -s $TD/theta_5.npy ] || miss="$miss theta_5"
for N in 600 750 900 1050 1200 1350; do [ -s $FD/round_1/theta_step$N.npy ] || miss="$miss theta_step$N"; done
for f in theta.npy theta_bar.npy; do [ -s $FD/round_1/$f ] || miss="$miss $f"; done
[ -n "$miss" ] && { say "NOT SUBMITTED: winner $W, missing locally:$miss"; exit 1; }
echo $W > $Q/readout/winner.txt
OUTL=$($SB --parsable --no-requeue --output=$Q/readout/slurm-%j.out $C/v_readout.sbatch); JOB=${OUTL%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { say "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > $Q/readout/job_id.txt; echo "v_readout $JOB (winner $W) $(date -Is)" >> $Q/job_ids.txt
setsid nohup $CW "$JOB" $Q/readout > $Q/readout/watcher.log 2>&1 < /dev/null &
sleep 2; WP=$(pgrep -f "r3_cpu_watch.sh $JOB" | head -1); echo "$WP" > $Q/readout/watcher_pid.txt
say "SUBMITTED readout job $JOB (winner $W; $(allv)); r3_cpu_watch pid $WP ppid=$(ps -o ppid= -p $WP 2>/dev/null)"
