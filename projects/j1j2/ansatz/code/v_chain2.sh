#!/bin/bash
# Gate V chain 2 (durable, setsid).  Copy of v_chain.sh (which is NOT edited: the first chain died with the 2026-11-04 08:30 reboot of m51 and is not running) with the FIRST-COMPLETED rule
# (PI 2026-11-04): copies are the fresh Paderborn copy "pdb2" (Q/pdb2/fit + Q/pdb2/tab) and the theorie A40 copy "th" (Q/fit_th + Q/tab_th).  The first copy whose fit AND table watcher verdicts
# are COMPLETED wins (tie in the same poll: Paderborn); its table is read out (winner.txt = pdb2|th; v_bare.py maps both).  The OTHER copy is NEVER cancelled by this script when it is
# merely still running or pending: one line goes to delta_today.md and chain2_verdict.txt.  Only a dead copy's own dependent table job (afterok never fires) is cancelled, as in v_chain.sh.
# The first copy 4107178/4107179 (Q/fit, Q/tab; cancelled by the old race rule) is ignored.  GIVEUP s (default 60 h).  Both dead: nothing submitted.
# Test hooks: R1_SBATCH, R1_CPUWATCH, R1_CHAIN_Q, R1_NOCANCEL, R1_SCANCEL, R1_DELTA, POLL, GIVEUP.
Q=${R1_CHAIN_Q:-/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v}; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code
POLL=${POLL:-120}; GIVEUP=${GIVEUP:-216000}; SB=${R1_SBATCH:-sbatch}; CW=${R1_CPUWATCH:-$C/r3_cpu_watch.sh}; SC=${R1_SCANCEL:-scancel}; t0=$(date +%s)
v() { head -c 12 "$1" 2>/dev/null | awk '{print $1}'; }
DELTA=${R1_DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
say() { echo "$(date -Is) $*" | tee -a $Q/readout/chain2_verdict.txt; }
mkdir -p $Q/readout; cancelled=""
cst() { f=$(v $1/watcher_verdict.txt); t=$(v $2/watcher_verdict.txt)
  if [ "$f" = COMPLETED ] && [ "$t" = COMPLETED ]; then echo OK
  elif { [ -n "$f" ] && [ "$f" != COMPLETED ]; } || { [ -n "$t" ] && [ "$t" != COMPLETED ]; }; then echo DEAD; else echo PENDING; fi; }
allv() { echo "pdb2 $(v $Q/pdb2/fit/watcher_verdict.txt)/$(v $Q/pdb2/tab/watcher_verdict.txt) th $(v $Q/fit_th/watcher_verdict.txt)/$(v $Q/tab_th/watcher_verdict.txt)"; }
while true; do
  [ $(( $(date +%s) - t0 )) -gt $GIVEUP ] && { say "GIVEUP: $(allv)"; exit 1; }
  for K in pdb2 th; do
    [ $K = pdb2 ] && { FD=$Q/pdb2/fit; TD=$Q/pdb2/tab; } || { FD=$Q/fit_th; TD=$Q/tab_th; }
    f=$(v $FD/watcher_verdict.txt)
    if [ -n "$f" ] && [ "$f" != COMPLETED ] && [[ "$cancelled" != *"<$K>"* ]]; then
      cancelled="$cancelled<$K>"; tj=$(cat $TD/job_id.txt 2>/dev/null)
      if [ -z "${R1_NOCANCEL:-}" ] && [ -n "$tj" ]; then
        if [ $K = pdb2 ]; then ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; scancel $tj" >/dev/null 2>&1
        else $SC $tj >/dev/null 2>&1; fi
      fi
      say "$K fit verdict $f: dependent table job $tj cancelled"
    fi
  done
  sp=$( [ -e $Q/pdb_skip2 ] && echo DEAD || cst $Q/pdb2/fit $Q/pdb2/tab ); st=$(cst $Q/fit_th $Q/tab_th)
  [ ! -e $Q/pdb2/fit/job_id.txt ] && [ "$st" != PENDING ] && sp=DEAD
  [ ! -e $Q/fit_th/job_id.txt ] && [ "$sp" != PENDING ] && st=DEAD
  if [ $sp = OK ]; then W=pdb2; break; elif [ $st = OK ]; then W=th; break; elif [ $sp = DEAD ] && [ $st = DEAD ]; then say "NOT SUBMITTED: both copies dead ($(allv))"; exit 1; fi
  sleep $POLL
done
[ $W = pdb2 ] && { FD=$Q/pdb2/fit; TD=$Q/pdb2/tab; } || { FD=$Q/fit_th; TD=$Q/tab_th; }
if [ $W = pdb2 ]; then OTH=th; OST=$st; else OTH=pdb2; OST=$sp; fi
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
if [ "$OST" = PENDING ]; then
  L="$(date '+%Y-%m-%d %H:%M') [pfaffian V chain2] copy $W finished first (readout $JOB); the other copy $OTH is still running/pending and is NOT cancelled ($(allv))"
  echo "$L" >> "$DELTA"; say "$L"
fi
