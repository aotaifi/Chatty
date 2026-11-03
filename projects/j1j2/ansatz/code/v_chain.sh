#!/bin/bash
# Gate V chain (durable, setsid; one-arm reduction of r5_chain.sh).  Waits for the Paderborn fit (Q/fit) and table (Q/tab) watcher verdicts.
# Both COMPLETED and every needed file local -> submits v_readout.sbatch (theorie) and arms r3_cpu_watch.sh on Q/readout.
# Fit verdict not COMPLETED -> cancels the dependent table job (a failed fit never satisfies afterok) and stops; table not COMPLETED -> stops.  GIVEUP s (default 24 h).
# Test hooks: R1_SBATCH, R1_CPUWATCH, R1_CHAIN_Q, R1_NOCANCEL, POLL, GIVEUP.
Q=${R1_CHAIN_Q:-/scratch/a/A.Otaifi/ansatz_runs/pfaffian_v}; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code
POLL=${POLL:-120}; GIVEUP=${GIVEUP:-86400}; SB=${R1_SBATCH:-sbatch}; CW=${R1_CPUWATCH:-$C/r3_cpu_watch.sh}; t0=$(date +%s)
v() { head -c 9 "$1" 2>/dev/null | awk '{print $1}'; }
say() { echo "$(date -Is) $*" | tee -a $Q/readout/chain_verdict.txt; }
mkdir -p $Q/readout
while true; do
  [ $(( $(date +%s) - t0 )) -gt $GIVEUP ] && { say "GIVEUP: fit $(v $Q/fit/watcher_verdict.txt) tab $(v $Q/tab/watcher_verdict.txt)"; exit 1; }
  f=$(v $Q/fit/watcher_verdict.txt); t=$(v $Q/tab/watcher_verdict.txt)
  if [ -n "$f" ] && [ "$f" != COMPLETED ]; then
    tj=$(cat $Q/tab/job_id.txt 2>/dev/null)
    [ -z "${R1_NOCANCEL:-}" ] && [ -n "$tj" ] && ssh -o BatchMode=yes -o ConnectTimeout=30 -o HostName=login4 paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; scancel $tj" >/dev/null 2>&1
    say "NOT SUBMITTED: fit verdict $f (table job $tj cancelled)"; exit 1; fi
  if [ -n "$t" ] && [ "$t" != COMPLETED ]; then say "NOT SUBMITTED: table verdict $t"; exit 1; fi
  [ "$f" = COMPLETED ] && [ "$t" = COMPLETED ] && break
  sleep $POLL
done
miss=""; [ -s $Q/tab/tab_t5/round_1/la_table.npy ] || miss="$miss tab_t5"; [ -s $Q/tab/theta_5.npy ] || miss="$miss theta_5"
for N in 600 750 900 1050 1200 1350; do [ -s $Q/fit/round_1/theta_step$N.npy ] || miss="$miss theta_step$N"; done
for f in theta.npy theta_bar.npy; do [ -s $Q/fit/round_1/$f ] || miss="$miss $f"; done
[ -n "$miss" ] && { say "NOT SUBMITTED: missing locally:$miss"; exit 1; }
OUTL=$($SB --parsable --no-requeue --output=$Q/readout/slurm-%j.out $C/v_readout.sbatch); JOB=${OUTL%%;*}
[[ "$JOB" =~ ^[0-9]+$ ]] || { say "sbatch failed: $OUTL"; exit 1; }
echo "$JOB" > $Q/readout/job_id.txt; echo "v_readout $JOB $(date -Is)" >> $Q/job_ids.txt
setsid nohup $CW "$JOB" $Q/readout > $Q/readout/watcher.log 2>&1 < /dev/null &
sleep 2; W=$(pgrep -f "r3_cpu_watch.sh $JOB" | head -1); echo "$W" > $Q/readout/watcher_pid.txt
say "SUBMITTED readout job $JOB; r3_cpu_watch pid $W ppid=$(ps -o ppid= -p $W 2>/dev/null)"
