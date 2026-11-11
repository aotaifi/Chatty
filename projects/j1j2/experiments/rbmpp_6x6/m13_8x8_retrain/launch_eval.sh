#!/bin/bash
# Evaluations of the retrained parameters: one job per chain, one durable watcher per job armed in the same action.
#   GROUP=simple bash launch_eval.sh   -> 4 chains (s306: 41 42, s307: 43 44), express, --time 00:40:00
#   GROUP=full   bash launch_eval.sh   -> 4 chains (s306: 51 52, s307: 53 54), default QoS (cont), --time 04:00:00
set -euo pipefail
H=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/rbmpp_6x6/m13_8x8_retrain
B=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8; LB=/scratch/a/A.Otaifi/rbmpp8/rb8_retrain
DELTA=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md
E="ssh -o BatchMode=yes -o HostName=login4"
case ${GROUP:?GROUP=simple|full} in
  simple) CH="s306:41 s306:42 s307:43 s307:44"; QOS="--qos=express_hpcalot"; TL=00:40:00; STALL=600;;
  full)   CH="s306:51 s306:52 s307:53 s307:54"; QOS="";                    TL=04:00:00; STALL=900;;
esac
rsync -a -e "$E" $H/specs_eval/ paderborn:$B/specs_eval/
rsync -a -e "$E" $H/eval.sbatch paderborn:$B/
JOBS=""
for c in $CH; do P=${c%%:*}; S=${c##*:}; L=$LB/$P/ev_${GROUP}_$S
  [ -f $L/ANALYSIS_BRIEF.md ] || { echo "missing $L/ANALYSIS_BRIEF.md"; exit 1; }
  OUT=$($E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; cd $B; sbatch --parsable $QOS --time=$TL eval.sbatch $P $GROUP $S")
  JOB=${OUT%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "submit failed for $c: $OUT"; continue; }
  echo "$JOB" >> $L/JOBID
  DELTA=$DELTA POLL=180 setsid nohup bash $H/watch_probe.sh $JOB rb8_retrain/$P/ev_${GROUP}_$S $L $STALL > $L/watcher.out 2>&1 < /dev/null &
  sleep 2; WPID=$(pgrep -f "watch_probe.sh $JOB " | head -1); echo "$c job $JOB watcher pid $WPID" | tee -a $L/WATCHER_PID
  JOBS="$JOBS $P/$S=$JOB/w$WPID"
done
$E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; sleep 5; squeue -u hpcalot -n rb8ev --start -o '%i %T %r %q %S'"
[ -n "$JOBS" ] && echo "$(date '+%Y-%m-%d %H:%M') j1j2 base-retrain worker: '$GROUP' evaluations submitted (Paderborn H100 ${QOS:-default QoS}, --time $TL, 1024 chains x 1.6e6 samples each, params_final s306 sha 45fa54e6 / s307 sha b86a0f36):$JOBS; run dirs $LB/<pset>/ev_${GROUP}_<seed>; gate: eval_combine.sh" >> $DELTA
echo "submitted:$JOBS"
