#!/bin/bash
# Production retrain of the 8x8 RBM+PP base: seeds 306 and 307 as separate Paderborn H100 jobs, DEFAULT QoS (no --qos;
# Paderborn assigns cont/lowcont itself), --time 20:00:00, and one durable watcher per job (watch_retrain.sh: completion,
# failure, disappearance, stall = 2 x the 50-step checkpoint interval, and the registered kill at step 1000) armed in the same action.
# Resume after a TIMEOUT: SEEDS=s306 bash launch_retrain.sh (same run dir; train/state.npz holds params, chains, RNG, step).
set -euo pipefail
H=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/rbmpp_6x6/m13_8x8_retrain
B=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8; LB=/scratch/a/A.Otaifi/rbmpp8/rb8_retrain
DELTA=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md
STALL=${STALL:-1200}
E="ssh -o BatchMode=yes -o HostName=login4"
rsync -a -e "$E" $H/specs_prod/ paderborn:$B/specs_retrain/
rsync -a -e "$E" $H/retrain.sbatch paderborn:$B/
JOBS=""
for A in ${SEEDS:-s306 s307}; do
  [ -f $LB/$A/ANALYSIS_BRIEF.md ] || { echo "missing $LB/$A/ANALYSIS_BRIEF.md"; exit 1; }
  OUT=$($E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; cd $B; sbatch --parsable --time=20:00:00 retrain.sbatch $A")
  JOB=${OUT%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "submit failed for $A: $OUT"; continue; }
  echo "$JOB" >> $LB/$A/JOBID
  DELTA=$DELTA POLL=300 setsid nohup bash $H/watch_retrain.sh $JOB rb8_retrain/$A $LB/$A $STALL > $LB/$A/watcher.out 2>&1 < /dev/null &
  sleep 2; WPID=$(pgrep -f "watch_retrain.sh $JOB " | head -1); echo "$A job $JOB watcher pid $WPID ppid $(ps -o ppid= -p $WPID | tr -d ' ')" | tee -a $LB/$A/WATCHER_PID
  JOBS="$JOBS $A=$JOB/w$WPID"
done
$E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; sleep 5; squeue -u hpcalot -n rb8rt --start -o '%i %T %r %q %S'"
[ -n "$JOBS" ] && echo "$(date '+%Y-%m-%d %H:%M') j1j2 base-retrain worker: PRODUCTION retrain submitted (Paderborn H100, default QoS, --time 20:00:00; warm rb8_prod sha 0520c5be, 'simple', lr 0.05->0.005, shift 1e-3, 16384 samples, 3000 steps; registered kill at step 1000):$JOBS; run dirs $LB/<seed>" >> $DELTA
echo "submitted:$JOBS"
