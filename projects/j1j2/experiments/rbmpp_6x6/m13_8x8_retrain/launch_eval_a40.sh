#!/bin/bash
# STAGED, NOT SUBMITTED. s307 'full' evaluation chains seeds 55 56 on theorie A40s (cip, gpu:a40:1), replacing the queued Paderborn pair.
# Probe 17903352 (A40, cip-cl-nv01): 14.6 s/sweep -> ~6.5 h per chain (1563 sweeps + 100 therm); --time = 2x = 13h (cap 48h).
# One job per chain; one durable setsid watcher per job (PPID 1), armed in the same action; each ends with job_finish.sh --wake none.
# STALL 1500 s: PROGRESS every 20 sweeps = ~295 s, first one after ~770 s (100 therm sweeps), so 1500 s = 5 intervals.
set -euo pipefail
H=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/rbmpp_6x6/m13_8x8_retrain
LB=/scratch/a/A.Otaifi/rbmpp8/rb8_retrain/s307
DELTA=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md
TL=${TL:-13:00:00}; STALL=1500; JOBS=""
[ "$(sha256sum $LB/train/params_final.npz | cut -c1-8)" = b86a0f36 ] || { echo "params sha mismatch"; exit 1; }
for S in 55 56; do L=$LB/ev_full_$S
  [ -f $L/ANALYSIS_BRIEF.md ] && [ -f $H/specs_eval/s307_full_$S.json ] || { echo "missing brief/spec for $S"; exit 1; }
  JOB=$(sbatch --parsable --no-requeue --time=$TL --output=$L/slurm-%j.out $H/eval_a40.sbatch $S)
  [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "submit failed for $S: $JOB"; continue; }
  echo "$JOB" >> $L/JOBID
  DELTA=$DELTA POLL=120 setsid nohup bash $H/watch_local.sh $JOB $L $STALL > $L/watcher.out 2>&1 < /dev/null &
  sleep 2; WPID=$(pgrep -f "watch_local.sh $JOB " | head -1); echo "$S job $JOB watcher pid $WPID ppid $(ps -o ppid= -p $WPID)" | tee -a $L/WATCHER_PID
  JOBS="$JOBS $S=$JOB/w$WPID"
done
squeue -u $USER -n rb8evA40 -o "%i %T %r %S"
[ -n "$JOBS" ] && echo "$(date '+%Y-%m-%d %H:%M') j1j2 base-retrain worker: s307 'full' A40 chains submitted (cip, --time $TL):$JOBS; run dirs $LB/ev_full_<seed>; gate: eval_combine_a40.sh" >> $DELTA
echo "submitted:$JOBS"
