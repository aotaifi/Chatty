#!/bin/bash
# S0b lr scan: arms lr0.05, lr0.1 at shift 1e-3 (specs_shift/lr*.json); same mechanics as launch_shiftscan.sh
# STAGED 2026-11-09; run only after the lead's go:  bash launch_shiftscan.sh
set -euo pipefail
H=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/rbmpp_6x6/m13_8x8_retrain
B=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8; LB=/scratch/a/A.Otaifi/rbmpp8/rb8_shiftscan
DELTA=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md
E="ssh -o BatchMode=yes -o HostName=login4"
rsync -a -e "$E" $H/specs_shift/ paderborn:$B/specs_shiftscan/
rsync -a -e "$E" $H/shiftscan.sbatch paderborn:$B/
JOBS=""
for A in ${ARMS:-lr0.05 lr0.1}; do
  OUT=$($E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; cd $B; sbatch --parsable --qos=express_hpcalot --time=01:10:00 shiftscan.sbatch $A")
  JOB=${OUT%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "submit failed for $A: $OUT"; continue; }
  echo "$JOB" > $LB/$A/JOBID
  DELTA=$DELTA POLL=180 setsid nohup bash $H/watch_probe.sh $JOB rb8_shiftscan/$A $LB/$A 600 > $LB/$A/watcher.out 2>&1 < /dev/null &
  sleep 2; WPID=$(pgrep -f "watch_probe.sh $JOB " | head -1); echo "$A job $JOB watcher pid $WPID" | tee $LB/$A/WATCHER_PID
  JOBS="$JOBS $A=$JOB/w$WPID"
done
$E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; sleep 5; squeue -u hpcalot -n rb8sh -o '%i %T %r %q'"
[ -n "$JOBS" ] && echo "$(date '+%Y-%m-%d %H:%M') j1j2 base-retrain worker: S0b SR lr scan submitted (shift 1e-3) (express, ${#ARMS} chars of arms, 1 H100 each, --time 01:10:00, 280 steps each, warm rb8_prod sha 0520c5be, 'simple'):$JOBS; run dirs $LB/<arm>" >> $DELTA
echo "submitted:$JOBS"
