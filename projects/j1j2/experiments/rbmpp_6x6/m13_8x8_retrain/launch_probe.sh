#!/bin/bash
# Submit the rb8_pgprobe cost probe on Paderborn express and arm the durable watcher in the same action.
set -euo pipefail
H=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/rbmpp_6x6/m13_8x8_retrain
B=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8; LOCAL=/scratch/a/A.Otaifi/rbmpp8/rb8_pgprobe
DELTA=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md
E="ssh -o BatchMode=yes -o HostName=login4"
rsync -a -e "$E" $H/specs/ paderborn:$B/specs_pgprobe/
rsync -a -e "$E" $H/probe.sbatch paderborn:$B/
OUT=$($E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; cd $B; sbatch --parsable --qos=express_hpcalot --time=01:00:00 probe.sbatch")
JOB=${OUT%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "submit failed: $OUT"; exit 1; }
echo "$JOB" > $LOCAL/JOBID
DELTA=$DELTA POLL=180 setsid nohup bash $H/watch_probe.sh $JOB rb8_pgprobe $LOCAL 900 > $LOCAL/watcher.out 2>&1 < /dev/null &
sleep 2; WPID=$(pgrep -f "watch_probe.sh $JOB " | head -1); echo "watcher pid $WPID ppid $(ps -o ppid= -p $WPID | tr -d ' ')" | tee $LOCAL/WATCHER_PID
$E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; sleep 5; squeue -j $JOB -o '%i %T %r %q'; sacct -j $JOB -X -o JobID,State,Reason%20"
echo "submitted job $JOB"
