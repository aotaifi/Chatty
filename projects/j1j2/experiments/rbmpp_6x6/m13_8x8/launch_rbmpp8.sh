#!/bin/bash
# M1.3 production launcher (STAGED, NOT RUN: the lead reads it, then runs it). Submits the Paderborn job, arms the durable
# watcher in the same action. Usage:  bash launch_rbmpp8.sh QOS HH:MM:SS [RUNNAME]
#   QOS: the lead's choice (express_hpcalot has a 20 H100-h monthly contingent; cont pends for days, normal is rejected).
#   Re-run with the same RUNNAME to continue after a TIMEOUT/preemption (the run resumes from state.npz).
set -euo pipefail
QOS=${1:?qos}; TL=${2:?time limit}; RUN=${3:-rb8_prod}
H=$(cd "$(dirname "$0")" && pwd); X=$H/..; BASE=/scratch/hpc-prf-ppdmrg/hpcalot/rbmpp8
LOCAL=${LOCAL_BASE:-/scratch/a/A.Otaifi/rbmpp8}/$RUN; DELTA=${DELTA:-/project/theorie/a/A.Otaifi/Chatty-organize/projects/j1j2/delta_today.md}
E="ssh -o BatchMode=yes -o HostName=login4"
# 1. sync code, specs, sbatch (code pinned by sha256 in RUN/run_metadata.txt at job start)
rsync -a -e "$E" $X/rbmpp_lib.py $X/rbmpp_train.py $X/rbmpp_eval.py paderborn:$BASE/code/
rsync -a -e "$E" $H/specs/ paderborn:$BASE/specs/
rsync -a -e "$E" $H/rbmpp8_prod.sbatch paderborn:$BASE/
mkdir -p $LOCAL; cp $H/ANALYSIS_BRIEF.md $LOCAL/ANALYSIS_BRIEF.md
# 2. submit (the job id is captured; nothing is claimed without it)
OUT=$($E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; cd $BASE; sbatch --parsable --qos=$QOS --time=$TL rbmpp8_prod.sbatch $RUN ${TRAIN_SPEC:-train_prod.json} ${EVAL_SPEC:-eval_prod.json}")
JOB=${OUT%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "submit failed: $OUT"; exit 1; }
echo "$JOB" > $LOCAL/JOBID
# 3. durable watcher: setsid orphan re-parented to PID 1; stall = no state/log update for STALL=600 s = 2 x the checkpoint interval (100 steps x 3 s = 5 min)
STALL=${STALL:-600}
DELTA=$DELTA POLL=${POLL:-300} setsid nohup bash $H/watch_rbmpp8.sh $JOB $RUN $LOCAL $STALL > $LOCAL/watcher.out 2>&1 < /dev/null &
sleep 2; WPID=$(pgrep -f "watch_rbmpp8.sh $JOB " | head -1); echo "watcher pid $WPID ppid $(ps -o ppid= -p $WPID | tr -d ' ')" | tee $LOCAL/WATCHER_PID
$E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; squeue -j $JOB -o '%i %T %r'"
echo "submitted job $JOB run $RUN; the owner waits with:"
echo "  until grep -q 'VERDICT job $JOB' $DELTA; do sleep 120; done; grep 'VERDICT job $JOB' $DELTA"
