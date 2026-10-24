#!/bin/bash
# M1 LxL production launcher for the RBM+PP VMC training (L = 10, 12). STAGED, NOT RUN: the lead reads it, then runs it.
# DRY RUN by default (prints every ssh/rsync/sbatch/watcher command, writes nothing remote); GO=1 submits and arms the durable watcher in the same action.
#   [GO=1] bash launch_rbmpp_L.sh L QOS HH:MM:SS [RUNNAME]
#   QOS: the lead's choice (express_hpcalot ~1 H100-h left in October; cont / lowcont start about 6-11 November; normal is rejected).
#   Re-run with the same RUNNAME to continue after a TIMEOUT/preemption (the run resumes from state.npz, bit-identical).
# Generalisation of ../m13_8x8/launch_rbmpp8.sh (unchanged). Specs: specs/train_prod_L$L.json, specs/eval_prod_L$L.json (sizing in ANALYSIS_BRIEF.md).
set -euo pipefail
L=${1:?L (10 or 12)}; QOS=${2:?qos}; TL=${3:?time limit HH:MM:SS}; RUN=${4:-rbL${L}_prod}; GO=${GO:-0}
H=$(cd "$(dirname "$0")" && pwd); X=$H/..; BASE=/scratch/hpc-prf-ppdmrg/hpcalot/rbmppL
LOCAL=${LOCAL_BASE:-/scratch/a/A.Otaifi/rbmppL/prod}/$RUN; DELTA=${DELTA:-/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/delta_today.md}
E="ssh -o BatchMode=yes -o HostName=login4"
TS=${TRAIN_SPEC:-train_prod_L$L.json}; ES=${EVAL_SPEC:-eval_prod_L$L.json}
[ -f $H/specs/$TS ] && [ -f $H/specs/$ES ] || { echo "missing spec $H/specs/$TS or $ES"; exit 1; }
# stall limit = 2 x checkpoint interval (ckpt_every steps x measured s/step, see ANALYSIS_BRIEF.md), in seconds
case $L in 10) STALL_DEF=900;; 12) STALL_DEF=2400;; *) echo "L must be 10 or 12"; exit 1;; esac
STALL=${STALL:-$STALL_DEF}
run() { if [ "$GO" = 1 ]; then eval "$@"; else echo "DRY: $*"; fi; }
# 1. sync code, specs, sbatch (code pinned by sha256 in RUN/run_metadata.txt at job start)
run "mkdir -p $LOCAL"
run "$E paderborn 'mkdir -p $BASE/code $BASE/specs $BASE/runs'"
run "rsync -a -e '$E' $X/rbmpp_lib.py $X/rbmpp_train.py $X/rbmpp_eval.py paderborn:$BASE/code/"
run "rsync -a -e '$E' $H/specs/ paderborn:$BASE/specs/"
run "rsync -a -e '$E' $H/rbmppL_prod.sbatch paderborn:$BASE/"
run "sed 's#@L@#$L#g' $H/ANALYSIS_BRIEF.md > $LOCAL/ANALYSIS_BRIEF.md"
# 2. submit (the job id is captured; nothing is claimed without it)
if [ "$GO" != 1 ]; then
  echo "DRY: $E paderborn 'cd $BASE; sbatch --parsable --qos=$QOS --time=$TL rbmppL_prod.sbatch $RUN $TS $ES'"
  echo "DRY: setsid nohup bash $H/watch_rbmppL.sh JOB $RUN $LOCAL $STALL  (STALL $STALL s, POLL ${POLL:-300} s)"; exit 0
fi
OUT=$($E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; cd $BASE; sbatch --parsable --qos=$QOS --time=$TL rbmppL_prod.sbatch $RUN $TS $ES")
JOB=${OUT%%;*}; [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "submit failed: $OUT"; exit 1; }
echo "$JOB" > $LOCAL/JOBID
# 3. durable watcher: setsid orphan re-parented to PID 1
DELTA=$DELTA POLL=${POLL:-300} setsid nohup bash $H/watch_rbmppL.sh $JOB $RUN $LOCAL $STALL > $LOCAL/watcher.out 2>&1 < /dev/null &
sleep 2; WPID=$(pgrep -f "watch_rbmppL.sh $JOB " | head -1); echo "watcher pid $WPID ppid $(ps -o ppid= -p $WPID | tr -d ' ')" | tee $LOCAL/WATCHER_PID
$E paderborn "export PATH=/opt/software/slurm/25.11.8-1/bin:\$PATH; squeue -j $JOB -o '%i %T %r'"
echo "submitted job $JOB run $RUN; the owner waits with:"
echo "  until grep -q 'VERDICT job $JOB' $DELTA; do sleep 120; done; grep 'VERDICT job $JOB' $DELTA"
