#!/bin/bash
# launch one exact_vmc job: launch_exact_vmc.sh NAME MODE WALL [SPEC.json] [extra sbatch args...]
# copies and pins the code into the run dir, submits, arms the durable watcher (setsid), prints job id and run dir.
set -euo pipefail
NAME=$1; MODE=$2; WALL=$3; SPEC=${4:-}; shift $(( $# >= 4 ? 4 : 3 ))
HERE=$(cd "$(dirname "$0")" && pwd); BASE=/scratch/a/A.Otaifi/ansatz_runs/exact_vmc_4x6
DELTA=/home/a/A.Otaifi/Chatty-organize/projects/j1j2/delta_today.md
RUN=$BASE/$NAME; mkdir -p "$RUN/code"
cp "$HERE/exact_vmc.py" "$HERE/exact_vmc.sbatch" "$RUN/code/"
cmp -s "$HERE/exact_vmc.py" "$RUN/code/exact_vmc.py"
sha256sum "$RUN/code/exact_vmc.py" > "$RUN/code/SHA256.txt"
ARGS=("$RUN" "$MODE"); if [ -n "$SPEC" ]; then cp "$SPEC" "$RUN/spec.json"; ARGS+=("$RUN/spec.json"); fi
[ -f "$RUN/ANALYSIS_BRIEF.md" ] || { echo "write $RUN/ANALYSIS_BRIEF.md first"; exit 1; }
# psi0 cache shared between jobs
[ -f "$BASE/psi0_4x6_J20.5.npy" ] && ln -sf "$BASE/psi0_4x6_J20.5.npy" "$RUN/" && ln -sf "$BASE/psi0_4x6_J20.5.npy.E" "$RUN/" || true
OUT=$(sbatch --parsable --job-name="$NAME" --time="$WALL" --output="$RUN/slurm-%j.out" --error="$RUN/slurm-%j.err" "$@" "$RUN/code/exact_vmc.sbatch" "${ARGS[@]}")
JOB=${OUT%%;*}; echo "$JOB" >> "$RUN/JOBID"
LOG=$RUN/exact_vmc_$MODE.log
STALL=${STALL:-1800} POLL=${POLL:-120} setsid nohup "$BASE/watch.sh" "$JOB" "$RUN" "$DELTA" "$LOG" "VERDICT exact_vmc" > "$RUN/watch_$JOB.out" 2>&1 < /dev/null &
sleep 2
WPID=$(pgrep -f "watch.sh $JOB " | head -1)
echo "job=$JOB run=$RUN watcher_pid=$WPID watcher_ppid=$(ps -o ppid= -p "$WPID" | tr -d ' ')"
sacct -j "$JOB" -X -n -o JobID,State,Partition,Reason%25
