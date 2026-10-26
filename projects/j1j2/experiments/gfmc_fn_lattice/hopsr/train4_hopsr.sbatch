#!/bin/bash
#SBATCH --job-name=hopsr_train4
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/hopsr/runs/train4-%j.out
# Deliverable 2: 4x4 hop-aware SR from the E_v optimum (RBM-only class, w30), runs A (lr 0.05->0.005) and B (lr 0.1->0.01), 1000 steps,
# and control C (objective E0, same machinery, 300 steps), in parallel on one H100; each stops at maxsec 5400 s.  The exact E_FN(a*h_a) of
# the checkpoints is computed afterwards on ws1 (l8_loop4x4.py evalfn).
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot; PY=$B/venv_jax_0.8.2/bin/python; C=$B/hopsr/code/gfmc_fn_lattice/hopsr
RUN=$B/hopsr/runs/train4_$SLURM_JOB_ID; mkdir -p $RUN
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 OMP_NUM_THREADS=2
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; sha256sum $C/*.py $C/../../rbmpp_6x6/rbmpp_lib.py $C/specs/train4_*.json $B/hopsr/data/init4_rbmonly_opt.npz; } > $RUN/run_metadata.txt
P() { echo "$(date -Is) PROGRESS $*" >> $RUN/progress.txt; }
PIDS=()
for s in A B C; do
  ( /usr/bin/time -v $PY $C/hopsr_train.py $RUN/$s $C/specs/train4_$s.json > $RUN/$s.log 2> $RUN/$s.time; rc=$?
    P "run $s exit $rc MaxRSS_kB $(grep 'Maximum resident' $RUN/$s.time | awk '{print $NF}'); $(grep TRAIN_DONE $RUN/$s/progress.txt | cut -c21-)" ) &
  PIDS+=($!)
done
# heartbeat for the watcher (the runs log per step in their own progress.txt)
( while true; do sleep 300; P "heartbeat $(for s in A B C; do echo -n "$s:$(grep -c 'PROGRESS it' $RUN/$s/progress.txt 2>/dev/null) "; done)"; done ) & HB=$!
wait "${PIDS[@]}"; kill $HB
echo finished=$(date -Is) >> $RUN/run_metadata.txt
P "TRAIN4_DONE"
