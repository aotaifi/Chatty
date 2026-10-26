#!/bin/bash
#SBATCH --job-name=hopsr_pilot
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --qos=cont
#SBATCH --output=/scratch/hpc-prf-ppdmrg/hpcalot/hopsr/runs/pilot-%j.out
# Hop-aware SR pilots (cost per step, E_1 noise): t4 (4x4 RBM-only, 5 steps, 16k samples), p8 (8x8 rb8_prod, 3 steps, 8192 chains),
# p6 (6x6 rpT4_sr2, full projection, 3 steps, 4096 chains).  Each stage has maxsec 600-900 s; MaxRSS per stage from /usr/bin/time -v.
set -uo pipefail
B=/scratch/hpc-prf-ppdmrg/hpcalot; PY=$B/venv_jax_0.8.2/bin/python; C=$B/hopsr/code/gfmc_fn_lattice/hopsr
RUN=$B/hopsr/runs/pilot_$SLURM_JOB_ID; mkdir -p $RUN
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=8
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; sha256sum $C/*.py $C/../vmc_moments.py $C/../../rbmpp_6x6/rbmpp_lib.py $C/specs/pilot_*.json; } > $RUN/run_metadata.txt
P() { echo "$(date -Is) PROGRESS $*" >> $RUN/progress.txt; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > $RUN/gpumem.log 2>&1 &
for s in t4 p8 p6; do
  P "stage $s start"
  /usr/bin/time -v $PY $C/hopsr_train.py $RUN/$s $C/specs/pilot_$s.json > $RUN/$s.log 2> $RUN/$s.time; rc=$?
  rss=$(grep "Maximum resident" $RUN/$s.time | awk '{print $NF}')
  P "stage $s exit $rc MaxRSS_kB $rss gpu_mem_peak_MiB $(sort -n $RUN/gpumem.log | tail -1); $(grep TRAIN_DONE $RUN/$s/progress.txt | cut -c21-)"
  : > $RUN/gpumem.log
done
echo finished=$(date -Is) >> $RUN/run_metadata.txt
P "PILOT_DONE"
