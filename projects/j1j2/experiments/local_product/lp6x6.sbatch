#!/bin/bash
#SBATCH --job-name=lp6x6
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/a/A.Otaifi/local_product/stageA/6x6/slurm-%j.out
# LP Stage A at 6x6 (experiments/local_product/PREREG.md + Amendment 1). Exact sector, A40 24 GB slice, jax 0.8.2 venv.
set -euo pipefail
OUT=/scratch/a/A.Otaifi/local_product/stageA/6x6; mkdir -p $OUT/code
cp /project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/local_product/lp6x6.py $OUT/code/
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.88 OMP_NUM_THREADS=4
{ echo "job_id=$SLURM_JOB_ID"; echo "host=$(hostname)"; echo "started=$(date -Is)"; sha256sum $OUT/code/lp6x6.py; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > $OUT/run_metadata.txt
/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python $OUT/code/lp6x6.py $OUT
echo "finished=$(date -Is) exit=$?" >> $OUT/run_metadata.txt
