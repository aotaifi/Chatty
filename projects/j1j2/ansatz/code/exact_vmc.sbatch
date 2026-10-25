#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --no-requeue
# exact_vmc on one GPU (A40 slice, jax 0.8.2 venv; A40_SLICES_HOWTO.md).
# usage (via launch_exact_vmc.sh): sbatch --job-name=NAME --time=T --output=RUN/slurm-%j.out exact_vmc.sbatch RUN MODE [SPEC]
set -uo pipefail
RUN=$1; MODE=$2; SPEC=${3:-}
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} PYTHONUNBUFFERED=1
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "mode=$MODE spec=$SPEC"; echo "started=$(date -Is)";
  sha256sum "$RUN/code/exact_vmc.py"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python "$RUN/code/exact_vmc.py" "$MODE" "$RUN" $SPEC
rc=$?
echo "finished=$(date -Is) rc=$rc" >> "$RUN/run_metadata.txt"
exit $rc
