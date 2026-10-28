#!/bin/bash
#SBATCH --job-name=g1
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
# Test G (PLAN.md "TEST G") on one Paderborn H100, as c1.sbatch.  The launcher adds --job-name, --time, --output.
# usage: g1.sbatch RUN MODE      runs  cx_g1fit.py RUN RUN/spec_g1.json --mode MODE   (MODE probe|prod)
set -uo pipefail
RUN=$1; MODE=$2
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
export ST6_CSR=$B/data/csr ST6_TABLE=$B/data/psi0_6x6_table.npz ST6_SYM=$B/data/sym_tables.npz CX_LOOP8=/unused_by_g1 CX_TABDIR=$RUN/tables
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd $B/codeG1
$PY cx_g1fit.py "$RUN" "$RUN/spec_g1.json" --mode "$MODE"; rc=$?
echo "g1_exit=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
exit $rc
