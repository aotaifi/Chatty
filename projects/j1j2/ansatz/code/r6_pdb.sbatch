#!/bin/bash
#SBATCH --job-name=pf_r6
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=cont
#SBATCH --time=05:00:00
#SBATCH --no-requeue
# Round 6 (PLAN_pfaffian.md "Round 6 two-PP + parameter-matched control"), Paderborn H100: one arm (r6, m5) per job; name and --time set by launch_r6_pdb.sh on the sbatch line; code dir codeR6 (cx_r1fit.py f1374bed: + rbm_init_sigma). Copy of r5_pdb.sbatch with codeR5 -> codeR6.
# usage: r6_pdb.sbatch RUN      runs  cx_r1.py RUN RUN/spec_r1.json
set -uo pipefail
RUN=$1
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
export ST6_CSR=$B/data/csr ST6_TABLE=$B/data/psi0_6x6_table.npz ST6_SYM=$B/data/sym_tables.npz CX_LOOP8=/unused_by_r1 CX_TABDIR=$RUN/tables
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd $B/codeR6
$PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc=$?
echo "r1_exit=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
exit $rc
