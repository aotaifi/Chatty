#!/bin/bash
#SBATCH --job-name=f1
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
# Test F1 on one Paderborn H100 (Otus).  Submitted by launch_f1.sh, which adds --job-name, --time and --output (RUN/slurm-%j.out).
# usage: f1.sbatch RUN   (RUN/spec_f1.json; code from $B/codeF1, sha256-pinned in the spec; cx_f1.py writes RUN/cx_f1.json, RUN/progress.txt, RUN/round_1/)
# Paths via the cx_lib env overrides as b0_pdb.sbatch / s1_h100.sbatch (cx_f1 never reads CX_LOOP8).
set -uo pipefail
RUN=$1
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
export ST6_CSR=$B/data/csr ST6_TABLE=$B/data/psi0_6x6_table.npz ST6_SYM=$B/data/sym_tables.npz CX_LOOP8=/unused_by_f1 CX_TABDIR=$RUN/tables
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd $B/codeF1
$PY cx_f1.py "$RUN" "$RUN/spec_f1.json"; rc=$?
echo "f1_exit=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
exit $rc
