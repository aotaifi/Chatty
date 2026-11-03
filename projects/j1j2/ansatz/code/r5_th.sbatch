#!/bin/bash
#SBATCH --job-name=pf_r5th
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
# Round 5 theorie race copy of one arm (copy of r3_a40.sbatch; partition, gres, --time, job name set by launch_r5_th.sh on the sbatch line).  Code: RUN/../../code_th (sha-pinned copy).
# Memory: O2b A40 MaxRSS 17.6 GB (single PP); 64G leaves room for the pp2 Jacobian (2x columns).  usage: r5_th.sbatch RUN CODEDIR   runs  CODEDIR/cx_r1.py RUN RUN/spec_r1.json
set -uo pipefail
RUN=$1; CODE=$2
L=/project/theorie/a/A.Otaifi/Chatty/runs
export ST6_CSR=$L/stall6/csr ST6_TABLE=$L/stall6/data/psi0_6x6_table.npz ST6_SYM=$L/stall6/data/sym_tables.npz CX_LOOP8=/unused_by_r1 CX_TABDIR=$RUN/tables
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=$L/a40slice/venv_jax_0.8.2/bin/python
export PYTHONPATH=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd "$CODE"
$PY -c "import jax; d=jax.devices(); print(d); assert d[0].platform in ('gpu','cuda'), d" >> "$RUN/run_metadata.txt" 2>&1 || { echo "no CUDA device" >> "$RUN/run_metadata.txt"; exit 3; }
$PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc=$?
echo "r1_exit=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
exit $rc
