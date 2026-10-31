#!/bin/bash
#SBATCH --job-name=r2ga40
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=13:00:00
#SBATCH --no-requeue
# R2G-conv (PLAN_pfaffian.md "R2G-conv"), copy of lswo_a40.sbatch (PYTHONPATH fix included; one arm per job); L-SWO on one full theorie A40 = r1_a40.sbatch = r1.sbatch (Paderborn) with local paths; only the spec's fit_sec differs (43200).
# O2b (job 17431886, full A40, cx.sbatch, 48G): MaxRSS 17.6 GB.  usage: r1_a40.sbatch RUN   runs  RUN/code/cx_r1.py RUN RUN/spec_r1.json
set -uo pipefail
RUN=$1
L=/project/theorie/a/A.Otaifi/Chatty/runs
export ST6_CSR=$L/stall6/csr ST6_TABLE=$L/stall6/data/psi0_6x6_table.npz ST6_SYM=$L/stall6/data/sym_tables.npz CX_LOOP8=/unused_by_r1 CX_TABDIR=$RUN/tables
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=$L/a40slice/venv_jax_0.8.2/bin/python
export PYTHONPATH=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code   # st6_sector (cx_lib.py l.19); missing in 17601653 (failed in 3 s)
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd "$RUN/code"
$PY -c "import jax; d=jax.devices(); print(d); assert d[0].platform in ('gpu','cuda'), d" >> "$RUN/run_metadata.txt" 2>&1 || { echo "no CUDA device" >> "$RUN/run_metadata.txt"; exit 3; }
$PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc=$?
echo "r1_exit=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
exit $rc
