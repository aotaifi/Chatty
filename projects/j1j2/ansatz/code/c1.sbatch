#!/bin/bash
#SBATCH --job-name=c1
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
# Test C (complex vs real PP) on one Paderborn H100, as f1.sbatch.  The launcher adds --job-name, --time, --output.
# usage: c1.sbatch RUN "SUB1:FLAGS1" ["SUB2:FLAGS2" ...]   each item runs  cx_c1.py RUN/SUB RUN/SUB/spec_c1.json FLAGS  (SUB "." = RUN itself), sequentially
#   FLAGS use '~' for spaces, e.g. "cplx:--complex~--im_sigma~0.01~--seed~1"
set -uo pipefail
RUN=$1; shift
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
export ST6_CSR=$B/data/csr ST6_TABLE=$B/data/psi0_6x6_table.npz ST6_SYM=$B/data/sym_tables.npz CX_LOOP8=/unused_by_c1 CX_TABDIR=$RUN/tables
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd $B/codeC1
rcall=0
for item in "$@"; do
  sub=${item%%:*}; flags=${item#*:}; flags=${flags//\~/ }
  [ "$sub" = . ] && D=$RUN || D=$RUN/$sub
  $PY cx_c1.py "$D" "$D/spec_c1.json" $flags; rc=$?
  echo "c1_exit[$sub]=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
  [ $rc -ne 0 ] && rcall=$rc
done
exit $rcall
