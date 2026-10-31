#!/bin/bash
#SBATCH --job-name=pf_r3ctab
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
# R3-cont, Paderborn: la tables of the saved checkpoints theta_step{N} of one arm (afterok on that arm's fit job; name and --time set by the launcher).  Table-only path of cx_r1.py (spec max_steps 0 + warm_theta = checkpoint), ~190 s per table.
# usage: r3c_pdb_tab.sbatch TABROOT FITRUN "N1 N2 ..."    (TABROOT/tab_N/spec_r1.json; FITRUN/round_1/theta_step{N}.npy)
set -uo pipefail
ROOT=$1; FITRUN=$2; NS=$3
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
export ST6_CSR=$B/data/csr ST6_TABLE=$B/data/psi0_6x6_table.npz ST6_SYM=$B/data/sym_tables.npz CX_LOOP8=/unused_by_r1
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
M=$ROOT/run_metadata.txt
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$M"
cd $B/codeR3c
rcmax=0
for N in $NS; do
  RUN=$ROOT/tab_$N; export CX_TABDIR=$RUN/tables; mkdir -p "$RUN"
  [ -s "$FITRUN/round_1/theta_step$N.npy" ] || { echo "tab_$N: theta_step$N.npy missing" >> "$M"; rcmax=5; continue; }
  echo "tab_$N start $(date -Is)" >> "$M"
  $PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc=$?
  echo "tab_${N}_exit=$rc finished=$(date -Is)" >> "$M"; [ $rc -gt $rcmax ] && rcmax=$rc
done
echo "tabs_exit=$rcmax finished=$(date -Is)" >> "$M"
exit $rcmax
