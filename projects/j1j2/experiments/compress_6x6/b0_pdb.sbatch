#!/bin/bash
#SBATCH --job-name=b0s2
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
# Test B0 seed 2 on one Paderborn H100 (Otus). Submitted by launch_b0_pdb.sh, which adds --time and --output (RUN/slurm-%j.out).
# (1) SR VMC training of PP + 3x3 backflow (rbmpp_bf_train.py: resumable, stops at spec wall_sec or the plateau rule), (2) cx_b0.py: control, table, E_FN.
# Same code as b0.sbatch/seed 1 (sha256-identical); paths via the cx_lib env overrides as s1_h100.sbatch (S1 never reads CX_LOOP8; neither does cx_b0).
set -uo pipefail
RUN=$1
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
export ST6_CSR=$B/data/csr ST6_TABLE=$B/data/psi0_6x6_table.npz ST6_SYM=$B/data/sym_tables.npz CX_LOOP8=/unused_by_b0 CX_TABDIR=$RUN/tables
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd "$RUN/code"
$PY rbmpp_bf_train.py "$RUN/train" "$RUN/spec_train.json"; rc=$?
echo "train_exit=$rc $(date -Is)" >> "$RUN/run_metadata.txt"
[ $rc -ne 0 ] && exit $rc
[ -f "$RUN/train/params_final.npz" ] || exit 5
$PY cx_b0.py "$RUN" "$RUN/spec_readout.json"; rc=$?
echo "readout_exit=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
exit $rc
