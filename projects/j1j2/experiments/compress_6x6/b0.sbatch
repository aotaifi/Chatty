#!/bin/bash
#SBATCH --job-name=b0
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/logs/%x-%j.out
# Test B0 (README compress_6x6): one job on a full A40: (1) SR VMC training of PP + 3x3 backflow (rbmpp_bf_train.py, resumable,
# stops itself at spec wall_sec / the plateau rule), (2) the exact readout (cx_b0.py: control, table, E_FN).  Usage: sbatch b0.sbatch RUNDIR
# RUNDIR holds spec_train.json, spec_readout.json (with code_sha256) and code/ (copied by launch_b0.sh).
set -uo pipefail
RUN=$1
export XLA_FLAGS="${CX_XLA:---xla_gpu_autotune_level=0}" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${CX_MEMFRAC:-0.92} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd "$RUN/code"
$PY rbmpp_bf_train.py "$RUN/train" "$RUN/spec_train.json"; rc=$?
echo "train_exit=$rc $(date -Is)" >> "$RUN/run_metadata.txt"
[ $rc -ne 0 ] && exit $rc
[ -f "$RUN/train/params_final.npz" ] || exit 5
$PY cx_b0.py "$RUN" "$RUN/spec_readout.json"; rc=$?
echo "readout_exit=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
exit $rc
