#!/bin/bash
#SBATCH --job-name=s1
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=38G
#SBATCH --time=01:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_compress6/logs/%x-%j.out
# Test S1 (README compress_6x6 "Test S1"): the fit-free hop+FN chain with the Pfaffian's own sign, up to 15 rounds, one A40 slice.
# Usage: sbatch [--time=..] s1.sbatch RUNDIR   (RUNDIR holds code/ and spec_s1.json with code_sha256; staged by launch_s1.sh)
set -uo pipefail
RUN=$1
export XLA_FLAGS="${CX_XLA:---xla_gpu_autotune_level=0}" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${CX_MEMFRAC:-0.85} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/project/theorie/a/A.Otaifi/a40slice/venv_jax_0.8.2/bin/python
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd "$RUN/code"
$PY cx_s1.py "$RUN" "$RUN/spec_s1.json"; rc=$?
echo "exit=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
exit $rc
