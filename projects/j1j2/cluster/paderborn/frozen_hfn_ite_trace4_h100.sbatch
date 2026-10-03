#!/bin/bash
#SBATCH --job-name=chatty-ite-trace
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/ite-trace-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/ite-trace-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN=$(mktemp -d "${TMPDIR:-/tmp}/itetrace.XXXXXX")
cp /pc2/users/h/hpcalot/chatty/j1j2/src/frozen_hfn_ite_firstepoch_trace4.py "$RUN/"
cp /pc2/users/h/hpcalot/chatty/j1j2/src/closed_fn_krylov_exact4x4.py "$RUN/"
cd "$RUN"
python frozen_hfn_ite_firstepoch_trace4.py
