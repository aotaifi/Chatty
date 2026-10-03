#!/bin/bash
#SBATCH --job-name=chatty-frozen-diag
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/frozen-diag-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/frozen-diag-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
RUN=$(mktemp -d "${TMPDIR:-/tmp}/frozen-diag.XXXXXX")
cp "$BASE/src/frozen_s1_a2_fn8.py" "$BASE/src/callable_fn_loop.py" "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_eta001.mpack" "$RUN/a1.mpack"
cp "$BASE/results/fnmle_sr_iter2_train_3506071/vit8_fnmle_sr_iter2.mpack" "$RUN/a2.mpack"
cp "$BASE/results/fnmle_sr_handoff_3506051/vit8_fnmle_sr_handoff.npz" "$RUN/h1.npz"
cd "$RUN"
python frozen_s1_a2_fn8.py --a1 a1.mpack --a2 a2.mpack --handoff1 h1.npz --seed 13001 --diag-n 128 --diag-only --out diag
