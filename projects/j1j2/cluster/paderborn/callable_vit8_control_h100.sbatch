#!/bin/bash
#SBATCH --job-name=chatty-vit8-callable-control
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/callable-vit8-control-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/callable-vit8-control-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUT="$BASE/results/callable_vit8_control_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/callable-vit8-control.XXXXXX")
finish(){ cp -f "$RUN"/*.npz "$OUT/" 2>/dev/null || true; cp -f "$RUN"/*.py "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$BASE/src/callable_fn_loop.py" "$RUN/"
cp "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/"
cp "$BASE/inputs/krylov_phys8_a2_fixedT.npz" "$RUN/"
cp "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$RUN/"
cd "$RUN"
for SEED in 10501 10502; do
 python callable_fn_loop.py    --backend vit8_callable_amplitude_adapter.py    --checkpoint vit_J2=0.50_N=8x8_k=0.mpack    --L 8 --J2 0.5    --threshold-npz krylov_scaling_8x8_a1.20_tr4096_va2048.npz    --threshold-states-key train_states --threshold-weights-key iwtrain    --pool-npz krylov_phys8_a2_fixedT.npz    --pool-states-key states --pool-weights-key ""    --M 128 --beta-target 1.4 --burn-beta 0.5 --tau-max 0.025    --seed "$SEED" --out "callable_vit8_control_seed${SEED}.npz"
done
python - <<'PY'
import numpy as np
for f in ['callable_vit8_control_seed10501.npz','callable_vit8_control_seed10502.npz']:
 z=np.load(f); E=z['Es']; print(f,'T',float(z['threshold']),'mean',float(E.mean()),'tail8',float(E[-8:].mean()),'amp_eval',int(z['amp_eval']))
PY
