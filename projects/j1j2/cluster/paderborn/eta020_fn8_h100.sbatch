#!/bin/bash
#SBATCH --job-name=chatty-eta020-fn8-h100
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/eta020-fn8-h100-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/eta020-fn8-h100-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUT="$BASE/results/eta020_fn8_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-eta020-fn8.XXXXXX")
finish(){ cp -f "$RUN"/*.npz "$RUN"/*.json "$RUN"/*.py "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$BASE/src/callable_fn_loop.py" "$BASE/src/learned_fn8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/inputs/fn_residual_cnn_8x8.mpack" "$BASE/inputs/learned_fn_handoff_eta020_8x8.npz" "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/"
cd "$RUN"
export BASE_VIT_CHECKPOINT="$RUN/vit_J2=0.50_N=8x8_k=0.mpack"
export FN_RESIDUAL_ETA_OVERRIDE=0.2
python callable_fn_loop.py \
  --backend learned_fn8_callable_amplitude_adapter.py \
  --checkpoint fn_residual_cnn_8x8.mpack \
  --L 8 --J2 0.5 \
  --threshold-npz learned_fn_handoff_eta020_8x8.npz \
  --threshold-states-key threshold_states --threshold-weights-key threshold_weights \
  --pool-npz learned_fn_handoff_eta020_8x8.npz \
  --pool-states-key pool_states --pool-weights-key pool_weights \
  --M 128 --beta-target 1.2 --burn-beta 0.4 --tau-max 0.025 \
  --seed 12201 --out eta020_fn8_seed12201.npz
python - <<'PY'
import json,numpy as np
b=np.load("krylov_scaling_8x8_a1.20_tr4096_va2048.npz")
z=np.load("eta020_fn8_seed12201.npz")
h=np.load("learned_fn_handoff_eta020_8x8.npz")
T0=-28.37107876288694; T=float(z["threshold"])
old=b["rtrain"]<=T0; new=z["threshold_r"]<=T
w0=b["iwtrain"].astype(float); w0/=w0.sum()
w=h["threshold_weights"].astype(float); w/=w.sum()
flip=old!=new
E=z["Es"].astype(float)
s={"eta":.2,"T":T,"flip_frac":float(flip.mean()),"flip_mass_old":float(w0@flip),
   "flip_mass_new":float(w@flip),"Emean":float(E.mean()),"tail8":float(E[-8:].mean()),
   "nE":len(E),"amp_eval":int(z["amp_eval"])}
print("ETA020_SUMMARY",s)
json.dump(s,open("eta020_summary.json","w"),indent=2)
PY
