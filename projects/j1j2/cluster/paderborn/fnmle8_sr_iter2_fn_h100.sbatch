#!/bin/bash
#SBATCH --job-name=chatty-fnmle-i2fn
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-i2fn-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-i2fn-%j.err
set -euo pipefail
: "${TRAIN_JOB:?TRAIN_JOB required}"
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2; H="$BASE/results/fnmle_sr_iter2_train_${TRAIN_JOB}"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/fnmle-i2fn.XXXXXX"); OUT="$BASE/results/fnmle_sr_iter2_fn_${SLURM_JOB_ID}"; mkdir -p "$OUT"
trap 'cp -f "$RUN"/*.npz "$RUN"/*.json "$OUT/" 2>/dev/null || true' EXIT
cp "$BASE/src/callable_fn_loop.py" "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$H/vit8_fnmle_sr_iter2.mpack" "$H/vit8_fnmle_sr_iter2_handoff.npz" "$RUN/"
cd "$RUN"
for SEED in 14001 14002; do
 python callable_fn_loop.py   --backend vit8_callable_amplitude_adapter.py --checkpoint vit8_fnmle_sr_iter2.mpack   --L 8 --J2 .5   --threshold-npz vit8_fnmle_sr_iter2_handoff.npz   --threshold-states-key threshold_states --threshold-weights-key threshold_weights   --pool-npz vit8_fnmle_sr_iter2_handoff.npz   --pool-states-key pool_states --pool-weights-key pool_weights   --M 128 --beta-target 1.2 --burn-beta .4 --tau-max .025   --seed "$SEED" --out "fnmle_sr_iter2_seed${SEED}.npz"
done
python - <<'PY'
import numpy as np,json
zs=[np.load(f"fnmle_sr_iter2_seed{s}.npz") for s in (14001,14002)]
m=np.array([z["Es"].mean() for z in zs]); t=np.array([z["Es"][-8:].mean() for z in zs])
o={"Emeans":m.tolist(),"Emean":float(m.mean()),"between_replica_SE":float(abs(m[0]-m[1])/2),
   "tail8":t.tolist(),"baseline0":-31.8854241096,"iter1_pending_compare":True}
print("ITER2_FN_SUMMARY",json.dumps(o),flush=True);json.dump(o,open("fnmle_sr_iter2_summary.json","w"),indent=2)
PY
