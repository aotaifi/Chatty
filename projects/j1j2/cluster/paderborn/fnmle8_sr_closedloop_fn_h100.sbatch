#!/bin/bash
#SBATCH --job-name=chatty-sr-closed-fn
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/sr-closed-fn-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/sr-closed-fn-%j.err
set -euo pipefail
: "${PARENT_JOB:?need PARENT_JOB}"
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
PARENT="$BASE/results/sr_prod_build_${PARENT_JOB}"
OUT="$BASE/results/sr_closed_fn_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/sr-closed-fn.XXXXXX")
finish(){
  cp -f "$RUN"/*.npz "$RUN"/*.json "$RUN"/*.py "$OUT/" 2>/dev/null || true
}
trap finish EXIT
cp "$BASE/src/callable_fn_loop.py" "$RUN/"
cp "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$PARENT/vit8_fnmle_sr_eta001.mpack" "$RUN/"
cp "$PARENT/vit8_fnmle_sr_eta001_handoff.npz" "$RUN/"
cp "$PARENT/vit8_fnmle_sr_eta001_node.json" "$RUN/"
cd "$RUN"
for SEED in 10501 10502; do
 python callable_fn_loop.py    --backend vit8_callable_amplitude_adapter.py    --checkpoint vit8_fnmle_sr_eta001.mpack    --L 8 --J2 .5    --threshold-npz vit8_fnmle_sr_eta001_handoff.npz    --threshold-states-key threshold_states    --threshold-weights-key threshold_weights    --pool-npz vit8_fnmle_sr_eta001_handoff.npz    --pool-states-key pool_states    --pool-weights-key pool_weights    --M 128 --beta-target 1.2 --burn-beta .4 --tau-max .025    --seed "$SEED" --out "sr_closed_fn_seed${SEED}.npz"
done
python - <<'PY'
import json,numpy as np
B=-31.8854241096; BSE=.00159161
zs=[np.load(f"sr_closed_fn_seed{s}.npz") for s in (10501,10502)]
means=np.array([float(z["Es"].mean()) for z in zs])
tails=np.array([float(z["Es"][-8:].mean()) for z in zs])
m=float(means.mean()); se=float(abs(means[0]-means[1])/2)
out={"baseline":B,"baseline_se":BSE,
     "E_replicas":means.tolist(),"Emean":m,
     "between_replica_SE":se,"delta_vs_baseline":m-B,
     "tail8":tails.tolist(),
     "thresholds":[float(z["threshold"]) for z in zs],
     "amp_eval":[int(z["amp_eval"]) for z in zs]}
json.dump(out,open("sr_closed_fn_summary.json","w"),indent=2)
print("SR_CLOSED_FN_SUMMARY",json.dumps(out),flush=True)
PY
