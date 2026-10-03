#!/bin/bash
#SBATCH --job-name=chatty-fnmle-k1fn
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-k1fn-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/fnmle-k1fn-%j.err
set -euo pipefail
: "${HANDOFF_JOB:?submit with HANDOFF_JOB}"
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/pc2/users/h/hpcalot/chatty/j1j2
H="$BASE/results/fnmle_sr_handoff_${HANDOFF_JOB}"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/fnmle-k1fn.XXXXXX")
OUT="$BASE/results/fnmle_sr_k1fn_${SLURM_JOB_ID}"; mkdir -p "$OUT"
finish(){ cp -f "$RUN"/*.npz "$RUN"/*.json "$OUT/" 2>/dev/null || true; }
trap finish EXIT
cp "$BASE/src/callable_fn_loop.py" "$BASE/src/vit8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$RUN/"
cp "$H/vit8_fnmle_sr_eta001.mpack" "$H/vit8_fnmle_sr_handoff.npz" "$RUN/"
cd "$RUN"
for SEED in 13001 13002; do
 python callable_fn_loop.py   --backend vit8_callable_amplitude_adapter.py   --checkpoint vit8_fnmle_sr_eta001.mpack   --L 8 --J2 0.5   --threshold-npz vit8_fnmle_sr_handoff.npz   --threshold-states-key threshold_states --threshold-weights-key threshold_weights   --pool-npz vit8_fnmle_sr_handoff.npz   --pool-states-key pool_states --pool-weights-key pool_weights   --M 128 --beta-target 1.2 --burn-beta 0.4 --tau-max 0.025   --seed "$SEED" --out "fnmle_sr_k1fn_seed${SEED}.npz"
done
python - <<'PY'
import json,numpy as np
T0=-28.37107876288694
h=np.load("vit8_fnmle_sr_handoff.npz")
base=np.load("krylov_scaling_8x8_a1.20_tr4096_va2048.npz")
zs=[np.load(f"fnmle_sr_k1fn_seed{s}.npz") for s in (13001,13002)]
T=float(zs[0]["threshold"]); rn=zs[0]["threshold_r"]; ro=h["r0_threshold"]
tw=h["threshold_weights"]; tw=tw/tw.sum()
flip=(ro<=T0)!=(rn<=T)
means=np.array([z["Es"].mean() for z in zs])
tails=np.array([z["Es"][-8:].mean() for z in zs])
summary={"T0":T0,"T1":T,
 "threshold_weighted_node_change":float(tw@flip),
 "threshold_raw_node_change":float(np.mean(flip)),
 "Emeans":means.tolist(),"Emean":float(means.mean()),
 "between_replica_SE":float(abs(means[0]-means[1])/2),
 "tail8":tails.tolist(),
 "baseline_Emean":-31.8854241096,"baseline_SE":0.00159161,
 "delta_vs_baseline":float(means.mean()+31.8854241096),
 "amp_eval":[int(z["amp_eval"]) for z in zs]}
print("SR_K1FN_SUMMARY",json.dumps(summary),flush=True)
json.dump(summary,open("fnmle_sr_k1fn_summary.json","w"),indent=2)
PY
