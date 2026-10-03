#!/bin/bash
#SBATCH --job-name=chatty-j1j2-learned-fn8-corr-a40
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/learned-fn8-corr-a40-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/learned-fn8-corr-a40-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
BASE=/pc2/users/h/hpcalot/chatty/j1j2
OUT="$BASE/results/learned_fn8_corr_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-learned-fn8-corr.XXXXXX")
finish(){
  cp -f "$RUN"/*.npz "$OUT/" 2>/dev/null || true
  cp -f "$RUN"/*.json "$OUT/" 2>/dev/null || true
  cp -f "$RUN"/*.py "$OUT/" 2>/dev/null || true
}
trap finish EXIT
cp "$BASE/src/callable_fn_loop.py" "$RUN/"
cp "$BASE/src/learned_fn8_callable_amplitude_adapter.py" "$RUN/"
cp "$BASE/inputs/fn_residual_cnn_8x8.mpack" "$RUN/"
cp "$BASE/inputs/learned_fn_handoff_corrected_8x8.npz" "$RUN/"
cp "$BASE/inputs/krylov_scaling_8x8_a1.20_tr4096_va2048.npz" "$RUN/"
cp "$BASE/inputs/vit_J2=0.50_N=8x8_k=0.mpack" "$RUN/"
cd "$RUN"
export BASE_VIT_CHECKPOINT="$RUN/vit_J2=0.50_N=8x8_k=0.mpack"
{
  echo "job=${SLURM_JOB_ID} host=$(hostname) started=$(date -Is)"
  nvidia-smi -L
  python - <<'PY'
import jax,flax,netket,numpy,scipy
print('jax',jax.__version__,'backend',jax.default_backend(),'devices',jax.devices())
print('flax',flax.__version__,'netket',netket.__version__,'numpy',numpy.__version__,'scipy',scipy.__version__)
PY
  sha256sum fn_residual_cnn_8x8.mpack learned_fn_handoff_corrected_8x8.npz vit_J2=0.50_N=8x8_k=0.mpack
} > "$OUT/run_metadata.txt"

for SEED in 12101 12102; do
  python callable_fn_loop.py \
    --backend learned_fn8_callable_amplitude_adapter.py \
    --checkpoint fn_residual_cnn_8x8.mpack \
    --L 8 --J2 0.5 \
    --threshold-npz learned_fn_handoff_corrected_8x8.npz \
    --threshold-states-key threshold_states --threshold-weights-key threshold_weights \
    --pool-npz learned_fn_handoff_corrected_8x8.npz \
    --pool-states-key pool_states --pool-weights-key pool_weights \
    --M 128 --beta-target 1.2 --burn-beta 0.4 --tau-max 0.025 \
    --seed "$SEED" --out "learned_fn8_corr_seed${SEED}.npz"
done
python - <<'PY'
import json,numpy as np
T0=-28.37107876288694
base=np.load('krylov_scaling_8x8_a1.20_tr4096_va2048.npz')
hand=np.load('learned_fn_handoff_corrected_8x8.npz')
z1=np.load('learned_fn8_corr_seed12101.npz'); z2=np.load('learned_fn8_corr_seed12102.npz')
T=float(z1['threshold']); rn=z1['threshold_r']; ro=hand['r0_threshold']
old=(ro<=T0); new=(rn<=T); flip=(old!=new)
wq=hand['threshold_weights'].astype(float); wq/=wq.sum()
eta=float(hand['eta']); gp=hand['train_g'].astype(float)
wp=base['iwtrain'].astype(float)*np.exp(2*eta*gp); wp/=wp.sum()
means=np.array([z1['Es'].mean(),z2['Es'].mean()])
tails=np.array([z1['Es'][-8:].mean(),z2['Es'][-8:].mean()])
summary={'T0':T0,'T_learned':T,'node_change_raw':float(flip.mean()),
 'node_change_qalpha':float(wq@flip),'node_change_physical':float(wp@flip),
 'Emeans':means.tolist(),'Emean':float(means.mean()),
 'between_replica_SE':float(abs(means[0]-means[1])/2),
 'tail8':tails.tolist(),'amp_eval':[int(z1['amp_eval']),int(z2['amp_eval'])]}
print('LEARNED_FN8_CORRECTED_SUMMARY',summary)
json.dump(summary,open('learned_fn8_corrected_summary.json','w'),indent=2)
PY
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
