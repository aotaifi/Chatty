#!/bin/bash
#SBATCH --job-name=pf_r4tab
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --no-requeue
# Round 4, theorie A40 (afterok on pf_r4): (1) theta_4 = mean of the 7 iterates (fit/round_1/theta_step{600,...,1350}.npy + theta.npy), shas in ROOT/tabs/theta_shas.json; (2) la tables tab_t4 first, then the 6 optional checkpoints.
# Sizing: ~40 min per A40 table pass (lswo_a40_watch.sh header; H100 table 242 s in pfaffian_r3/pdb_tabs/run_metadata.txt) -> 7 tables ~4.7 h against 10 h wall.  Environment as r3_tab_a40.sbatch.
# usage: r4_tab_a40.sbatch ROOT   (ROOT/fit/code, ROOT/tabs/tab_{t4,600,...}/spec_r1.json)
set -uo pipefail
ROOT=$1; CODE=$ROOT/fit/code; FITRUN=$ROOT/fit
L=/project/theorie/a/A.Otaifi/Chatty/runs
export ST6_CSR=$L/stall6/csr ST6_TABLE=$L/stall6/data/psi0_6x6_table.npz ST6_SYM=$L/stall6/data/sym_tables.npz CX_LOOP8=/unused_by_r1
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=$L/a40slice/venv_jax_0.8.2/bin/python
export PYTHONPATH=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code
M=$ROOT/tabs/run_metadata.txt
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$M"
cd "$CODE"
$PY -c "import jax; d=jax.devices(); print(d); assert d[0].platform in ('gpu','cuda'), d" >> "$M" 2>&1 || { echo "no CUDA device" >> "$M"; exit 3; }
$PY - "$FITRUN/round_1" "$ROOT/tabs" >> "$M" 2>&1 <<'PY'
import sys, json, hashlib, numpy as np
R, O = sys.argv[1:3]; names = [f'theta_step{n}.npy' for n in (600, 750, 900, 1050, 1200, 1350)] + ['theta.npy']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
th = [np.load(f'{R}/{f}') for f in names]; assert all(t.shape == (2740,) and t.dtype == np.float64 and np.isfinite(t).all() for t in th)
m = np.mean(np.stack(th), axis=0); np.save(f'{O}/theta_4.npy', m)
d = {f: sha(f'{R}/{f}') for f in names}; d['theta_4.npy'] = sha(f'{O}/theta_4.npy'); json.dump(d, open(f'{O}/theta_shas.json', 'w'), indent=1); print('theta_4', d['theta_4.npy'], flush=True)
PY
[ -s "$ROOT/tabs/theta_4.npy" ] || { echo "theta_4.npy not built" >> "$M"; exit 4; }
rcmax=0
for N in t4 600 750 900 1050 1200 1350; do
  RUN=$ROOT/tabs/tab_$N; export CX_TABDIR=$RUN/tables; mkdir -p "$RUN"
  [ "$N" = t4 ] && TH=$ROOT/tabs/theta_4.npy || TH=$FITRUN/round_1/theta_step$N.npy
  [ -s "$TH" ] || { echo "tab_$N: $TH missing" >> "$M"; rcmax=5; continue; }
  echo "tab_$N start $(date -Is)" >> "$M"
  $PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc=$?
  echo "tab_${N}_exit=$rc finished=$(date -Is)" >> "$M"; [ $rc -gt $rcmax ] && rcmax=$rc
done
echo "tabs_exit=$rcmax finished=$(date -Is)" >> "$M"
exit $rcmax
