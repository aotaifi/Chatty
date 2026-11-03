#!/bin/bash
#SBATCH --job-name=pf_r5tth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
# Round 5 theorie race copy, table job of one arm (afterok on its fit; copy of r4_tab_a40.sbatch with the mean step of r5_pdb_tab.sbatch): (1) theta_5 = mean of the 7 iterates
# (FIT/round_1/theta_step{600,...,1350}.npy + theta.npy), shas in TAB/theta_shas.json; (2) tab_t5 only.  ~40 min per A40 table pass for one PP (r4_tab_a40 header), ~2x for pp2.
# usage: r5_tab_th.sbatch TAB FIT CODEDIR
set -uo pipefail
TAB=$1; FITRUN=$2; CODE=$3
L=/project/theorie/a/A.Otaifi/Chatty/runs
export ST6_CSR=$L/stall6/csr ST6_TABLE=$L/stall6/data/psi0_6x6_table.npz ST6_SYM=$L/stall6/data/sym_tables.npz CX_LOOP8=/unused_by_r1
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=$L/a40slice/venv_jax_0.8.2/bin/python
export PYTHONPATH=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code
M=$TAB/run_metadata.txt
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$M"
cd "$CODE"
$PY -c "import jax; d=jax.devices(); print(d); assert d[0].platform in ('gpu','cuda'), d" >> "$M" 2>&1 || { echo "no CUDA device" >> "$M"; exit 3; }
$PY - "$FITRUN/round_1" "$TAB" >> "$M" 2>&1 <<'PY'
import sys, json, hashlib, numpy as np
R, O = sys.argv[1:3]; names = [f'theta_step{n}.npy' for n in (600, 750, 900, 1050, 1200, 1350)] + ['theta.npy']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
th = [np.load(f'{R}/{f}') for f in names]; assert th[0].shape in ((2740,), (5334,)) and all(t.shape == th[0].shape and t.dtype == np.float64 and np.isfinite(t).all() for t in th)
m = np.mean(np.stack(th), axis=0); np.save(f'{O}/theta_5.npy', m)
d = {f: sha(f'{R}/{f}') for f in names}; d['theta_5.npy'] = sha(f'{O}/theta_5.npy'); json.dump(d, open(f'{O}/theta_shas.json', 'w'), indent=1); print('theta_5', d['theta_5.npy'], th[0].size, flush=True)
PY
[ -s "$TAB/theta_5.npy" ] || { echo "theta_5.npy not built" >> "$M"; exit 4; }
RUN=$TAB/tab_t5; export CX_TABDIR=$RUN/tables; mkdir -p "$RUN"
echo "tab_t5 start $(date -Is)" >> "$M"
$PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc=$?
echo "tab_t5_exit=$rc finished=$(date -Is)" >> "$M"
echo "tabs_exit=$rc finished=$(date -Is)" >> "$M"
exit $rc
