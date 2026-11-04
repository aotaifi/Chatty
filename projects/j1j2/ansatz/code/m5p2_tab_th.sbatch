#!/bin/bash
#SBATCH --job-name=pf_m5p2tab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
# M5' RERUN table job (PLAN_pfaffian.md "M5' rerun"), derived from m5p_tab_th.sbatch (t5 only; m5p_tab_th.sbatch is not edited).  usage: m5p2_tab_th.sbatch TAB FIT CODEDIR
# (1) theta_5 = mean of the 7 iterates (FIT/round_1/theta_step{600,750,900,1050,1200,1350}.npy + theta.npy), shas in TAB/theta_shas.json; (2) three la-table passes, A40 ~40 min each (r4_tab_a40 header):
#   tab_t5 (tail average theta_5; the registered readouts (1),(2)), tab_s1200 (theta_step1200.npy), tab_s1500 (theta.npy = theta at step 1500)  -> the plateau data (E_FN of theta(1200), theta(1500), m5p2_plateau.py).
# Sizing: MaxRSS of the m5p tab job is unmeasured for 3 passes in one job but each pass is a separate python process; 64G as m5p_tab_th.sbatch (single PP + RBM 5256).  Wall set on the sbatch line: 3 passes x ~40 min = ~2 h, wall 04:00.
set -uo pipefail
TAB=$1; FITRUN=$2; CODE=$3; TH=$TAB/theta_5.npy
L=/project/theorie/a/A.Otaifi/Chatty/runs
export ST6_CSR=$L/stall6/csr ST6_TABLE=$L/stall6/data/psi0_6x6_table.npz ST6_SYM=$L/stall6/data/sym_tables.npz CX_LOOP8=/unused_by_r1
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=$L/a40slice/venv_jax_0.8.2/bin/python
export PYTHONPATH=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code
M=$TAB/run_metadata.txt
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$M"
cd "$CODE"
$PY -c "import jax; d=jax.devices(); print(d); assert d[0].platform in ('gpu','cuda'), d" >> "$M" 2>&1 || { echo "no CUDA device" >> "$M"; exit 3; }
$PY - "$FITRUN/round_1" "$TAB" "$TH" >> "$M" 2>&1 <<'PY'
import sys, os, json, hashlib, numpy as np
R, O, TH = sys.argv[1:4]; names = [f'theta_step{n}.npy' for n in (600, 750, 900, 1050, 1200, 1350)] + ['theta.npy']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
th = [np.load(f'{R}/{f}') for f in names]; assert th[0].shape == (5256,) and all(t.shape == th[0].shape and t.dtype == np.float64 and np.isfinite(t).all() for t in th)
m = np.mean(np.stack(th), axis=0); np.save(TH, m); b = os.path.basename(TH)
d = {f: sha(f'{R}/{f}') for f in names}; d[b] = sha(TH); json.dump(d, open(f'{O}/theta_shas.json', 'w'), indent=1); print(b, d[b], th[0].size, flush=True)
PY
[ -s "$TH" ] || { echo "$TH not built" >> "$M"; exit 4; }
rc=0
for TAG in t5 s1200 s1500; do
  RUN=$TAB/tab_$TAG; export CX_TABDIR=$RUN/tables; mkdir -p "$RUN"
  echo "tab_$TAG start $(date -Is)" >> "$M"
  $PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; r=$?
  echo "tab_${TAG}_exit=$r finished=$(date -Is)" >> "$M"; [ $r -gt $rc ] && rc=$r
done
echo "tabs_exit=$rc finished=$(date -Is)" >> "$M"
exit $rc
