#!/bin/bash
#SBATCH --job-name=pf_r6tth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
# Round 6 theorie race copy, table job of one arm (afterok on its fit) = a COPY of r5_tab_th.sbatch with the round tag as an argument (t6: arm r6 -> theta_6.npy, tab_t6;
# t5: arm m5 -> theta_5.npy, tab_t5) and theta sizes 5334 (pp2) / 5256 (rbmpp alpha 72).  (1) theta_R = mean of the 7 iterates (FIT/round_1/theta_step{600,...,1350}.npy + theta.npy),
# shas in TAB/theta_shas.json; (2) tab_TAG only.  ~40 min per A40 table pass for one PP (r4_tab_a40 header), ~2x for pp2.
# usage: r6_tab_th.sbatch TAB FIT CODEDIR TAG
set -uo pipefail
TAB=$1; FITRUN=$2; CODE=$3; TAG=$4; TH=$TAB/theta_${TAG#t}.npy
case "$TAG" in t5|t6) ;; *) echo "bad TAG $TAG" >> "$TAB/run_metadata.txt"; exit 2;; esac
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
th = [np.load(f'{R}/{f}') for f in names]; assert th[0].shape in ((5334,), (5256,)) and all(t.shape == th[0].shape and t.dtype == np.float64 and np.isfinite(t).all() for t in th)
m = np.mean(np.stack(th), axis=0); np.save(TH, m); b = os.path.basename(TH)
d = {f: sha(f'{R}/{f}') for f in names}; d[b] = sha(TH); json.dump(d, open(f'{O}/theta_shas.json', 'w'), indent=1); print(b, d[b], th[0].size, flush=True)
PY
[ -s "$TH" ] || { echo "$TH not built" >> "$M"; exit 4; }
RUN=$TAB/tab_$TAG; export CX_TABDIR=$RUN/tables; mkdir -p "$RUN"
echo "tab_$TAG start $(date -Is)" >> "$M"
$PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc=$?
echo "tab_${TAG}_exit=$rc finished=$(date -Is)" >> "$M"
if [ -s "$TAB/tab_t0/spec_r1.json" ]; then                     # M5: la table of the INITIAL theta (spec = fit spec with max_steps 0) -> e_start
  RUN=$TAB/tab_t0; export CX_TABDIR=$RUN/tables; echo "tab_t0 start $(date -Is)" >> "$M"
  $PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc0=$?; echo "tab_t0_exit=$rc0 finished=$(date -Is)" >> "$M"; [ $rc0 -gt $rc ] && rc=$rc0
fi
echo "tabs_exit=$rc finished=$(date -Is)" >> "$M"
exit $rc
