#!/bin/bash
#SBATCH --job-name=pf_r6tab
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --qos=cont
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# Round 6 (PLAN_pfaffian.md "Round 6 two-PP + parameter-matched control") = a COPY of r5_pdb_tab.sbatch with the round tag as an argument (t6: arm r6 -> theta_6.npy, tab_t6;
# t5: arm m5 -> theta_5.npy, tab_t5), code dir codeR6, theta sizes 5334 (pp2) and 5256 (rbmpp alpha 72; all 7 must agree).  (1) theta_R = mean of the 7 iterates of the fit
# (round_1/theta_step{600,750,900,1050,1200,1350}.npy + round_1/theta.npy = step 1500), shas in TABROOT/theta_shas.json; (2) la table tab_TAG.  Table-only path of cx_r1.py
# (spec max_steps 0 + warm_theta).  H100 table pass: single PP ~240 s, pp2 measured in round 5 (job 4099839).
# usage: r6_pdb_tab.sbatch TABROOT FITRUN TAG   (TAG t6 | t5; TABROOT/tab_TAG/spec_r1.json, its warm_theta = TABROOT/theta_${TAG#t}.npy)
set -uo pipefail
ROOT=$1; FITRUN=$2; TAG=$3; TH=$ROOT/theta_${TAG#t}.npy
case "$TAG" in t5|t6) ;; *) echo "bad TAG $TAG" >> "$ROOT/run_metadata.txt"; exit 2;; esac
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
export ST6_CSR=$B/data/csr ST6_TABLE=$B/data/psi0_6x6_table.npz ST6_SYM=$B/data/sym_tables.npz CX_LOOP8=/unused_by_r1
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
M=$ROOT/run_metadata.txt
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$M"
$PY - "$FITRUN/round_1" "$ROOT" "$TH" >> "$M" 2>&1 <<'PY'
import sys, os, json, hashlib, numpy as np
R, O, TH = sys.argv[1:4]; names = [f'theta_step{n}.npy' for n in (600, 750, 900, 1050, 1200, 1350)] + ['theta.npy']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
th = [np.load(f'{R}/{f}') for f in names]; assert th[0].shape in ((5334,), (5256,)) and all(t.shape == th[0].shape and t.dtype == np.float64 and np.isfinite(t).all() for t in th)
m = np.mean(np.stack(th), axis=0); np.save(TH, m); b = os.path.basename(TH)
d = {f: sha(f'{R}/{f}') for f in names}; d[b] = sha(TH); json.dump(d, open(f'{O}/theta_shas.json', 'w'), indent=1); print(b, d[b], th[0].size, flush=True)
PY
[ -s "$TH" ] || { echo "$TH not built" >> "$M"; exit 4; }
cd $B/codeR6
RUN=$ROOT/tab_$TAG; export CX_TABDIR=$RUN/tables; mkdir -p "$RUN"
echo "tab_$TAG start $(date -Is)" >> "$M"
$PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc=$?
echo "tab_${TAG}_exit=$rc finished=$(date -Is)" >> "$M"
if [ -s "$ROOT/tab_t0/spec_r1.json" ]; then                     # M5: la table of the INITIAL theta (spec = fit spec with max_steps 0) -> e_start
  RUN=$ROOT/tab_t0; export CX_TABDIR=$RUN/tables; echo "tab_t0 start $(date -Is)" >> "$M"
  $PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc0=$?; echo "tab_t0_exit=$rc0 finished=$(date -Is)" >> "$M"; [ $rc0 -gt $rc ] && rc=$rc0
fi
echo "tabs_exit=$rc finished=$(date -Is)" >> "$M"
exit $rc
