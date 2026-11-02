#!/bin/bash
#SBATCH --job-name=pf_r5tab
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
# Round 5 (PLAN_pfaffian.md "Two-PP class (d)") = a COPY of r4_pdb_tab.sbatch (sha 4913bca5) with theta_4 -> theta_5, code dir codeR5, and the theta size taken from the iterates
# (2740 single PP, 5334 pp2; all 7 must agree).  (1) theta_5 = mean of the 7 iterates of the fit (round_1/theta_step{600,750,900,1050,1200,1350}.npy + round_1/theta.npy = step 1500),
# shas in TABROOT/theta_shas.json; (2) la table tab_t5 (theta_5 = TABROOT/theta_5.npy) only.  Table-only path of cx_r1.py (spec max_steps 0 + warm_theta), ~240 s (1 PP) / ~500 s (pp2, est.) on H100.
# usage: r5_pdb_tab.sbatch TABROOT FITRUN "t5"   (TABROOT/tab_t5/spec_r1.json; warm_theta of tab_t5 = TABROOT/theta_5.npy, of tab_N = FITRUN/round_1/theta_step{N}.npy)
set -uo pipefail
ROOT=$1; FITRUN=$2; NS=$3
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
export ST6_CSR=$B/data/csr ST6_TABLE=$B/data/psi0_6x6_table.npz ST6_SYM=$B/data/sym_tables.npz CX_LOOP8=/unused_by_r1
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
M=$ROOT/run_metadata.txt
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$M"
$PY - "$FITRUN/round_1" "$ROOT" >> "$M" 2>&1 <<'PY'
import sys, json, hashlib, numpy as np
R, O = sys.argv[1:3]; names = [f'theta_step{n}.npy' for n in (600, 750, 900, 1050, 1200, 1350)] + ['theta.npy']
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
th = [np.load(f'{R}/{f}') for f in names]; assert th[0].shape in ((2740,), (5334,)) and all(t.shape == th[0].shape and t.dtype == np.float64 and np.isfinite(t).all() for t in th)
m = np.mean(np.stack(th), axis=0); np.save(f'{O}/theta_5.npy', m)
d = {f: sha(f'{R}/{f}') for f in names}; d['theta_5.npy'] = sha(f'{O}/theta_5.npy'); json.dump(d, open(f'{O}/theta_shas.json', 'w'), indent=1); print('theta_5', d['theta_5.npy'], th[0].size, flush=True)
PY
[ -s "$ROOT/theta_5.npy" ] || { echo "theta_5.npy not built" >> "$M"; exit 4; }
cd $B/codeR5
rcmax=0
for N in $NS; do
  RUN=$ROOT/tab_$N; export CX_TABDIR=$RUN/tables; mkdir -p "$RUN"
  [ "$N" = t5 ] && TH=$ROOT/theta_5.npy || TH=$FITRUN/round_1/theta_step$N.npy
  [ -s "$TH" ] || { echo "tab_$N: $TH missing" >> "$M"; rcmax=5; continue; }
  echo "tab_$N start $(date -Is)" >> "$M"
  $PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc=$?
  echo "tab_${N}_exit=$rc finished=$(date -Is)" >> "$M"; [ $rc -gt $rcmax ] && rcmax=$rc
done
echo "tabs_exit=$rcmax finished=$(date -Is)" >> "$M"
exit $rcmax
