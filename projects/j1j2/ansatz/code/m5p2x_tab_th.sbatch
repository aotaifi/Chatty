#!/bin/bash
#SBATCH --job-name=pf_m5p2e1tab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --no-requeue
# M5' RERUN, EXTENSION SEGMENT 1 table job (derived from m5p2_tab_th.sbatch, not edited).  usage: m5p2x_tab_th.sbatch TAB FIT CODEDIR   (TAB = pfaffian_m5p2/ext1/tab_th, FIT = pfaffian_m5p2/ext1/fit_th)
# (1) theta_avg = mean of 7 saved iterates, global steps 1200,1350,1500,1650,1800,1950,2100: round-1 theta_step1200, theta_step1350, theta.npy(=1500); ext1 theta_step150/300/450, theta.npy (=2100); shas + sources in TAB/theta_shas.json;
# (2) two la-table passes (A40 ~42 min each, measured 41.8 min in the 1500 run): tab_avg (tail average), tab_s1800 (ext1 theta_step300.npy).  theta(2100)'s table is the fit's own la_table.npy (bit-identical to a separate pass at 1500).
set -uo pipefail
TAB=$1; FITRUN=$2; CODE=$3; TH=$TAB/theta_avg.npy; RF=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_m5p2/round/fit_th/round_1
L=/project/theorie/a/A.Otaifi/Chatty/runs
export ST6_CSR=$L/stall6/csr ST6_TABLE=$L/stall6/data/psi0_6x6_table.npz ST6_SYM=$L/stall6/data/sym_tables.npz CX_LOOP8=/unused_by_r1
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=$L/a40slice/venv_jax_0.8.2/bin/python
export PYTHONPATH=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code
M=$TAB/run_metadata.txt
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$M"
cd "$CODE"
$PY -c "import jax; d=jax.devices(); print(d); assert d[0].platform in ('gpu','cuda'), d" >> "$M" 2>&1 || { echo "no CUDA device" >> "$M"; exit 3; }
$PY - "$RF" "$FITRUN/round_1" "$TAB" "$TH" >> "$M" 2>&1 <<'PY'
import sys, os, json, hashlib, numpy as np
RF, EF, O, TH = sys.argv[1:5]
src = [(1200, f'{RF}/theta_step1200.npy'), (1350, f'{RF}/theta_step1350.npy'), (1500, f'{RF}/theta.npy'), (1650, f'{EF}/theta_step150.npy'), (1800, f'{EF}/theta_step300.npy'), (1950, f'{EF}/theta_step450.npy'), (2100, f'{EF}/theta.npy')]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
th = [np.load(p) for _, p in src]; assert len(th) == 7 and all(t.shape == (5256,) and t.dtype == np.float64 and np.isfinite(t).all() for t in th)
m = np.mean(np.stack(th), axis=0); np.save(TH, m)
d = {f'global_{g}': dict(path=p, sha256=sha(p)) for g, p in src}; d['theta_avg.npy'] = sha(TH); json.dump(d, open(f'{O}/theta_shas.json', 'w'), indent=1); print('theta_avg.npy', d['theta_avg.npy'], m.size, flush=True)
PY
[ -s "$TH" ] || { echo "$TH not built" >> "$M"; exit 4; }
rc=0
for TAG in avg s1800; do
  RUN=$TAB/tab_$TAG; export CX_TABDIR=$RUN/tables; mkdir -p "$RUN"
  echo "tab_$TAG start $(date -Is)" >> "$M"
  $PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; r=$?
  echo "tab_${TAG}_exit=$r finished=$(date -Is)" >> "$M"; [ $r -gt $rc ] && rc=$r
done
echo "tabs_exit=$rc finished=$(date -Is)" >> "$M"
exit $rc
