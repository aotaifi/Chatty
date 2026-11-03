#!/bin/bash
#SBATCH --job-name=pf_v
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=cont
#SBATCH --time=05:00:00
#SBATCH --no-requeue
# Gate V (PLAN_pfaffian.md "B design" / "V (narrowed)"), Paderborn H100: the round-5 two-PP fit (theta_4 -> T_5, s_G, seed 1, 1500 steps) driven by the SAMPLED POOL
# (pool_fit.py fit; pool = data/v/pool.npz, S = 4e5) instead of the exact table.  Code dir codeV (pool_fit.py + cx_lib.py + rbmpp_lib.py + st6_sector.py).
# Sizing: per step the same rows as round 5's pp2 arm (B 8208 x (1 + K 2), acceptance 8208) -> 5.80 s/step measured (pfaffian_r5/pp2_s1/fit/cx_r1.json, job 4099837)
# -> 2.4 h + start-up against 5 h wall.  No Lab is loaded (memory < round 5's).  On success writes RUN/cx_r1.json {"done": true} for lswo_pdb_watch.sh.
# usage: v_pdb.sbatch RUN      runs  pool_fit.py fit RUN/round_1 RUN/spec_pool_fit.json
set -uo pipefail
RUN=$1
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd $B/codeV
$PY pool_fit.py fit "$RUN/round_1" "$RUN/spec_pool_fit.json"; rc=$?
echo "pool_fit_exit=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
if [ $rc = 0 ]; then
  $PY - "$RUN" <<'PY'
import json, sys
R = sys.argv[1]; d = json.load(open(f'{R}/round_1/pool_fit.json')); f = d['fit']
ok = bool(d.get('done')) and f['steps'] == d['spec']['max_steps'] and not d.get('stop')
out = dict(driver='pool_fit.py', done=ok, fit=dict(steps=f['steps'], median_sec_per_step=f['median_sec_per_step'], acceptance_all=f['acceptance_all'], L_ho_step0=f['curve'][0]['L_ho'],
           L_ho_final=f['L_ho_final'], saved_steps=f['saved_steps'], theta_bar_sha256=f['theta_bar_sha256']))
if not ok: out['stop'] = 'pool_fit not done or wrong step count'
json.dump(out, open(f'{R}/cx_r1.json', 'w'), indent=1)
PY
fi
exit $rc
