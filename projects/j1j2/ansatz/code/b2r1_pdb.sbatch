#!/bin/bash
#SBATCH --job-name=pf_b2r1
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --no-requeue
# Branch 2 R1 fit (PLAN_pfaffian.md "BRANCH 2 = OPTION (ii)", "Branch 2 GO" step 3), Paderborn H100: pool_fit.py arm 'pp8' (two-PP + complex RBM, 18466 params, warm from rb8_prod)
# on the labelled 8x8 pool (labels ln T = ln|B_0| + 4 f_J[B_0; sgn(M1)] mode 'fix' f0 1.22, partners on sgn(M1)'s blocked bonds, decades -2..-9, band-uniform weights),
# through b2_prep.py fit (= pool_fit.fit with the held-out weights at the pool's stored decade origin).  Code dir codeB2R1 (b2_prep.py pool_fit.py pp2_lemma.py cx_lib.py rbmpp_lib.py).
# Sizing (derived, NOT measured at 8x8 on an H100): 3-12 s/step x 1500 steps = 1.3-5 h; fit_sec in the spec stops at 1.5x the upper estimate (7.5 h) -> 7.5 h against 15 h wall.
# Memory: G = 3 B rows x 18466 x 8 B = 3.6 GB at B 8208 + an 18466^2 solve (2.7 GB); CPU 15.2 GB MaxRSS measured at the same rows (t8e2) -> 64G host.
# On success writes RUN/cx_r1.json {"done": true} for lswo_pdb_watch.sh.   usage: b2r1_pdb.sbatch RUN     runs  b2_prep.py fit RUN/round_1 RUN/spec_r1.json
set -uo pipefail
RUN=$1
B=/scratch/hpc-prf-ppdmrg/hpcalot/chatty_compress6
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd $B/codeB2R1
$PY b2_prep.py fit "$RUN/round_1" "$RUN/spec_r1.json"; rc=$?
echo "b2_prep_fit_exit=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
if [ $rc = 0 ]; then
  $PY - "$RUN" <<'PY'
import json, sys
R = sys.argv[1]; d = json.load(open(f'{R}/round_1/pool_fit.json')); f = d['fit']
ok = bool(d.get('done')) and f['steps'] == d['spec']['max_steps'] and not d.get('stop')
out = dict(driver='b2_prep.py fit (pool_fit.py arm pp8)', done=ok, fit=dict(steps=f['steps'], median_sec_per_step=f['median_sec_per_step'], acceptance_all=f['acceptance_all'],
           L_ho_step0=f['curve'][0]['L_ho'], L_ho_final=f['L_ho_final'], saved_steps=f['saved_steps'], theta_bar_sha256=f['theta_bar_sha256']))
if not ok: out['stop'] = 'pool_fit not done or wrong step count (fit_sec reached?)'
json.dump(out, open(f'{R}/cx_r1.json', 'w'), indent=1)
PY
fi
exit $rc
