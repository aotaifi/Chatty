#!/bin/bash
#SBATCH --job-name=b2r1a40
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
# Branch 2 R1 fit, theorie A40 race copy of b2r1_pdb.sbatch (Paderborn H100), same spec except local paths and fit_sec 86400 (an A40 is expected ~3x slower than the H100:
# 1.3-5 H100-h -> ~4-15 A40-h, derived; fit_sec 24 h against 48 h wall).  Memory: CPU MaxRSS 15.2 GB at the same rows (t8e2) -> 48G.  The race (race_watch_eta.sh, kind b2r1)
# keeps the copy with the earlier expected finish.  usage: b2r1_a40.sbatch RUN    runs  RUN/code/b2_prep.py fit RUN/round_1 RUN/spec_r1.json
set -uo pipefail
RUN=$1
L=/project/theorie/a/A.Otaifi/Chatty/runs
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=$L/a40slice/venv_jax_0.8.2/bin/python
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$RUN/run_metadata.txt"
cd "$RUN/code"
$PY -c "import jax; d=jax.devices(); print(d); assert d[0].platform in ('gpu','cuda'), d" >> "$RUN/run_metadata.txt" 2>&1 || { echo "no CUDA device" >> "$RUN/run_metadata.txt"; exit 3; }
$PY b2_prep.py fit "$RUN/round_1" "$RUN/spec_r1.json"; rc=$?
echo "b2_prep_fit_exit=$rc finished=$(date -Is)" >> "$RUN/run_metadata.txt"
if [ $rc = 0 ]; then
  $PY - "$RUN" <<'PY'
import json, sys
R = sys.argv[1]; d = json.load(open(f'{R}/round_1/pool_fit.json')); f = d['fit']
ok = bool(d.get('done')) and f['steps'] == d['spec']['max_steps'] and not d.get('stop')
out = dict(driver='b2_prep.py fit (pool_fit.py arm pp8), A40 copy', done=ok, fit=dict(steps=f['steps'], median_sec_per_step=f['median_sec_per_step'], acceptance_all=f['acceptance_all'],
           L_ho_step0=f['curve'][0]['L_ho'], L_ho_final=f['L_ho_final'], saved_steps=f['saved_steps'], theta_tail=d.get('theta_tail')))
if not ok: out['stop'] = 'pool_fit not done or wrong step count (fit_sec reached?)'
json.dump(out, open(f'{R}/cx_r1.json', 'w'), indent=1)
PY
fi
exit $rc
