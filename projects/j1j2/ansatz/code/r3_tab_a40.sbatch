#!/bin/bash
#SBATCH --job-name=pf_r3tab
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
# Round 3 under s_G: la tables of theta_step900 and theta_step1200 (afterok on pf_r3).  Table-only path of cx_r1.py: spec max_steps 0 + warm_theta = checkpoint -> cx_r1fit.py loads
# theta = warm_theta, skips the step loop (n < 0 false), saves theta/offset, then table(theta) -> round_1/la_table.npy (+ decade/bond readout, skip_efn).  Environment as r2g_a40.sbatch.
# usage: r3_tab_a40.sbatch ROOT   (ROOT/fit/code, ROOT/tabs/tab_{900,1200}/spec_r1.json)
set -uo pipefail
ROOT=$1; CODE=$ROOT/fit/code
L=/project/theorie/a/A.Otaifi/Chatty/runs
export ST6_CSR=$L/stall6/csr ST6_TABLE=$L/stall6/data/psi0_6x6_table.npz ST6_SYM=$L/stall6/data/sym_tables.npz CX_LOOP8=/unused_by_r1
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=$L/a40slice/venv_jax_0.8.2/bin/python
export PYTHONPATH=/project/theorie/a/A.Otaifi/Chatty/runs/compress6/code
M=$ROOT/tabs/run_metadata.txt
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } >> "$M"
cd "$CODE"
$PY -c "import jax; d=jax.devices(); print(d); assert d[0].platform in ('gpu','cuda'), d" >> "$M" 2>&1 || { echo "no CUDA device" >> "$M"; exit 3; }
rcmax=0
for N in 900 1200; do
  RUN=$ROOT/tabs/tab_$N; export CX_TABDIR=$RUN/tables; mkdir -p "$RUN"
  [ -s "$ROOT/fit/round_1/theta_step$N.npy" ] || { echo "tab_$N: theta_step$N.npy missing" >> "$M"; rcmax=5; continue; }
  echo "tab_$N start $(date -Is)" >> "$M"
  $PY cx_r1.py "$RUN" "$RUN/spec_r1.json"; rc=$?
  echo "tab_${N}_exit=$rc finished=$(date -Is)" >> "$M"; [ $rc -gt $rcmax ] && rcmax=$rc
done
echo "tabs_exit=$rcmax finished=$(date -Is)" >> "$M"
exit $rcmax
