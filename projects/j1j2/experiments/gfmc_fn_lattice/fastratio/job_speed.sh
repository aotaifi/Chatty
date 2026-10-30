#!/bin/bash
# Runs on the GPU node (called by speed.sbatch): matched-input timing ref vs fast + walker equivalence.  usage: job_speed.sh RUN
RUN=$1; CODE=$RUN/code; DATA=/scratch/a/A.Otaifi/ansatz_runs/fastratio/data
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
CFG=/scratch/a/A.Otaifi/ansatz_runs/gpu_profile/acc_f64/cfg.npz
DEADLINE=${DEADLINE:-420}      # s from job start after which no new optional stage starts (target <= 720 s = 0.2 GPU-h; wall limit 1440 s)
T0=$(date +%s); cd $CODE
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 OMP_NUM_THREADS=8   # as production
mark() { echo "$(date -Is) PROGRESS $*" >> $RUN/progress.txt; }
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; date -Is; md5sum $CODE/*.py; sha256sum $DATA/params_final.npz $DATA/params10_random.npz;
  $PY -c "import jax;print('jax',jax.__version__,jax.devices())"; } > $RUN/run_metadata.txt 2>&1
mark "START job $SLURM_JOB_ID"
mark "BEGIN bench"; timeout 360 $PY bench_gpu.py $RUN/bench.json --p8 $DATA/params_final.npz --consts $DATA/consts.json --cfg8 $CFG --p10 $DATA/params10_random.npz --reps 8 > $RUN/bench.out 2>&1; mark "END bench rc $?"
for m in ref fast; do     # walker equivalence, as gpu_profile check_equiv.py: M 64, 3 blocks, dtau 0.1, seed 1, production params and constants
  mark "BEGIN eq_$m"; timeout 300 $PY gfmc_run.py $RUN/eq_$m --params $DATA/params_final.npz --consts $DATA/consts.json --p 1 --L 8 --proj translations --M 64 --dtau 0.1 --blocks 3 --burn 0 --seed 1 --chunk 256 --ratio $m > $RUN/eq_$m.out 2>&1; mark "END eq_$m rc $?"
done
if [ $(( $(date +%s) - T0 )) -lt $DEADLINE ]; then       # optional: 10x10 GFMC (random params, fake constants), small M, bounded
  for m in ref fast; do
    mark "BEGIN g10_$m"; timeout 200 $PY gfmc_run.py $RUN/g10_$m --params $DATA/params10_random.npz --consts $DATA/consts10_fake.json --p 1 --L 10 --proj translations --M 32 --dtau 0.1 --blocks 2 --burn 0 --seed 1 --chunk 256 --maxsec 60 --ratio $m > $RUN/g10_$m.out 2>&1; mark "END g10_$m rc $?"
  done
else mark "SKIPPED g10 (deadline)"; fi
echo "FASTRATIO_DONE elapsed_s $(( $(date +%s) - T0 ))" >> $RUN/progress.txt
