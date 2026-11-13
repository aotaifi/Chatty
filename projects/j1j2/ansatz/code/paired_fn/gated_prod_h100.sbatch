#!/bin/bash
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --qos=express_hpcalot
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
# 2026-11-13 G(-5) production, Paderborn H100 copy.  usage (on login4, PATH incl. /opt/software/slurm/25.11.8-1/bin):
#   sbatch --output RUN/slurm-%j.out gated_prod_h100.sbatch RUN      (RUN = /scratch/hpc-prf-ppdmrg/hpcalot/sign_strong_g5/h100)
# Budget: A40 slice 24.9 h; the H100/A40 ratio for THIS kernel is unmeasured (5x gives 5.0 h, 15x 1.7 h on the related 6x6 kernel), so 08:00 is the max the lead allowed (<= 8 h, 1.6x margin at 5x,
# fails to finish below 3.1x).  Billing 16 per H100-h on this account: the lead's grant is <= 40 H100-h.  If it times out, chain_*.done persist: resubmit the same script to resume at chain granularity.
set -uo pipefail
RUN=$1; mkdir -p "$RUN"
PD=/scratch/hpc-prf-ppdmrg/hpcalot/paired_fn; W=/scratch/hpc-prf-ppdmrg/hpcalot/sign_strong_g5; B=/scratch/hpc-prf-ppdmrg/hpcalot/base_s306_p0; C=$W/code
export GFMC_M1_CODE=$PD/base_code/m1 G8_PROD8=$B/inputs/params_final_s306.npz \
       G8_PROD8_SHA=45fa54e601bcf9c1fc79d24825066e9f58bbabca855b6a0de88ce6e120c36c5d G8_CONSTS=$B/inputs/consts_s306.json S306_CODE=$PD/base_code
export JAX_COMPILATION_CACHE_DIR=$PD/jaxcache JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 OMP_NUM_THREADS=8
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
{ echo "job_id=$SLURM_JOB_ID"; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is
  sha256sum $C/*.py $PD/base_code/*.py $PD/base_code/m1/*.py $W/inputs/chains_g5.json $W/inputs/consts_s306_p1.json $G8_PROD8; } >> $RUN/run_metadata.txt
$PY $C/gated_prod.py $RUN --table $W/inputs/chains_g5.json --x0dir $B/inputs --consts $W/inputs/consts_s306_p1.json --wchunk 128 > $RUN/gated_prod.log 2>&1; rc=$?
echo "gated_prod_exit=$rc $(date -Is)" >> $RUN/run_metadata.txt
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $RUN/run_metadata.txt 2>/dev/null
exit $rc
