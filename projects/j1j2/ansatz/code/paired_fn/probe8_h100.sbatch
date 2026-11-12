#!/bin/bash
#SBATCH --account=hpc-prf-ppdmrg
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --no-requeue
# Paderborn H100 copy of probe8_a40.sbatch (same arguments; placed here because theorie inter queued > 1 h).  usage: sbatch [--qos express_hpcalot] --output RUN/slurm-%j.out probe8_h100.sbatch RUN [pF]
set -uo pipefail
RUN=$1; PF=${2:-1e-3}; mkdir -p "$RUN"
PD=/scratch/hpc-prf-ppdmrg/hpcalot/paired_fn; B=/scratch/hpc-prf-ppdmrg/hpcalot/base_s306_p0; C=$PD/code
export GFMC_M1_CODE=$PD/base_code/m1 G8_PROD8=$B/inputs/params_final_s306.npz \
       G8_PROD8_SHA=45fa54e601bcf9c1fc79d24825066e9f58bbabca855b6a0de88ce6e120c36c5d G8_CONSTS=$B/inputs/consts_s306.json S306_CODE=$PD/base_code
export JAX_COMPILATION_CACHE_DIR=$PD/jaxcache JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 OMP_NUM_THREADS=8
PY=/scratch/hpc-prf-ppdmrg/hpcalot/venv_jax_0.8.2/bin/python
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; sha256sum $C/*.py $PD/base_code/*.py $PD/base_code/m1/*.py $G8_PROD8; } > $RUN/run_metadata.txt
$PY $C/probe8.py $RUN --pF $PF --control_blocks 40 --burn 30 --maxsec 720 --sa_sec 240 --M 1024 --X0 $B/inputs/X0_B0s_0.npy > $RUN/probe8.log 2>&1; rc=$?
echo "probe8_exit=$rc $(date -Is)" >> $RUN/run_metadata.txt
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $RUN/run_metadata.txt 2>/dev/null
exit $rc
