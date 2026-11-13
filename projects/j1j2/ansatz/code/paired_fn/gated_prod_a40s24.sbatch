#!/bin/bash
#SBATCH --job-name=g5-prod-cip
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=38G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
# 2026-11-13 G(-5) production, theorie cip copy (one a40-24gb slice).  usage: sbatch --output RUN/slurm-%j.out gated_prod_a40s24.sbatch RUN
# Budget: 4 chains x 170 blocks = 680 blocks x 131 s (measured, gated_probe timing, cip 17967173) = 24.7 h + control 5 blocks (~10 s each) + JIT/warm-up ~0.1 h = ~24.9 h against the 48 h partition limit
# (52%; the 2 d cap is the longest wall cip allows, so the margin is 1.9x).  --requeue: a node failure restarts the script; chain_<id>.done files make it resume at chain granularity (one chain ~6.2 h).
set -uo pipefail
RUN=$1; mkdir -p "$RUN"
B=/project/theorie/a/A.Otaifi/runs/j1j2/base_s306_p0; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/paired_fn; G=/project/theorie/a/A.Otaifi/runs/j1j2/sign_strong/g5
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export GFMC_M1_CODE=$B/code/m1 G8_PROD8=/scratch/a/A.Otaifi/rbmpp8/rb8_retrain/s306/train/params_final.npz \
       G8_PROD8_SHA=45fa54e601bcf9c1fc79d24825066e9f58bbabca855b6a0de88ce6e120c36c5d G8_CONSTS=$B/inputs/consts_s306.json S306_CODE=$B/code
export JAX_COMPILATION_CACHE_DIR=/scratch/a/A.Otaifi/ansatz_runs/paired_fn/jaxcache JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 OMP_NUM_THREADS=8
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
{ echo "job_id=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0}"; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is
  sha256sum $C/*.py $B/code/*.py $B/code/m1/*.py $G/chains_g5.json $G/consts_s306_p1.json $G8_PROD8; } >> $RUN/run_metadata.txt
$PY $C/gated_prod.py $RUN --table $G/chains_g5.json --x0dir $B/inputs --consts /project/theorie/a/A.Otaifi/runs/j1j2/headline_s306/inputs/consts_s306_p1.json --wchunk 128 > $RUN/gated_prod.log 2>&1; rc=$?
echo "gated_prod_exit=$rc $(date -Is)" >> $RUN/run_metadata.txt
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $RUN/run_metadata.txt 2>/dev/null
exit $rc
