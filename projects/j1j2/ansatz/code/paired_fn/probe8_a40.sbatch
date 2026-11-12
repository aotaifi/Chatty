#!/bin/bash
#SBATCH --job-name=pfn-probe8
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --no-requeue
# 8x8 rho probe of the paired FN estimator (PROPOSAL, not submitted; needs the lead's OK).  usage: sbatch --output RUN/slurm-%j.out probe8_a40.sbatch RUN [pF]
# RUN must be a new directory under /scratch/a/A.Otaifi/ansatz_runs/paired_fn/probe8_*.  Budget: control 40 blocks + coupled probe within --maxsec 720 s, then two standalone arms of 240 s each
# (for kappa_work); ~22 min of work against a 50 min wall (~0.37 A40-h).  Memory: base_s306_p0 peak 2.6 GiB GPU per chain; host RSS unmeasured for this
# driver -> 32G (the per-configuration cache is capped at 150k entries, ~0.3 GB).
set -uo pipefail
RUN=$1; PF=${2:-1e-3}; mkdir -p "$RUN"
B=/project/theorie/a/A.Otaifi/runs/j1j2/base_s306_p0; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/paired_fn
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export GFMC_M1_CODE=$B/code/m1 G8_PROD8=/scratch/a/A.Otaifi/rbmpp8/rb8_retrain/s306/train/params_final.npz \
       G8_PROD8_SHA=45fa54e601bcf9c1fc79d24825066e9f58bbabca855b6a0de88ce6e120c36c5d G8_CONSTS=$B/inputs/consts_s306.json S306_CODE=$B/code
export JAX_COMPILATION_CACHE_DIR=/scratch/a/A.Otaifi/ansatz_runs/paired_fn/jaxcache JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 OMP_NUM_THREADS=8
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; sha256sum $C/*.py $B/code/*.py $B/code/m1/*.py $G8_PROD8; } > $RUN/run_metadata.txt
$PY $C/probe8.py $RUN --pF $PF --control_blocks 40 --burn 30 --maxsec 720 --sa_sec 240 --M 1024 --X0 $B/inputs/X0_B0s_0.npy > $RUN/probe8.log 2>&1; rc=$?
echo "probe8_exit=$rc $(date -Is)" >> $RUN/run_metadata.txt
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $RUN/run_metadata.txt 2>/dev/null
exit $rc
