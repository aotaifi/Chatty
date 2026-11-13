#!/bin/bash
#SBATCH --job-name=pfn-gated
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=38G
#SBATCH --time=01:15:00
#SBATCH --requeue
# 2026-11-13: timing + controls of the gated exact one-step sign (gated_probe.py), one a40-24gb slice.  usage: sbatch --output RUN/slurm-%j.out gated_probe_a40s24.sbatch RUN [gate]
# Plan: warm-up, plain A=A 6 blocks, control 1 (gate -inf) 20 blocks, broken 3 blocks, paired gate -5 until --maxsec 780 s (~28-32 min of work against a 75 min wall).
set -uo pipefail
RUN=$1; GATE=${2:--5}; mkdir -p "$RUN"
B=/project/theorie/a/A.Otaifi/runs/j1j2/base_s306_p0; C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/paired_fn
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export GFMC_M1_CODE=$B/code/m1 G8_PROD8=/scratch/a/A.Otaifi/rbmpp8/rb8_retrain/s306/train/params_final.npz \
       G8_PROD8_SHA=45fa54e601bcf9c1fc79d24825066e9f58bbabca855b6a0de88ce6e120c36c5d G8_CONSTS=$B/inputs/consts_s306.json S306_CODE=$B/code
export JAX_COMPILATION_CACHE_DIR=/scratch/a/A.Otaifi/ansatz_runs/paired_fn/jaxcache JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 OMP_NUM_THREADS=8
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; date -Is; sha256sum $C/*.py $B/code/*.py $B/code/m1/*.py $G8_PROD8; } > $RUN/run_metadata.txt
$PY $C/gated_probe.py $RUN --gate $GATE --maxsec 780 --wchunk 128 > $RUN/gated_probe.log 2>&1; rc=$?
echo "gated_probe_exit=$rc $(date -Is)" >> $RUN/run_metadata.txt
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $RUN/run_metadata.txt 2>/dev/null
exit $rc
