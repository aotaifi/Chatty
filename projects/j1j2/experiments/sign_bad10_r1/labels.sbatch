#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=36G
#SBATCH --time=13:00:00
#SBATCH --no-requeue
#SBATCH --job-name=sb10lab
#SBATCH --output=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1/labels/slurm_%j.out
set -uo pipefail
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONPATH=/scratch/a/A.Otaifi/vit10_runs/pkgs XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_COMPILATION_CACHE_DIR
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; C=$R/code; PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
# STAGE 3: K[Marshall] labels, 1000 tempered chains x 4 samples = 4000 roots (800 train chains = 3200 roots, 200 held-out chains = 800 roots), controls after chunk 0 and at the end (gate, exit 3).
# Restartable by chunk (resubmit the same script after a TIMEOUT; the sampler samples.npz is reused).
O=$R/labels; cd $C; V=$R/vmc
{ echo job_id=$SLURM_JOB_ID; hostname; date -Is; sha256sum $C/*.py $V/consts_alpha0_E0.json; cat $V/consts_alpha0_E0.json.sha256; } >> $O/run_metadata.txt
$PY -u labels10.py $O --consts $V/consts_alpha0_E0.json --consts_sha $V/consts_alpha0_E0.json.sha256 --chains 1000 --nsamp 4 --seed 11 >> $O/all.log 2>&1; rc=$?; echo "labels rc=$rc" >> $O/all.log
[ $rc -eq 0 ] || exit 3
echo "LABELS_DONE $(date -Is)"; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $O/run_metadata.txt 2>/dev/null
