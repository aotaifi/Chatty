#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=36G
#SBATCH --time=03:30:00
#SBATCH --no-requeue
#SBATCH --job-name=sb10fit
#SBATCH --output=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1/fit/slurm_%j.out
set -uo pipefail
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONPATH=/scratch/a/A.Otaifi/vit10_runs/pkgs XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_COMPILATION_CACHE_DIR
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; C=$R/code; PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
# STAGE 4: fit n_1 (64x10 CNN, screen arm d recipe) + learning curve 1000/2000/3200 train roots; final net = largest.
O=$R/fit; cd $C
{ echo job_id=$SLURM_JOB_ID; hostname; date -Is; sha256sum $C/*.py $R/labels/labels_done.json; } >> $O/run_metadata.txt
grep -q '"controls": "PASS"' $R/labels/labels_done.json || { echo "labels controls not PASS" >> $O/all.log; exit 2; }
$PY -u fit10.py $O $R/labels --roots 1000 2000 3200 >> $O/all.log 2>&1; rc=$?; echo "fit rc=$rc" >> $O/all.log
[ $rc -eq 0 ] && [ -s $O/final_net.txt ] || exit 3
echo "FIT_DONE $(date -Is)"; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $O/run_metadata.txt 2>/dev/null
