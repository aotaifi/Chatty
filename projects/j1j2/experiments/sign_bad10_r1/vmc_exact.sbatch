#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=36G
#SBATCH --time=01:40:00
#SBATCH --no-requeue
#SBATCH --job-name=sb10vmcx
#SBATCH --output=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1/vmc_exact/slurm_%j.out
set -uo pipefail
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONPATH=/scratch/a/A.Otaifi/vit10_runs/pkgs XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_COMPILATION_CACHE_DIR
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; C=$R/code; PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
# STAGE B: alpha_0, E_0 with the EXACT two-hop estimator: 400 samples, 8 independent Metropolis chains, jackknife over chains; gate alpha_err/|alpha| <= 0.15 (exit 3 otherwise). Constants json + sha written once.
O=$R/vmc_exact; cd $C
{ echo job_id=$SLURM_JOB_ID; hostname; date -Is; sha256sum $C/*.py; } > $O/run_metadata.txt
$PY -u vmc_run3.py $O --C 8 --K 100 --samples 400 --maxsec 4800 --seed 5 >> $O/all.log 2>&1; rc=$?; echo "vmc_exact rc=$rc" >> $O/all.log
[ $rc -eq 0 ] && [ -s $O/consts_alpha0_E0.json.sha256 ] || exit 3
echo "VMCX_DONE $(date -Is)"; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $O/run_metadata.txt 2>/dev/null
