#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=36G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --job-name=sb10smoke4
#SBATCH --output=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1/smoke4/slurm_%j.out
# SMOKE of the five round-1 stages with tiny sizes + their controls (and checks that must FAIL when broken).  Not production data.
set -uo pipefail
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONPATH=/scratch/a/A.Otaifi/vit10_runs/pkgs XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_COMPILATION_CACHE_DIR
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; C=$R/code; O=$R/smoke4; PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python; L=$O/all.log
mkdir -p $O; cd $C
echo "SMOKE4 $(date -Is)" >> $L
$PY -u vmc_run3.py $O/vmc --C 4 --samples 8 --maxsec 200 --seed 5 >> $L 2>&1; echo "vmc_run3 rc=$? (3 expected: 8 samples cannot pass the alpha gate)" >> $L
python3 -c "import json;d=json.load(open('$O/vmc/consts_alpha0_E0.json'));print({k:d[k] for k in ('n_samples','chains','estimator','alpha','alpha_err','passed','gate_reasons')}, 'paired' in str(d.keys()))" >> $L 2>&1
echo "SMOKE_DONE $(date -Is)" | tee -a $L
