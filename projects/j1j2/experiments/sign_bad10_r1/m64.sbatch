#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=36G
#SBATCH --time=02:50:00
#SBATCH --no-requeue
#SBATCH --job-name=sb10m64
#SBATCH --output=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1/m64/slurm_%j.out
set -uo pipefail
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONPATH=/scratch/a/A.Otaifi/vit10_runs/pkgs XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_COMPILATION_CACHE_DIR
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; C=$R/code; PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
# STAGE 2: population-bias check: ONE e_0 chain at M=64 (same code, guide, dtau, burn as the M16 chains; 300 post-burn blocks, about 20 s/block), compared with the extended M16 e_0 chains.  Gate |e_0(M64) - e_0(M16)| < 2e-4, exit 3 on failure.
O=$R/m64; cd $C
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name --format=csv,noheader; date -Is; sha256sum $C/gfmc_sign.py $C/s10_net.py $C/vitlib.py; } > $O/run_metadata.txt
$PY -u gfmc_sign.py $O/c1 --sign marshall --guide vit10 --kind symm --prec f32h --L 10 --M 64 --dtau 0.1 --init shipped --chunk 1024 --blocks 300 --burn 30 --seed 61 --maxsec 8400 >> $O/all.log 2>&1; echo "m64 rc=$?" >> $O/all.log
$PY $C/m64_gate.py $R/e0ext/c1 $R/e0ext/c2 $O/c1 $O/gate.json >> $O/all.log 2>&1; rc=$?; echo "gate rc=$rc" >> $O/all.log
[ $rc -eq 0 ] || exit 3
echo "M64_DONE $(date -Is)"; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $O/run_metadata.txt 2>/dev/null
