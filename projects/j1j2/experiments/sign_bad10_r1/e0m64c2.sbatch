#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=36G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --job-name=sb10m64c2
#SBATCH --output=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1/m64/slurm_%j.out
set -uo pipefail
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONPATH=/scratch/a/A.Otaifi/vit10_runs/pkgs XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_COMPILATION_CACHE_DIR
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; C=$R/code; PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
# STAGE A (restart chain): e_0 chain 2 at M=64 (new seed 62; same code file, guide, dtau, burn as m64/c1; 300 post-burn blocks, ~22 s/block). m64/c1 (job 17997721) is e_0 chain 1.
O=$R/m64; cd $C
{ echo job_id=$SLURM_JOB_ID; hostname; date -Is; sha256sum $C/gfmc_sign.py $C/s10_net.py $C/vitlib.py; } > $O/run_metadata_c2.txt
$PY -u gfmc_sign.py $O/c2 --sign marshall --guide vit10 --kind symm --prec f32h --L 10 --M 64 --dtau 0.1 --init shipped --chunk 1024 --blocks 300 --burn 30 --seed 62 --maxsec 9500 >> $O/all_c2.log 2>&1; rc=$?; echo "m64 c2 rc=$rc" >> $O/all_c2.log
[ $rc -eq 0 ] && [ -s $O/c2/E_mix.npy ] || exit 3
python3 /project/theorie/a/A.Otaifi/tools/quotable.py series $O/c1/E_mix.npy $O/c2/E_mix.npy --name e0_M64 --plot $O/e0_M64_quotable.png >> $O/all_c2.log 2>&1
echo "E0M64C2_DONE $(date -Is)"; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $O/run_metadata_c2.txt 2>/dev/null
