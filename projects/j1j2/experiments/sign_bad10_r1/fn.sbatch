#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=36G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --job-name=sb10fn
#SBATCH --output=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1/fn_net/slurm_%j.out
set -uo pipefail
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONPATH=/scratch/a/A.Otaifi/vit10_runs/pkgs XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_COMPILATION_CACHE_DIR
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; C=$R/code; PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
# STAGE 5: FN c=0 of (|ViT|, n_1): 2 chains, M16, dtau 0.1, burn 30, 330 blocks; then analyze_fn10 (quotable on e_0 extended and e_1, g_1).  Net path + sha read from fit/final_net.txt.
O=$R/fn_net; cd $C; read NETP NETSHA < $R/fit/final_net.txt
{ echo job_id=$SLURM_JOB_ID; hostname; date -Is; echo net $NETP $NETSHA; sha256sum $C/gfmc_sign.py $C/s10_net.py $C/vitlib.py; } >> $O/run_metadata.txt
FN="--guide vit10 --kind symm --prec f32h --L 10 --M 16 --dtau 0.1 --init shipped --chunk 1024 --blocks 330 --burn 30 --sign cnn --cnn_path $NETP --cnn_sha256 $NETSHA"
$PY -u gfmc_sign.py $O/c1 $FN --seed 41 --maxsec 6500 >> $O/all.log 2>&1; echo "c1 rc=$?" >> $O/all.log
$PY -u gfmc_sign.py $O/c2 $FN --seed 42 --maxsec 6500 >> $O/all.log 2>&1; echo "c2 rc=$?" >> $O/all.log
$PY $C/analyze_fn10.py --e0 $R/e0ext/c1 $R/e0ext/c2 --e1 $O/c1 $O/c2 --out $O >> $O/all.log 2>&1
echo "FN_R1_DONE $(date -Is)"; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $O/run_metadata.txt 2>/dev/null
