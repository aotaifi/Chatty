#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=36G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --job-name=sb10e0x
#SBATCH --output=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1/e0ext/slurm_%j.out
set -uo pipefail
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONPATH=/scratch/a/A.Otaifi/vit10_runs/pkgs XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_COMPILATION_CACHE_DIR
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; C=$R/code; PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
# STAGE 1: extend the step-0 e_0 chains (Marshall x |ViT|, M16) to >= 300 post-burn blocks by RESUME of the checkpoints (original code copy step0/code, sha in run_metadata).
O=$R/e0ext; S0=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/step0; K=$S0/code; cd $K
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name --format=csv,noheader; date -Is; sha256sum $K/gfmc_sign.py $S0/prod/e0_c1/ckpt.npz $S0/prod/e0_c2/ckpt.npz $S0/prod/e0_c1/E_mix.npy $S0/prod/e0_c2/E_mix.npy; } > $O/run_metadata.txt
for c in 1 2; do
  mkdir -p $O/c$c; cp -p $S0/prod/e0_c$c/ckpt.npz $S0/prod/e0_c$c/progress.txt $S0/prod/e0_c$c/E_mix.npy $O/c$c/ ; cp -p $S0/prod/e0_c$c/E_mix.npy $O/c$c/E_mix_before.npy
  cmp $S0/prod/e0_c$c/ckpt.npz $O/c$c/ckpt.npz || exit 2
  NB=$($PY -c "import numpy as np;print(int(np.load('$O/c$c/ckpt.npz',allow_pickle=True)['nb']))"); echo "chain $c ckpt nb=$NB" >> $O/all.log
  echo $NB > $O/c$c/orig_nblocks.txt
  $PY -u gfmc_sign.py $O/c$c --resume --sign marshall --guide vit10 --kind symm --prec f32h --L 10 --M 16 --dtau 0.1 --init shipped --chunk 1024 --blocks 330 --burn 30 --seed 2$c --maxsec 1500 >> $O/all.log 2>&1; echo "c$c rc=$?" >> $O/all.log
  grep -c "RESUMED at block" $O/c$c/progress.txt >> $O/all.log
  $PY $C/verify_resume.py $O/c$c/E_mix_before.npy $O/c$c $NB 30 >> $O/all.log 2>&1 || { echo "RESUME_PROOF failed chain $c" >> $O/all.log; exit 3; }
done
echo "E0EXT_DONE $(date -Is)"; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $O/run_metadata.txt 2>/dev/null
