#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=36G
#SBATCH --time=00:50:00
#SBATCH --no-requeue
#SBATCH --job-name=sb10smoke
#SBATCH --output=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1/smoke/slurm_%j.out
# SMOKE of the five round-1 stages with tiny sizes + their controls (and checks that must FAIL when broken).  Not production data.
set -uo pipefail
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONPATH=/scratch/a/A.Otaifi/vit10_runs/pkgs XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_COMPILATION_CACHE_DIR
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; C=$R/code; O=$R/smoke; S0=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/step0; PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python; L=$O/all.log
echo "SMOKE job $SLURM_JOB_ID $(date -Is) $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> $L
echo "=== S1 resume (3 extra blocks of e0_c1, step0 code)" >> $L; cd $S0/code; mkdir -p $O/e0; cp -p $S0/prod/e0_c1/{ckpt.npz,progress.txt,E_mix.npy} $O/e0/; cp -p $S0/prod/e0_c1/E_mix.npy $O/e0/E_mix_before.npy
NB=$($PY -c "import numpy as np;print(int(np.load('$O/e0/ckpt.npz',allow_pickle=True)['nb']))"); echo "ckpt nb=$NB" >> $L
$PY -u gfmc_sign.py $O/e0 --resume --sign marshall --guide vit10 --kind symm --prec f32h --L 10 --M 16 --dtau 0.1 --init shipped --chunk 1024 --blocks $((NB-30+3)) --burn 30 --seed 21 --maxsec 600 >> $L 2>&1
$PY $C/verify_resume.py $O/e0/E_mix_before.npy $O/e0 $NB 30 >> $L 2>&1; echo "S1 verify rc=$?" >> $L
$PY $C/verify_resume.py $O/e0/E_mix_before.npy $O/e0 $((NB+5)) 30 >> $L 2>&1; echo "S1 BROKEN verify (wrong block count) rc=$? (must be 3)" >> $L
cd $C
echo "=== S2 vmc: exact vs subsampled h2, then the stage script" >> $L
$PY -u vmc_check_sub.py 5 8 16 > $O/vmc_check_sub.json 2>>$L; echo "S2a rc=$?" >> $L
$PY -u vmc_run2.py $O/vmc --C 4 --Kb 16 --samples 40 --maxsec 150 --seed 5 --nexact_rounds 1 >> $L 2>&1; echo "S2b rc=$?" >> $L
echo '{"alpha": -0.1698576, "E": -49.22316, "note": "smoke constants = step-0 VMC (job 17996634), NOT production"}' > $O/consts_real.json; sha256sum $O/consts_real.json | cut -d' ' -f1 > $O/consts_real.json.sha256
echo "=== S3 labels (6 chains x 1 sample, 2 chunks of 3), consts from the smoke vmc json" >> $L
V=$O; $PY -u labels10.py $O/labels --consts $V/consts_real.json --consts_sha <(echo deadbeef) --chains 6 --nsamp 1 --burn_sweeps 2 --thin_sweeps 1 --chunk 3 >> $L 2>&1; echo "S3 BROKEN sha rc=$? (must be non-zero)" >> $L
$PY -u labels10.py $O/labels --consts $V/consts_real.json --consts_sha $V/consts_real.json.sha256 --chains 6 --nsamp 1 --burn_sweeps 2 --thin_sweeps 1 --chunk 3 >> $L 2>&1; echo "S3 rc=$?" >> $L
echo "=== S4 fit (roots 2 4, 100 steps)" >> $L
$PY -u fit10.py $O/fit $O/labels --roots 2 4 >> $L 2>&1; echo "S4 rc=$?" >> $L
echo "=== S5 FN with the smoke net (2 chains x 3 blocks) + analysis" >> $L
read NETP NETSHA < $O/fit/final_net.txt
for c in 1 2; do $PY -u gfmc_sign.py $O/fn$c --guide vit10 --kind symm --prec f32h --L 10 --M 16 --dtau 0.1 --init shipped --chunk 1024 --blocks 3 --burn 0 --sign cnn --cnn_path $NETP --cnn_sha256 $NETSHA --seed 4$c --maxsec 300 >> $L 2>&1; echo "S5 c$c rc=$?" >> $L; done
$PY -u gfmc_sign.py $O/fnbad --guide vit10 --kind symm --prec f32h --L 10 --M 16 --blocks 1 --burn 0 --sign cnn --cnn_path $NETP --cnn_sha256 0000 >> $L 2>&1; echo "S5 BROKEN net sha rc=$? (must be non-zero)" >> $L
$PY $C/analyze_fn10.py --e0 $S0/probe/fn_marshall_M16 $S0/probe/fn_marshall_M64 --e1 $O/fn1 $O/fn2 --out $O/an >> $L 2>&1; echo "S5 analyze rc=$?" >> $L
echo "SMOKE_DONE $(date -Is)" | tee -a $L
