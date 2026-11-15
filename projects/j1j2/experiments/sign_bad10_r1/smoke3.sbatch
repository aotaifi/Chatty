#!/bin/bash
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=36G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --job-name=sb10smoke3
#SBATCH --output=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1/smoke3/slurm_%j.out
# SMOKE of the five round-1 stages with tiny sizes + their controls (and checks that must FAIL when broken).  Not production data.
set -uo pipefail
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONPATH=/scratch/a/A.Otaifi/vit10_runs/pkgs XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset XLA_FLAGS JAX_COMPILATION_CACHE_DIR
R=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/r1; C=$R/code; O=$R/smoke3; S0=/project/theorie/a/A.Otaifi/runs/j1j2/sign_bad10/step0; PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python; L=$O/all.log
mkdir -p $O; cp $R/smoke/consts_real.json $R/smoke/consts_real.json.sha256 $O/; sed -i "s#r1/smoke/#r1/smoke3/#" $O/consts_real.json 2>/dev/null; sha256sum $O/consts_real.json | cut -d" " -f1 > $O/consts_real.json.sha256
echo "SMOKE3 $(date -Is)" >> $L
V=$O; cd $C
echo "=== S3 labels (6 chains x 1 sample, 2 chunks of 3), consts from the smoke vmc json" >> $L
V=$O; $PY -u labels10.py $O/labels --consts $V/consts_real.json --consts_sha <(echo deadbeef) --chains 6 --nsamp 1 --burn_sweeps 2 --thin_sweeps 1 --chunk 3 >> $L 2>&1; echo "S3 BROKEN sha rc=$? (must be non-zero)" >> $L
$PY -u labels10.py $O/labels --consts $V/consts_real.json --consts_sha $V/consts_real.json.sha256 --chains 6 --nsamp 1 --burn_sweeps 2 --thin_sweeps 1 --chunk 3 --no_sampler_gate >> $L 2>&1; echo "S3 rc=$?" >> $L
echo "=== S4 fit (roots 2 4, 100 steps)" >> $L
$PY -u fit10.py $O/fit $O/labels --roots 2 4 >> $L 2>&1; echo "S4 rc=$?" >> $L
echo "=== S5 FN with the smoke net (2 chains x 3 blocks) + analysis" >> $L
read NETP NETSHA < $O/fit/final_net.txt
for c in 1 2; do $PY -u gfmc_sign.py $O/fn$c --guide vit10 --kind symm --prec f32h --L 10 --M 16 --dtau 0.1 --init shipped --chunk 1024 --blocks 3 --burn 0 --sign cnn --cnn_path $NETP --cnn_sha256 $NETSHA --seed 4$c --maxsec 300 >> $L 2>&1; echo "S5 c$c rc=$?" >> $L; done
$PY -u gfmc_sign.py $O/fnbad --guide vit10 --kind symm --prec f32h --L 10 --M 16 --blocks 1 --burn 0 --sign cnn --cnn_path $NETP --cnn_sha256 0000 >> $L 2>&1; echo "S5 BROKEN net sha rc=$? (must be non-zero)" >> $L
$PY $C/analyze_fn10.py --e0 $S0/probe/fn_marshall_M16 $S0/probe/fn_marshall_M64 --e1 $O/fn1 $O/fn2 --out $O/an >> $L 2>&1; echo "S5 analyze rc=$?" >> $L
echo "SMOKE_DONE $(date -Is)" | tee -a $L
