#!/bin/bash
#SBATCH --job-name=brcfn2
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=12G
#SBATCH --time=07:30:00
#SBATCH --no-requeue
# Branched FN grade (STEP 14): A = n_2 (r2), B = CF (tcf/CF/fit/net_8192.pkl sha 5dcca951), 8x8, M 256, A burn 100, K 8 pairs spaced 25, each 20 burn + 100 measured, A=A control on every snapshot, seed 7101.
# usage: sbatch --output RUN/slurm-%j.out branched_probe_a40s24.sbatch RUN [M] [MAXSEC]      (compute guard MAXSEC s: no new pair if projected to overrun)
# Budget: from probe 2 (15.2 s/pair block, 10.1 s/A block): 8 x 120 x 15.2 + 275 x 10.1 + ~300 = ~17700 s = 4.9 A40-h; x1.4 = 24700 s: wall 7.5 h, maxsec 24500; host MaxRSS 12 G requested as probe 2.
set -uo pipefail
RUN=$1; M=${2:-256}; MAXSEC=${3:-24500}; mkdir -p "$RUN"
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/paired_fn
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; date -Is; echo "M $M maxsec $MAXSEC"; (cd $C; sha256sum engine.py providers.py analysis.py cnn_sign.py branched_fn.py branched_prod.py); } >> $RUN/run_metadata.txt
cd $C; $PY branched_prod.py $RUN --M $M --K 8 --netA r2 --netB cf --seed 7101 --burn_A 100 --spacing 25 --burn_pair 20 --measure_pair 100 --control_blocks 5 --maxsec $MAXSEC > $RUN/branched_prod.log 2>&1; rc=$?
echo "branched_prod_exit=$rc $(date -Is)" >> $RUN/run_metadata.txt; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $RUN/run_metadata.txt 2>/dev/null
[ $rc -eq 0 ] && grep -q '"done": true' $RUN/result.json && { MSG="BRANCHED_DONE $(date -Is)"; echo "$MSG"; echo "$MSG" >> $RUN/progress.txt; }
exit $rc
