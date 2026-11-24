#!/bin/bash
#SBATCH --job-name=br2probe
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
# Branched probe 2 (positive control): A = n_2 (r2) , B = n_3 (r3) at 8x8, M 256, A burn 100, K 6 pairs spaced 25, each 20 burn + 100 measured, A=A control on every snapshot.
# usage: sbatch --output RUN/slurm-%j.out branched_probe_a40s24.sbatch RUN [M] [MAXSEC]      (compute guard MAXSEC s: no new pair if projected to overrun)
# Budget: ~2.7 A40-h (6 x 120 pair blocks x 11 s + 225 A blocks x 5.5 s + controls + JIT = ~9600 s) against 6 h wall (45%); host MaxRSS 1.5 GB at M64 (grade4 1.7 GB at M256), 12 G requested.
set -uo pipefail
RUN=$1; M=${2:-256}; MAXSEC=${3:-20000}; mkdir -p "$RUN"
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/paired_fn
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; date -Is; echo "M $M maxsec $MAXSEC"; (cd $C; sha256sum engine.py providers.py analysis.py cnn_sign.py branched_fn.py branched_prod.py); } >> $RUN/run_metadata.txt
cd $C; $PY branched_prod.py $RUN --M $M --K 6 --netA r2 --netB r3 --burn_A 100 --spacing 25 --burn_pair 20 --measure_pair 100 --control_blocks 5 --maxsec $MAXSEC > $RUN/branched_prod.log 2>&1; rc=$?
echo "branched_prod_exit=$rc $(date -Is)" >> $RUN/run_metadata.txt; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $RUN/run_metadata.txt 2>/dev/null
[ $rc -eq 0 ] && grep -q '"done": true' $RUN/result.json && { MSG="BRANCHED_DONE $(date -Is)"; echo "$MSG"; echo "$MSG" >> $RUN/progress.txt; }
exit $rc
