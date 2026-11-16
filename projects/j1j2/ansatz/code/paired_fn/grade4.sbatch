#!/bin/bash
#SBATCH --job-name=sepgrade4
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=12G
#SBATCH --time=13:00:00
#SBATCH --no-requeue
# T-SEP stored arm = 8x8 T-BAD round 4 grade: paired coupled FN of A = StoredCNNSign(n_3) vs B = StoredCNNSign(n_4), M 256, 4 chains x (150 burn + 375 measured), A=A control first (exit 3 if Delta != 0).
# usage: sbatch --output RUN/slurm-%j.out grade4.sbatch RUN     (RUN = /project/theorie/a/A.Otaifi/runs/j1j2/sign_bad/r4/grade4, a symlink to /scratch/a/A.Otaifi/runs_bulk/sign_bad/r4/grade4)
# Budget (measured, probe 17999273): 10.6 s/block, 4 x 525 = 2100 blocks = 6.2 h + control 5 blocks + JIT ~0.1 h = ~6.4 h against 13 h (49%); MaxRSS of the probe 1.7 GB, 12 G requested (7x).
# Watcher (armed at submission): watch2.sh <jobid> GRADE4_DONE $RUN/slurm-<jobid>.out $RUN/progress.txt 1500 $RUN ; GRADE4_DONE is written to BOTH the slurm out and progress.txt.
# Resume: chain_<id>.done files make a resubmission skip finished chains (granularity one chain ~1.55 h). XLA autotune ON (net convs 9-12x slower off), as sep_probe.sbatch.
set -uo pipefail
RUN=$1; mkdir -p "$RUN"
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/paired_fn; T=$RUN/chains_grade4.json
export OMPI_MCA_pml=ob1 OMPI_MCA_btl="vader,self,tcp"
export PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 OMP_NUM_THREADS=1
unset JAX_COMPILATION_CACHE_DIR XLA_FLAGS
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
F=/scratch/a/A.Otaifi/ansatz_runs/sign_bad/r4/final_net.txt; read N4P N4SHA < $F
[ "$N4SHA" = 6ebf24474e2341e79783fd045c47956f93a3596f0a73c5ee943d7325b4f6e34a ] && [ "$(sha256sum $N4P | cut -d' ' -f1)" = "$N4SHA" ] || { echo "GRADE4_FAILED n4 sha" >> $RUN/progress.txt; exit 3; }
{ echo job_id=$SLURM_JOB_ID; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; date -Is; echo "n4 $N4P $N4SHA"; (cd $C; sha256sum engine.py providers.py analysis.py cnn_sign.py sep_prod.py); sha256sum $T; } >> $RUN/run_metadata.txt
cd $C; $PY sep_prod.py $RUN --table $T --net4 $N4P > $RUN/sep_prod.log 2>&1; rc=$?
echo "sep_prod_exit=$rc $(date -Is)" >> $RUN/run_metadata.txt; sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,State -P >> $RUN/run_metadata.txt 2>/dev/null
[ $rc -eq 0 ] && grep -q '"done": true' $RUN/result.json && { M="GRADE4_DONE $(date -Is)"; echo "$M"; echo "$M" >> $RUN/progress.txt; }
exit $rc
