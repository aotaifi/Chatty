#!/bin/bash
#SBATCH --job-name=pf_w
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
# Test W (PLAN_pfaffian.md), CPU only.  Sizing: quick run (2M of 15.8M rows, 300 orbits/decade) 109 s wall on 4 threads ws1 -> full est. part 1 ~10 min, part 2 ~30 min; RSS est. < 15 GB.
set -uo pipefail
C=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code; O=/scratch/a/A.Otaifi/ansatz_runs/pfaffian_w
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
echo "START $(date -Is) $(hostname) job=${SLURM_JOB_ID}" > $O/verdict.txt
$PY $C/test_w_pf.py $O > $O/test_w.log 2> $O/test_w.err; rc=$?
echo "END $(date -Is) rc=$rc" >> $O/verdict.txt
exit $rc
