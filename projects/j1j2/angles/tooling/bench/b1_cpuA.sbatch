#!/bin/bash
#SBATCH --job-name=b1cpuA
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/a/A.Otaifi/tooling_bench/b1/cpuA/slurm-%j.out
# B1 correctness + exact-rational check + drift, one core.  usage: sbatch b1_cpuA.sbatch
BENCH=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/angles/tooling/bench
OUT=/scratch/a/A.Otaifi/tooling_bench/b1
PY=/scratch/a/A.Otaifi/venvs/tooling/bin/python
export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "node $(hostname) $(lscpu | grep 'Model name')"
sha256sum $BENCH/b1_lib.py $BENCH/correct.py $BENCH/drift.py $BENCH/hp_check.py
rc=0
$PY -I $BENCH/correct.py $OUT || rc=1
echo "STAGE correct rc=$rc"
$PY -I $BENCH/hp_check.py $OUT/hp_check.json || rc=1
echo "STAGE hp_check rc=$rc"
$PY -I $BENCH/drift.py $OUT 2000 || rc=1
echo "STAGE drift rc=$rc"
if [ $rc -eq 0 ]; then echo "B1_VERDICT A COMPLETE"; else echo "B1_VERDICT A FAILED rc=$rc"; fi
