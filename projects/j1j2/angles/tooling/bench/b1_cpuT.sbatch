#!/bin/bash
#SBATCH --job-name=b1cpuT
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/a/A.Otaifi/tooling_bench/b1/cpuT/slurm-%j.out
# B1 CPU timing, one thread, B=64, n=32 and 50.  usage: sbatch b1_cpuT.sbatch
BENCH=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/angles/tooling/bench
OUT=/scratch/a/A.Otaifi/tooling_bench/b1
PY=/scratch/a/A.Otaifi/venvs/tooling/bin/python
export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CPU0=$(taskset -pc $$ | sed 's/.*: //' | cut -d, -f1 | cut -d- -f1)
echo "node $(hostname) $(lscpu | grep 'Model name') pinned core $CPU0"; uptime
sha256sum $BENCH/b1_lib.py $BENCH/time_bench.py
rc=0
taskset -c $CPU0 $PY -I $BENCH/time_bench.py --out $OUT/time_cpu.json --B 64 --n 32,50 --arms R,Q,L,H,Qc,Lc,INV --nsteps 8 --reps 15 || rc=1
echo "STAGE time_cpu rc=$rc"
taskset -c $CPU0 $PY -I $BENCH/time_bench.py --out $OUT/time_cpu_realf64.json --B 64 --n 32,50 --arms R,Q,L,H,INV --nsteps 8 --reps 15 --real || rc=1
echo "STAGE time_cpu_real rc=$rc"
uptime
if [ $rc -eq 0 ]; then echo "B1_VERDICT T COMPLETE"; else echo "B1_VERDICT T FAILED rc=$rc"; fi
