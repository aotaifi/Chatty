#!/bin/bash
#SBATCH --job-name=b1a40
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/a/A.Otaifi/tooling_bench/b1/a40/slurm-%j.out
# B1 A40 timing: ONE job, hard cap 30 min = 0.5 A40-h.  usage: sbatch b1_a40.sbatch
BENCH=/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/angles/tooling/bench
OUT=/scratch/a/A.Otaifi/tooling_bench/b1
PY=/scratch/a/A.Otaifi/venvs/tooling/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "node $(hostname)"; nvidia-smi -L; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
sha256sum $BENCH/b1_lib.py $BENCH/time_bench.py
rc=0
A="--arms R,Q,L,H,Qc,Lc,INV --nsteps 16 --reps 9"
timeout 420 $PY -I $BENCH/time_bench.py --out $OUT/a40/s1.json --n 50 --B 1024,256 $A;  echo "STAGE s1 rc=$?"
timeout 420 $PY -I $BENCH/time_bench.py --out $OUT/a40/s2.json --n 50 --B 4096 $A;      echo "STAGE s2 rc=$?"
timeout 360 $PY -I $BENCH/time_bench.py --out $OUT/a40/s3.json --n 32 --B 1024,256,4096 $A; echo "STAGE s3 rc=$?"
timeout 300 $PY -I $BENCH/time_bench.py --out $OUT/a40/s4.json --n 50 --B 1024 --arms L --delayed 8,16 --nsteps 16 --reps 9; echo "STAGE s4 rc=$?"
timeout 150 $PY -I $BENCH/time_bench.py --out $OUT/a40/s5_realf64.json --n 50 --B 1024 --arms R,Q,L,H,INV --nsteps 16 --reps 9 --real; echo "STAGE s5 rc=$?"
$PY -I $BENCH/merge_a40.py $OUT && rc=0 || rc=1
if [ -s $OUT/time_a40.json ] && [ $rc -eq 0 ]; then echo "B1_VERDICT G COMPLETE"; else echo "B1_VERDICT G FAILED rc=$rc"; fi
