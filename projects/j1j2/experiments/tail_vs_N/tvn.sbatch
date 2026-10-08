#!/bin/bash
#SBATCH --job-name=tvn
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/project/theorie/a/A.Otaifi/chatty_tail_vs_N/logs/%x-%j.out
# usage: sbatch [--cpus-per-task=32 --mem=100G --job-name=tvn36] tvn.sbatch CLUSTER KMAX [extra args of tvn.py]
set -euo pipefail
B=/project/theorie/a/A.Otaifi/chatty_tail_vs_N
ED=/home/a/A.Otaifi/chatty_ed6x6
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
CL=$1; K=$2; shift 2
$ED/ls_env/bin/python $B/code/tvn.py --cluster $CL --kmax $K --cache-dir $B/cache --out $B/out/tvn_$CL.json "$@"
