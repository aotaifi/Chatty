#!/bin/bash
#SBATCH --job-name=sdt-b
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/sign_design_tests/logs/%x-%A_%a.out
# array: 0 N16(4x4) 1 N20 2 N24(4x6) 3 N28 4 N32 5 N36(6x6)
set -euo pipefail
B=/home/a/A.Otaifi/ChattyRun/j1j2/sign_design_tests
ED=/home/a/A.Otaifi/chatty_ed6x6
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
i=$SLURM_ARRAY_TASK_ID
CACHE=$B/cache
EXTRA=("--L 4" "--L 0 --torus 4,0,1,5" "--L 0 --torus 4,0,0,6" "--L 0 --torus 4,0,1,7" "--L 0 --torus 4,4,-4,4" "--L 6 --gs-vec $ED/run_gs/vectors_k00_A1_p.npy --gs-states $ED/run_gs/states_k00_A1_p.npy")
NAME=(N16 N20 N24 N28 N32 N36)
$ED/ls_env/bin/python $B/b_depth_sym.py ${EXTRA[$i]} --dmax 8 --cache-dir $CACHE --out $B/results/b_depth_${NAME[$i]}.json
