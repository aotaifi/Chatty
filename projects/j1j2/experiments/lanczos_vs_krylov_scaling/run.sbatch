#!/bin/bash
#SBATCH --job-name=lvk
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/home/a/A.Otaifi/ChattyRun/j1j2/lanczos_vs_krylov_scaling/logs/%x-%A_%a.out
# array: 0 N16(4x4) 1 N20 2 N24(4x6) 3 N28 4 N32 5 N36(6x6).  Code copied to $B/experiments/.
set -euo pipefail
B=/home/a/A.Otaifi/ChattyRun/j1j2/lanczos_vs_krylov_scaling
ED=/home/a/A.Otaifi/chatty_ed6x6
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
i=$SLURM_ARRAY_TASK_ID
CACHE=/project/theorie/a/A.Otaifi/chatty_lvk/cache
EXTRA=("--L 4" "--L 0 --torus 4,0,1,5" "--L 0 --torus 4,0,0,6" "--L 0 --torus 4,0,1,7" "--L 0 --torus 4,4,-4,4" "--L 6 --gs-vec $ED/run_gs/vectors_k00_A1_p.npy --gs-states $ED/run_gs/states_k00_A1_p.npy")
NAME=(N16 N20 N24 N28 N32 N36)
SEEDS=(64 32 16 8 4 4)
GUIDES=${GUIDES:-A,B,Cx,Cm}
PFX=${PFX:-lvk}
$ED/ls_env/bin/python $B/experiments/lanczos_vs_krylov_scaling/scaling.py ${EXTRA[$i]} --tag ${NAME[$i]} --sigma 0.018 --kappa ${KAPPA:-0.0067} --seeds ${SEEDS[$i]} --guides $GUIDES --cache-dir $CACHE --out $B/results/${PFX}_${NAME[$i]}.json
