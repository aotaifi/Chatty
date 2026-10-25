#!/bin/bash
#SBATCH --job-name=sdt-a4x4
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=/project/theorie/a/A.Otaifi/Chatty/runs/ChattyRun/j1j2/sign_design_tests/logs/%x-%A_%a.out
# array: 0..3 = (work net|hop) x (seed 0|1)
set -euo pipefail
module purge; module load python/3.12-2024.10
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
B=/project/theorie/a/A.Otaifi/Chatty/runs/ChattyRun/j1j2/sign_design_tests
export SSL_DIR=/project/theorie/a/A.Otaifi/Chatty/runs/ChattyRun/j1j2/stored_signs
W=(net net hop hop); SD=(0 1 0 1); i=$SLURM_ARRAY_TASK_ID
python $B/a_net_hop_4x4.py --work ${W[$i]} --seed ${SD[$i]} --out $B/results/a4x4_${W[$i]}_s${SD[$i]}.json
