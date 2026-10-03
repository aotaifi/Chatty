#!/bin/bash
#SBATCH --job-name=chatty-k2temp
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alyotaifi@gmail.com
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/k2temp-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/k2temp-%j.err
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source /pc2/users/h/hpcalot/chatty/.venv-j1j2-py311/bin/activate
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
BASE=/pc2/users/h/hpcalot/chatty/j1j2
cd "$BASE/src"
python k2_gnn_tempered_edges_oneblock20.py   --M 100000 --dt .0125 --ntrain 30000 --nval 10000   --epochs 15 --alpha .6 --mix .5   --out "$BASE/results/k2temp_M100000_dt0125_a06_mix05.json"
