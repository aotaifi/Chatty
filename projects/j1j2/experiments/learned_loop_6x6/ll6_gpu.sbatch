#!/bin/bash
#SBATCH --job-name=ll6
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=/home/a/A.Otaifi/chatty_ll6/logs/%x-%j.out
#SBATCH --error=/home/a/A.Otaifi/chatty_ll6/logs/%x-%j.err
# Learned FN/Krylov loop on 6x6 (experiments/learned_loop_6x6). LMU theorie full A40 (vGPU slices a40-*gb lack CUDA VMM and do not work with jax 0.10).
# Usage: sbatch --job-name=NAME [--gres ...] ll6_gpu.sbatch <script.py> <args...>
set -euo pipefail
BASE=/home/a/A.Otaifi/chatty_ll6
OUT="$BASE/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cp "$BASE"/code/*.py "$OUT/"
cd "$OUT"
for f in "vit_J2=0.50_N=6x6_k=0.mpack" energy_krylov_vs_vit_6x6_indep.npz krylov_scaling_6x6_a1.20_tr4096_va2048.npz krylov_scaling_6x6_a1.20_tr4096_va2048.json samples_psi0sq_6x6.npz samples_extra_psi0sq_6x6.npz state_base.json state_it1.json state_it1b_amp.json state_it1b.json; do ln -sf "$BASE/data/$f" .; done
{ echo "job_id=${SLURM_JOB_ID}"; echo "host=$(hostname)"; echo "args=$*"; echo "started=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > run_metadata.txt
"$BASE/venv_gpu/bin/python" "$@"
echo "finished=$(date -Is)" >> run_metadata.txt
