#!/bin/bash
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=vit10val
#SBATCH --output=/scratch/a/A.Otaifi/vit10_runs/job/vit10_%j.out
# Paderborn H100 est. start was 2026-11-10 (>1 day) so the brief's fallback (full theorie A40) is used.
export PYTHONPATH=/scratch/a/A.Otaifi/vit10_runs/pkgs XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
cd /project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/experiments/addon_10x10
PY=/project/theorie/a/A.Otaifi/Chatty/runs/a40slice/venv_jax_0.8.2/bin/python
nvidia-smi -L
$PY -u validate_vit10.py bench 2>&1
$PY -u validate_vit10.py energy 1024 2>&1
echo "VIT10 DONE"
