#!/bin/bash
#SBATCH --job-name=chatty-wake-smoke
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M
#SBATCH --time=00:01:00
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/slack-wake-smoke-%j.out
#SBATCH --error=/pc2/users/h/hpcalot/chatty/j1j2/results/slack-wake-smoke-%j.err
set -euo pipefail
echo "wake smoke start"
sleep 5
echo "wake smoke done"
