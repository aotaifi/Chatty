#!/bin/bash
# DRAFT, not submitted. FN-DMC with the one-Lanczos-step guide psi_1 (PREREG E2 protocol, M = 512, beta to 3.2,
# window [0.8, 2.4] + late-window transient check). Each walker move needs psi_1 on ~145 neighbours = psi_P on the
# two-hop ball (~1e4 projected evaluations). Restartable (fn.npz); resubmit until SE <= 1e-5 or the cap.
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=fnpsi1
#SBATCH --output=/pc2/users/h/hpcalot/chatty/j1j2/results/delC-fnpsi1-%j.out
source /pc2/users/h/hpcalot/chatty/j1j2/src/delC/dc_common.sh
I=$BASE/inputs/delC
python dc_fn.py "$BASE/results/delC_fnpsi1" psi1 --beta 3.2 --batch_pops 2 --npop_max 64 --min_pop 16 --se_target 1e-5 \
  --hours 46 --evbatch 8192 --pool "$I/l8ref7000_X.npz,$I/l8ref8000_X.npz,$I/l8ref10000_X.npz"
echo "finished=$(date -Is)" >> "$OUT/run_metadata.txt"
