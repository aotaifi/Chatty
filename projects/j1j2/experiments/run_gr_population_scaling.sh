#!/bin/zsh
set -euo pipefail
cd /Users/aliotaifi/j1j2_vit_bench
for M in 32 64 128; do
  echo "=== POPSIZE $M ==="
  .venv311/bin/python gfmc_6x6_gr_population_size.py "$M"
done
echo "=== AGGREGATE ==="
/Users/aliotaifi/j1j2_vit_bench/.venv311/bin/python /Users/aliotaifi/Chatty/projects/j1j2/experiments/aggregate_gr_population_scaling.py
