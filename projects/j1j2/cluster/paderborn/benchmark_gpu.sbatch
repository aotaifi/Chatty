#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
module load lang
module load Python/3.11.5-GCCcore-13.2.0
source ~/chatty/.venv-j1j2-py311/bin/activate
RUN=$(mktemp -d "${TMPDIR:-/tmp}/chatty-j1j2-bench.XXXXXX")
cp ~/chatty/j1j2/src/gfmc_6x6_gr_population_size.py "$RUN/"
cp ~/chatty/j1j2/src/benchmark_vit_steady.py "$RUN/"
cp ~/chatty/j1j2/inputs/* "$RUN/"
cd "$RUN"
nvidia-smi -L
python benchmark_vit_steady.py
