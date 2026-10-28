#!/bin/bash
# Test C production: six jobs, NOT run by the worker.  Sized from probe 4040973 (H100, B 8208): real 3.67 s/step, complex 6.68 s/step (ratio 1.82); F1 A overhead outside the fit about 10 min (75 min wall, 1070 steps x 3.66 s).
#   real: ~100 steps = 6 min + ~10 min = ~16 min expected; wall 00:45:00 (fit cap 1200 s) -> <= 0.75 H100-h each
#   cplx: max_steps 1100; wall 02:25:00 (fit cap 7200 s) -> <= 2.42 H100-h each
#   total worst case 3 x 0.75 + 3 x 2.42 = 9.5 H100-h (+ probe 0.02) <= 10.
# usage: submit_c1_all.sh --submit   (without --submit: stage only, no sbatch)
cd "$(dirname "$0")"
for s in 1 2 3; do
  MODE=real SEED=$s SBATCH_TIME=00:45:00 ./launch_c1.sh "$@"
  MODE=cplx SEED=$s SBATCH_TIME=02:25:00 MAX_STEPS=1100 ./launch_c1.sh "$@"
done
