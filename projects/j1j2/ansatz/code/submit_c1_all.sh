#!/bin/bash
# Test C production: six jobs, NOT run by the worker.  Sized from probe 4040973 (H100, B 8208): real 3.67 s/step, complex 6.68 s/step (ratio 1.82); F1 A overhead outside the fit about 10 min (75 min wall, 1070 steps x 3.66 s).
#   real: ~100 steps = 6 min + ~10 min = ~16 min expected; wall 00:40:00 (fit_sec 900 s cap) -> <= 0.67 H100-h each
#   cplx: max_steps 600 = 67 min + ~12 min = ~80 min expected; wall 02:00:00 (fit_sec 5700 s) -> <= 2.0 H100-h each
#   total worst case 3 x 0.67 + 3 x 2.0 = 8.0 H100-h (+ probe 0.02) <= 10; expected ~ 3 x 0.27 + 3 x 1.35 = 4.9.
# usage: submit_c1_all.sh --submit   (without --submit: stage only, no sbatch)
cd "$(dirname "$0")"
for s in 1 2 3; do
  MODE=real SEED=$s SBATCH_TIME=00:40:00 ./launch_c1.sh "$@"
  MODE=cplx SEED=$s SBATCH_TIME=02:00:00 MAX_STEPS=600 ./launch_c1.sh "$@"
done
