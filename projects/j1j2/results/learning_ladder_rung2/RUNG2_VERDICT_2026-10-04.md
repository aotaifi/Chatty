# Rung 2, 4x4 learning ladder: network trained on samples, iterated in the FN/Krylov loop (2026-10-04)

Code: `experiments/learning_ladder/rung2_sampled_sr_loop_4x4.py` (+ `rung2_lmu.sbatch`, `rung2_summary.py`).
LMU theorie CPU, venv `~/chatty_fnguides/venv311`, run dir `~/ChattyRun/j1j2/learning_ladder_rung2/`.

## Setup
- Loop: guide (a_k,s_k) -> exact H_FN[a_k,s_k] -> train b = a_k exp(r_theta) -> exact energy-optimal current-sign Krylov step on the learned b -> repeat. Start: |psi(J2=0)|, Marshall signs (same as the exact loops).
- Objective: frozen-H_FN Rayleigh quotient. Production estimator (`--mode vmc`): every SR step draws N fresh i.i.d. samples from the exact |b_theta|^2, half train / half val. E_loc uses the local rows of H_FN plus the network on the H-neighbours. SR uses g = 2<(E_loc-E)(O-<O>)> and the Fisher matrix on the train half (CG, 200 iterations). A step is accepted only if the reweighted val estimate decreases. Training stops at the first rejected step or after 20 steps.
- Network: residual ViT (154,780 params), warm-started across loop iterations. SR shift 0.1, trust RMS(delta r) = 0.02, adaptive shrink.
- Comparison estimator (`--mode reweight`): one fixed set of N samples from the exact a_k^2 per loop iteration, with weights exp(2r).

## Results (mean [min,max] over seeds)
See `rung2_summary.py` output. Key numbers, given as w_s / eps:

| run | it 5 | it 20 | it 30-40 | final |
|---|---|---|---|---|
| exact plain | 1.1e-3 / 4.4e-3 | 8.9e-7 / 2.0e-4 | 8.9e-7 / 2.4e-5 (it 40) | 0 / 1e-8 |
| exact geometric mean | 1.9e-3 / 1.4e-2 | 2.3e-4 / 1.3e-3 | 3.2e-5 / 2.1e-4 (it 40) | 0 / 3e-6 |
| vmc N=1e3 (3 seeds, 150 its) | 1.3e-2 / 5.2e-2 | 1.3e-2 / 3.6e-2 | 1.0e-2 / 2.9e-2 (it 40) | 2.9e-3 / 1.5e-2 (it 150) |
| vmc N=1e4 (3 seeds, 150 its) | 2.8e-3 / 1.7e-2 | 1.1e-3 / 5.6e-3 | 5.0e-4 / 2.7e-3 (it 40) | 2.3e-5 / 8.4e-4 (it 150) |
| vmc N=1e5 (3 seeds, 30 its) | 1.1e-3 / 5.5e-3 | 2.3e-4 / 1.1e-3 | 7.1e-5 / 6.4e-4 (it 30) | (still improving) |
| reweight N=1e5 (1 seed) | 1.1e-3 / 5.4e-3 | 2.2e-4 / 1.2e-3 | 3.7e-5 / 7.2e-4 (it 30) | |

## Verdict
- The loop is stable for every budget and seed. Nothing diverged. E_FN rises only in small noise-level steps (at most about 1.6e-3).
- The loop does not stall hard, but it crawls. Each iteration's refresh is limited by a sampling floor on the guide-to-own-phi_FN log-ratio RMS: about 0.06 at N=1e3, about 0.014 at N=1e4 (it 150) and about 0.011 at N=1e5 (it 30). Once a run is at this floor, the frozen gain fraction per iteration is about 0 plus or minus noise, and most loop iterations accept 1-3 SR steps.
- At N=1e5 per SR step (about 1e6 samples per loop iteration), w_s and eps track the exact geometric-mean loop up to iteration 20. They stay well behind the exact plain loop. Exact-phi counterfactual at the last iteration: the sign step is not the bottleneck; the learned amplitude is.
- Scaling at fixed iteration 20: eps = 3.6e-2, 5.6e-3 and 1.1e-3 for N = 1e3, 1e4 and 1e5, i.e. eps roughly proportional to N^-0.75.
- The reweight estimator can be exploited. Pilot 16809897 (N=1e4) at loop iteration 3: the val estimate fell by 0.037 while the exact frozen energy rose by 0.106. The network raises b on unsampled neighbours. Production reweight runs show negative gains in 25-30% of iterations. This bears directly on the 8x8 frozen-H_FN SR pilots, which used a1^2 samples with reweighting.

## Caveats
- The sampler is exact i.i.d. (an ideal VMC chain). Real walker correlation is rung 3.
- Exact diagonalisation is used for sampling and scoring. The guide is stored as the network chain evaluated on the basis.
- The N=1e5 runs are limited by the number of iterations (30), not yet by the noise floor.
- Budgets in vmc mode are per SR step. Samples per loop iteration were about 3e3, 4-6e4 and 9e5 for the three budgets.
