# 4x4 J1-J2 (J2/J1 = 0.5): does the FN/Krylov loop improve a VMC-trained network? (2026-10-06)

Exact ED (D = 12870, E0 = -8.457923351) is used only for scoring and for the exact sampler. eps = (E - E0)/|E0|,
w_s = wrong-sign probability vs ED. CPU-h = wall x 16 cores (16-core jobs on LMU `cluster`, same hardware for every run).
Figure: `net_start_4x4.png` (+ `.pdf`). Code: `experiments/net_start_4x4/` (`vmc_train_4x4.py`, `loop_from_net_4x4.py`,
`exact_loop_from_net.py`, `make_start.py`, `summarize.py`, `common.py`, `vmc_lmu.sbatch`). Raw per-run data: the `*.json` here.

## Setup
1. Network: complex-output translational ViT (`nqsmagic`, log-cosh head, log psi = amplitude + i sign), 2 layers, d=32, 4 heads,
   23,680 parameters. Amplitude AND sign learned; the guide is `a = |psi|`, `s = sgn Re(psi e^{-i th0})` (th0 = arg(sum psi^2)/2;
   imaginary-part fraction <= 2e-5).
   Plain VMC + SR: every step draws N = 4000 i.i.d. samples from the exact |psi|^2, E_loc from exact H (full psi), complex SR with
   real parameters (CG, shift 3e-3), lr cosine-decayed 0.25 -> 0.02 over 1500 steps, trust RMS(delta log psi) <= 0.25.
   Smaller lr (0.05, constant) was far slower (eps 1e-2 after 500 steps); the 155k-param 4-layer ViT (rung-2 size) also trained slower and
   was dropped. Exploratory runs (A-F, I; not used) are not kept.
2. Loop: exactly the rung-2 loop (`loop_from_net_4x4.py` is a copy of `rung2_sampled_sr_loop_4x4.py` with the start replaced):
   exact H_FN[a_k,s_k], residual ViT (155k params, fresh each run) trained on the frozen-H_FN Rayleigh quotient by SR with
   fresh i.i.d. samples every SR step (N per SR step, half train/half val, trust 0.02, shift 0.1, <= 20 SR steps, stop at first
   val reject), then the exact energy-optimal current-sign Krylov step. "E_FN" = ground energy of exact H_FN[a_k,s_k]
   (the reported FN energy; its eps is `eps_FN`).
3. Control: plain VMC continued from the same parameters with the same code.
4. Ceiling: the ideal loop (`exact_loop_from_net.py`), a_{k+1} = exact ground state of H_FN, same Krylov step, no learner, no sampling.

## Trained networks (start points)
| net | recipe | CPU-h | eps(psi) | eps(guide) | w_s | fidelity |
|---|---|---|---|---|---|---|
| **H** (main) | no sign prior, 1500 steps | 13.0 | 3.63e-4 | 3.58e-4 | 2.4e-4 | 0.99926 |
| G | Marshall sign rule x network, 1500 steps | 12.4 | 2.04e-4 | 2.03e-4 | 1.2e-4 | 0.99961 |
| J | as H, only 500 steps (not converged) | 4.9 | 2.87e-3 | 2.76e-3 | 2.0e-3 | 0.9923 |

H is not stuck in a bad sign basin: it learns the sign without a prior. The "plateau" is the end of the annealed schedule
(mean eps of last 100 steps 3.60e-4). Caveat from the control below: with a constant lr 0.05 the same net keeps improving slowly, so
the plateau is a property of the lr schedule, not of the architecture.

## Loop from net H, eps / w_s per iteration (selected, full data in `loopH_*.json`, ideal loop in `exactloop_H.json`)
eps = eps of <H>_guide; eps_FN = eps of E_FN; cum CPU-h counts the loop only (training CPU-h excluded).

| it | N=1e5: eps / eps_FN / CPU-h | N=1e6: eps / eps_FN / CPU-h | N=1e7: eps / eps_FN / CPU-h | ideal: eps / eps_FN | w_s (N=1e7 / ideal) |
|---|---|---|---|---|---|
| 0 (net) | 3.58e-4 / 3.18e-4 / 0 | same | same | same | 2.4e-4 |
| 1 | 2.62e-4 / 3.18e-4 / 0.6 | 2.62e-4 / 3.18e-4 / 0.5 | 2.51e-4 / 3.18e-4 / 1.2 | 2.36e-4 / 2.13e-4 | 2.0e-6 / 1.6e-6 |
| 2 | 2.59e-4 / 2.26e-4 / 1.0 | 2.59e-4 / 2.26e-4 / 0.9 | 2.28e-4 / 2.19e-4 / 5.4 | 1.97e-4 / 1.79e-4 | 1.5e-6 / 1.2e-6 |
| 3 | 2.60e-4 / 2.22e-4 / 1.5 | 2.37e-4 / 2.24e-4 / 3.5 | 2.11e-4 / 2.00e-4 / 10.9 | 1.65e-4 / 1.49e-4 | |
| 5 | 2.60e-4 / 2.19e-4 / 1.8 | 2.24e-4 / 2.07e-4 / 5.9 | 2.02e-4 / 1.87e-4 / 13.0 | 1.14e-4 / 1.03e-4 | |
| 10 | 2.54e-4 / 2.17e-4 / 3.4 | 2.04e-4 / 1.82e-4 / 10.8 | 1.69e-4 / 1.59e-4 / 25.6 | 4.94e-5 / 4.61e-5 | |
| 15 | | 1.89e-4 / 1.68e-4 / 16.6 | 1.47e-4 / 1.30e-4 / 34.6 | 2.65e-5 / 2.51e-5 | 1.3e-6 / 6.6e-7 |
| 20 / 40 (ideal only) | | | | 1.59e-5 / 1.52e-5 (it 20); 4.05e-6 / 3.98e-6 (it 40) | |

N=1e4 (per SR step; 2 seeds, 10 iterations, 1.8-4.8 CPU-h): eps 2.9e-4 (seed 0, best 2.89e-4) and 2.6e-4 -> 5.2e-4 (seed 1; noisy, E_FN 2.3-2.7e-4).
Tuned loop hyper-parameters (shift 1e-2 / 1e-3, trust 0.01, <= 30 SR steps; N=1e5 and 1e6) were not better (final eps 2.1-2.8e-4).

## Other starts (robustness; N per SR step, 10 iterations)
| net | loop | eps start -> final (E_FN) | VMC continuation at equal CPU-h | ideal loop it 10 / it 40 |
|---|---|---|---|---|
| G | N=1e4 (1.6 CPU-h) | 2.03e-4 -> 2.15e-4 (1.64e-4); best 1.62e-4 | 2.07e-4 | 3.2e-5 / 4.9e-6 |
| G | N=1e5 (3.5 CPU-h) | 2.03e-4 -> 1.48e-4 (1.29e-4) | 2.00e-4 | |
| J | N=1e5 (6.9 CPU-h) | 2.76e-3 -> 2.15e-3 (1.84e-3) | 2.2e-3 | 2.2e-4 / 1.2e-5 |
| J | N=1e6 (29.6 CPU-h) | 2.76e-3 -> 1.54e-3 (1.42e-3) | **8.0e-4** (25 CPU-h) | |

## VMC continuation control (same parameters, same code; eps_guide, 100-step mean; extra CPU-h after the 13 CPU-h of training)
| continuation | extra CPU-h | eps_guide | w_s |
|---|---|---|---|
| N=4000, lr 0.02 const | 51 | 2.5e-4 | 1.6e-4 |
| N=1e4, lr 0.02 const | 42 | 1.7e-4 | 8e-5 |
| N=4000, lr 0.05 const (best) | 51 | 1.31e-4 | 6.5e-5 |

Equal-cost comparison on net H (`equal_cost.md` has all rows; VMC = lr 0.05 N=4000 unless said):
loop N=1e5 at 3.4 CPU-h: 2.54e-4 vs VMC 3.37e-4; loop N=1e6 at 16.6: 1.89e-4 vs 2.40e-4; loop N=1e7 at 34.6: 1.47e-4 vs 1.79e-4
(N=1e4/lr 0.02: 2.0e-4). VMC at 51 CPU-h reaches 1.31e-4, i.e. about what the loop reaches at ~45 CPU-h (extrapolated): the advantage is gone by ~50 CPU-h.

## Verdict
- Does the loop improve the trained net? Yes, modestly: 3.58e-4 -> 2.5e-4 after 3 CPU-h (x0.71), 1.5e-4 after 35 CPU-h (x0.41). E_FN: 3.2e-4 -> 1.3e-4.
  The first iteration alone does the sign job: w_s 2.4e-4 -> 2e-6 in every run (the Krylov step, exact in this test; the net, plain VMC, and every
  VMC continuation keep w_s >= 6e-5), giving the first 25-30% of the energy gain for 0.5-1 CPU-h. Same on G and J (J less).
- Not clearly below plain VMC continuation. At equal cost the loop is 20-25% lower than the best continuation (0.75-0.8x), a difference
  of ~0.1 in log10 eps and inside what lr tuning of the control moves; for the unconverged net J the control wins outright
  (8e-4 vs 1.5e-3). The loop is not an order-of-magnitude win and not better than a VMC budget of ~1.5x its cost.
- Iterations needed: the learned loop is still creeping down after 15 iterations (about -3%/iteration at N=1e7); no iteration count closes the gap.
- Where and why it stalls: the ideal loop from the same guide contracts steadily by 0.85/iteration (eps 3.6e-4 -> 5e-5 at it 10, 4e-6 at it 40; w_s ~ 5e-7),
  so FN information is not exhausted. The learned amplitude refresh is the bottleneck, not the sign step:
  (i) the target is tiny: log-ratio RMS between the guide and phi_FN is 0.003-0.005 (fidelity 0.99999), i.e. a 0.4% amplitude correction;
  (ii) at N=1e4 and 1e5 the SR refresh finds nothing resolvable (frozen gain fraction ~0 +/- noise, often 0 accepted SR steps;
  N=1e5 stalls completely at eps 2.6e-4 after iteration 1); more samples help (N=1e7 realises 0.3-0.5 of the frozen gain
  per iteration, but still stops at the first val reject, and sampling-exactness does not remove that); (iii) the cost of resolving
  such small corrections grows (1.2 -> 35 CPU-h for 1e5 -> 1e7), the same VMC-noise-limited scaling as plain VMC.
  Tuning shift/trust did not change this (rows loopH_N1e5_sh*, loopH_N1e6_sh1e-2).
- Practical reading for 6x6: the sign step is a clean, cheap gain (removes the wrong-sign mass in one iteration). The amplitude refresh by a sampled
  network SR does not beat spending the same CPU on VMC; a better refresh learner (the ideal loop shows the headroom: 100x in 40 iterations) is what is missing.

## Caveats
- Exact i.i.d. sampler for both VMC and the loop (ideal walkers); H_FN, sign step and scoring use ED/full basis, so the sign step is exact,
  not sampled. On 4x4 the number of unique configurations saturates (<= 12870), so N=1e7 is nearly the exact-weight limit.
- One seed for the nets and one for most loops; the N=1e4 seed-to-seed spread (final eps 2.9e-4 vs 5.2e-4) is as large as the differences between N=1e5/1e6 loops.
- CPU-h is wall x 16 on shared nodes; the VMC control sampler cost is the same exact-sampler cost.
- VMC continuation runs at constant lr 0.05 were still decreasing when stopped; a longer control would be lower.
