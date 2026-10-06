# Amplitude update for the FN/Krylov loop: design memo (draft 3, 2026-10-06)

Draft 3 adds results: the RGN trust-region prototype (linear models), T1 (one amplitude refresh with the ViT) and T2 (a 10-15 iteration loop vs VMC at equal CPU-h), all on 4x4 from the trained ViT "net H".
Drafts 1 and 2 are kept as `DESIGN_MEMO_draft1.md` and `DESIGN_MEMO_draft2.md`. They contain the identities, Test A (FN surrogate vs VMC objective), the 6x6 sampling pre-check and the literature, which this draft does not repeat.

Figure: `vit_t1t2.png` / `.pdf`. Tables: `vit_t1t2_table.md`. Per-run data: `vit/*.json`.

## 0. Bottom line
**Headline:** on 4x4, starting from a ViT trained by VMC for 13 CPU-h (eps = 3.6e-4), the loop "Krylov sign step + fixed-sign second-order VMC amplitude update" lowers the energy error by:
- **83% at 0.5 CPU-h, 90% at 1 CPU-h, 95% at 2 CPU-h and 97% at 3 CPU-h** (variational <H> of the guide; 2 seeds).
- The fixed-node energy E_FN of the final guides is 99% lower: 4.0e-6 vs 3.2e-4.

At equal CPU-h the best VMC baselines do much less:
- The same RGN optimiser on the complex ViT, learning sign and amplitude jointly: 40 / 61 / 72 / 77%. It stalls at about 6e-5 and its steps cost 4-5x more.
- Standard SR VMC (the recipe that trained the net): 2-11% within 5 CPU-h. It needs about 50 CPU-h for 62%.
- The same RGN optimiser with the net's signs held fixed: 32-46%.

The answer to the user's question is therefore **yes on 4x4**: "~90% lower error than the trained net at 1 CPU-h (8% of the training cost), at least 3.5x lower error than equal-cost VMC with the same optimiser" (3.5x at 0.5 CPU-h, 4x at 1, 6x at 2, 9x at 3 CPU-h). Caveats are in §4. The most important is that the sign step is exact on 4x4 (§4.1).

What makes it work, in order of importance:
1. **The Krylov sign step.** The same optimiser with net signs reaches only 32-47%; the net's sign error w_s = 2.4e-4 is never repaired. With Krylov signs, w_s drops to about 1e-6 at iteration 1 and below 1e-7 later.
2. **A second-order amplitude step with a step-length check on fresh, independent samples.** A gate built on reweighted validation samples can be exploited; this check cannot.
3. **Tempered sampling (beta = 1/4)** and an adaptive sample size N.

The FN surrogate (C1, draft 1) **diverges** on the ViT without a trust region. FN is used only for signs and as the referee/bound.

## 1. Prototype: RGN with a trust region checked on fresh samples (linear models; `vmc_vs_fn_4x4.py --grid proto`)
- RGN-TR solves (M_H + λ A_+) d = -g_H.
  - A_+ is the kept-edge stoquastic Laplacian.
  - Steps follow Levenberg-Marquardt: accept if the paired decrease measured on an *independent fresh tempered sample* is at least 0.25 x the predicted decrease. Then λ /= 3 if the ratio is above 0.75; otherwise λ *= 4, with up to 6 tries.
- Settings: 6 steps, N = 1e4 per step, 2 seeds. eps of the next-iteration E_FN at k1 (ideal FN step: 1.79e-4):

| model, beta | C1 | hybrid | RGN-TR, S damping | **RGN-TR, A_+ damping, λ0 = 0.03** |
|---|---|---|---|---|
| rf, 1/4 | 1.83e-4 | 1.26e-4 | 7.0e-5 | **1.9e-5** |
| tab, 1/4 | 1.79e-4 | 0.93e-4 | 1.36e-4 | **5.3e-5** |
| rf, 1/2 | 1.87e-4 | 1.54e-4 | 2.05e-4 | 1.54e-4 |

- At N = 1e5 (tab, beta = 1/4): RGN-TR gives 5.7e-6.
- At k0 (net signs): RGN-TR gives 1.85-1.9e-4 vs C1 2.13e-4. The gain is small without the sign step.
- **Choice for the ViT:** RGN-TR with A_+ damping, λ0 = 0.03, beta = 1/4.

## 2. T1: one amplitude refresh with the ViT (`vit_rgn_4x4.py`, 6 steps)
- Setup:
  - net H (23,680 parameters), fine-tuned directly.
  - Signs from one exact Krylov step on the net amplitude.
  - Exact sampler at beta = 1/4; 2 seeds.
  - CPU-h on 12-core cip nodes.
- eps of E_FN of the guide; the start is 3.18e-4 (`vit_t1t2_table.md`):

| optimiser | N = 1e4 | N = 1e5 | CPU-h |
|---|---|---|---|
| **RGN-TR (A_+)** | **4.4e-5 (-86%)** | **3.2e-5 (-90%)** | 0.5-0.6 |
| SR + quadratic-model step, checked on fresh samples | 5.7e-5 (-82%) | 4.2e-5 (-87%) | 0.3 |
| hybrid (A_+-preconditioned gradient) | 9.7e-5 (-70%) | 9.1e-5 (-71%) | 0.3 |
| C1 (FN surrogate, no gate) | diverges at step 3 (eps ~0.2) | diverges | 0.3 |
| RGN-TR, **net signs** (control) | 2.2e-4 (-31%) | 2.3e-4 (-28%) | 0.4 |

- **T1 passes** (criteria: ≤ 1.2e-4 at N = 1e4, ≤ 6e-5 at N = 1e5).
- Notes:
  - SR does well here once the step length comes from the quadratic model and the check uses fresh samples. On the ViT, the second-order advantage over SR is about 1.3x per refresh, not 4-10x as in the linear models.
  - Without a trust region, C1's linearisation fails on the network.

## 3. T2: loop vs VMC at equal CPU-h (`vit_rgn_4x4.py`, `t1t2_summary.py`)
Arms, all at N = 1e4 per step and beta = 1/4 unless stated:
- **Loop:** per iteration, one exact Krylov sign step on the current amplitude, then 3 RGN-TR steps at fixed sign; 15 iterations or 4 CPU-h; 2 seeds.
  - "Adaptive" means λ is reset at every iteration and capped at 30 λ0. N is doubled, up to 1.6e5, after a step that fails all 6 checks (the noise floor).
  - Without adaptation the loop stalls at about 5.5e-5 after 0.5 CPU-h, because λ grows without bound once the N = 1e4 estimates can no longer verify a step.
- **Net-sign control:** the same optimiser with the net's signs fixed.
- **Complex-ViT VMC control:** the same RGN-TR on the full complex log ψ (sign and amplitude learned jointly), with S damping because A_+ needs a fixed sign. Runs: 2 non-adaptive seeds and 1 adaptive seed.
- **Standard VMC:** the net's own SR recipe (N = 4000, lr 0.05) on the same nodes. A 51 CPU-h run of the same recipe on 16-core nodes is shown for reference.

Error reduction vs the trained net, at equal CPU-h after training (geometric mean over runs):

| CPU-h | loop (adaptive) | loop, 1 RGN step/iter, non-adaptive | RGN, net signs | RGN, complex ViT | SR VMC |
|---|---|---|---|---|---|
| 0.5 | 6.1e-5 (**83%**) | 5.8e-5 (84%) | 2.5e-4 (32%) | 2.2e-4 (40%) | 3.6e-4 (2%) |
| 1 | 3.5e-5 (**90%**) | 5.5e-5 (85%, stalled) | 2.3e-4 (37%) | 1.4e-4 (61%) | 3.4e-4 (6%) |
| 2 | 1.7e-5 (**95%**) | stalled | 2.1e-4 (42%) | 1.0e-4 (72%) | 3.5e-4 (4%) |
| 3 | 9.0e-6 (**97%**) | stalled | 1.9e-4 (46%) | 8.3e-5 (77%) | 3.2e-4 (11%) |
| final | 7.5e-6 at 3.4 CPU-h (98%); E_FN 4.0e-6 (99%) | | 1.9e-4 at 3.7 (47%) | 5.8e-5 at 10 (84%); E_FN 3.1e-5 | 3.3e-4 at 4.6 (10%); long run 1.4e-4 at 50 (62%) |

Reading:
- The loop's advantage over complex-ViT RGN, which finds good signs itself (w_s 2e-6), has two sources:
  - The sign is not a learned variable, so each amplitude step is a cheap real fixed-sign step: 20-35 s vs 160-700 s.
  - The amplitude optimisation has no sign degrees of freedom, so it does not stall.
- The ideal FN loop (draft 1) reached 4.9e-5 at iteration 10. The learned loop does better (7.5e-6), because its amplitude objective is the VMC energy, not the conservative MM step.

## 4. Caveats (honest limits)
1. **The sign step is exact on 4x4:** an energy-optimal global threshold over the full basis, at a cost of about 1 s. On 6x6 it is a stored sign net plus one hop, which was lossy for the factor nets.
   - **For T3 (coordinator, angle 2):** distil the *composite* sign (s_hat x c_k), not the step factors. Use one-hop neighbours, about 8e4 steps, beta = 0.5.
2. **Jacobians are computed on the full basis**, for every arm alike (on 4x4 the one-hop neighbourhood of 1e4 samples is the full basis). On 6x6 the A_+ products need neighbour Jacobians, about 0.6-2 ms per sample on an A40 (draft 2 §3). Tempering at 6x6 gave only a 1.3x gain for paired differences, and **the noise floor is reached earlier there**.
3. **Seeds:** 2 for the loop, the non-adaptive net-sign and complex controls, and T1; 1 for the adaptive complex and net-sign controls and for SR VMC. The complex-ViT RGN might do better with a sign-aware damping or adaptive N tuned for it. Its adaptive run hit the noise floor at about 6e-5 and then spent 6 CPU-h failing checks.
4. **The exact sampler is ideal VMC.** Real MCMC adds autocorrelation; this is the same for all arms.
5. **The net-sign control is a ViT whose phase was learned by VMC.** The loop's sign step is free on 4x4 but costs about 1 GPU-h per iteration on 6x6 (it2 accounting).

## 5. Plan
- **T3 (6x6), not started; same structure:**
  - Sign: stored composite-sign net plus one hop.
  - Amplitude: 3-6 RGN-TR steps with A_+ damping, beta in {1/4, 1/2}, adaptive N from 3.2e4.
  - Referee: FN at M = 512, with a cumulative certificate.
  - Controls: complex-ViT RGN-TR and the ViT's own SR at equal GPU-h.
- **Go/no-go 0 (about 1.5 A40-h):** one RGN-TR step from G2 must reduce the paired <H> by more than 3σ at N ≤ 2e5.
- **Pass:** guide <H> below the ViT VMC energy by ≥ 2σ after 3 iterations (about 15-20 GPU-h).

## 6. Open questions
1. Is the 4x4 result paper-worthy as "second-order fixed-sign VMC + Krylov sign = 90% error reduction at 8% of training cost", with FN only as signs and referee? Or do we need the 6x6 result first?
2. Should the complex-ViT control get more tuning (adaptive N, sign-aware damping) before we quote "≥3.5x lower error at equal cost"?
3. For 6x6: the sign step (1.1 GPU-h) now costs more than the amplitude step. Do we amortise it, i.e. one sign step per 3-6 amplitude steps?

Files:
- Experiments (`experiments/amp_design/`):
  - `vmc_vs_fn_4x4.py` (Test A, prototype)
  - `vit_rgn_4x4.py` and `vit_rgn_lmu.sbatch` (T1/T2)
  - `t1t2_summary.py` (table and figure)
- ws1 run dir: `~/ChattyRun/j1j2/amp_design/` (`logs/`, `results/`).
- Compute this round: 67.5 CPU-h allocated on cip 12-core nodes (59 CPU-h excluding exact scoring), plus about 3 local CPU-h for the prototype.
