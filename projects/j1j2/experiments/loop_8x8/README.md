# 8x8 route-B test: one FN write-back iteration from the 8x8 ViT, with a same-sample VMC control (pre-registration)

Periodic 8x8 J1-J2, J2/J1 = 0.5. Written 2026-10-09 BEFORE any decisive run.
The only runs before this file were measurements: probe job 17092583 (ViT throughput, pilot VMC, phase, tempered-chain
ESS). Those runs did not train any write-back network and did not evaluate any write-back.
Context: RESEARCH_MAP 3h-3k, writeback_tail README 6b-6d and SCALING_PLAN option B, itfit_6x6 frozen-base feature loop.

## Question
Does one realistic FN write-back of the semi-implicit FN step, stored as b = a exp(f) with a network on frozen-base
one-hop features, lower the variational energy <H_ph> of the 8x8 ViT guide (same sign)?
And does it do so by more than the same network trained by fixed-sign VMC on the same samples?

There are no exact data at 8x8, so everything is sampled. This is the first 8x8 test of the loop's amplitude write-back.

## Method (fixed here)
- **Guide (a, s).**
  - a = |psi_P| of the 8x8 ViT checkpoint `vit_J2=0.50_N=8x8_k=0.mpack`, the network behind our -0.498823(35).
  - psi_P is the projection onto D4 x spin flip: psi_P(x) = sum over the 16 images of the complex ViT, as at 6x6
    (stall_6x6 `st6_run.py`).
  - Rule fixed before the decision: use the projected guide if pool + referee + training fit in <= 14 GPU-h at the
    measured throughput. Otherwise use the translation-invariant network and say so.
  - The probe measured the cost of the projection as 16x per evaluation: 8.5e4 vs 5.2e3 evaluations/s on an A40-24gb
    slice.
  - s = the binarised own sign, s(x) = sign cos(arg psi_P(x) - phi), with one global phase phi from a pilot.
  - **No Krylov sign step in this iteration.** Both arms and the baseline use the same fixed sign s.
- **Frozen-base features.** For each configuration x, from one hop of the guide (~145 valid exchange neighbours):
  - log a(x);
  - V(x) = sum over sign-violating neighbours (s_x s_y = +1) of J/2 a(y)/a(x);
  - W(x) = sum over kept neighbours (s_x s_y = -1) of J/2 a(y)/a(x).
  The network inputs are (log a, log(V + 1e-6), log(W + 1e-6)), standardised by the unweighted pool mean and sd.
  These are affine constants from a guide-only measure.
- **Target.** The semi-implicit FN step, guide only:
  T(x) = log(1 + W) - log(1 + max(H_xx + V - E_a, 1e-12)).
  - E_a = <H> of the guide (a, s), from an independent pilot sample.
  - No FN walkers enter the target, the proposal, the loss or the selection.
- **Network.** The FEAT residual CNN of writeback_tail, adapted to 8x8:
  - periodic 3x3 convolutions, C = 32, 4 layers;
  - a Fourier embedding of the 3 features, a 2-layer MLP head with zero initialisation;
  - about 37k parameters.
  - f is exactly D4 x flip symmetric at evaluation (mean over the 16 images). Training uses one random image per row.
  - b = a exp(f).
- **Samples (pool).** A tail-aware proposal built from guide quantities only.
  - A defensive mixture: half the Metropolis chains sample a^2 (beta = 1), half sample a^1 (beta = 1/2).
  - Balance-heuristic importance weights to a^2: w = a / (s_{1/2} r + s_1 a), where s_1 = s_{1/2} = 1/2 are the chain
    fractions and r = Z_2/Z_1 is the Meng-Wong bridge estimate. The weights are bounded by 1/s_1 = 2.
  - Each configuration gets K = 8 uniformly drawn valid bonds. The pool stores one-hop guide data at x and at the 8
    neighbours y_k.
  - 20% of the chains are held out, split by chain.
  - Measured in the probe: pure a^1 sampling has an ESS of only 2.2% (translation-invariant) / 3.5% (projected) of
    the samples. The mixture has 61% / 76%. This is why the defensive mixture is used.
- **FN arm (write-back).** Edge least squares, as arm F-SI-Adam at 6x6:
  0.5 sum_x w_x sum_k c_xk ((f_x - f_yk) - (T_x - T_yk))^2,
  where c_xk = J/2 a_y/a_x nval/K on kept bonds and 0 on violating bonds.
  - Primary arm FN-Adam: Adam, lr 3e-3, warmup-cosine to 2%, 20k steps, B = 256 (the 6x6 F-SI-Adam settings).
  - Secondary arm FN-minSR: eta 0.05, lambda 1e-3, trust radius 0.01 rms in f (the 6x6 settings).
- **Control (same net, same inputs, same samples, same guide evaluations).** Fixed-sign VMC on <H_ph>(b, s):
  - gradient 2 <(E_L^b - E) df/dtheta>, with self-normalised weights w_x exp(2 f_x) on the same pool;
  - E_L^b(x) = E_L^a(x) + nval/K sum_k J/2 s_x s_y (a_y/a_x)(exp(f_y - f_x) - 1).
  - The guide local energy E_L^a(x) is exact; it is a by-product of the features. The correction uses the same K = 8
    neighbours as the FN arm.
  - So the control sees exactly the FN arm's data: equal samples and equal guide evaluations. This is a lower-variance
    form of the 6x6 C-VMC estimator; same expectation, more generous to the control.
  - Arms:
    - VMC-Adam at lr 3e-4, 1e-4 and 3e-5 (the 6x6 scan), 20k steps, B = 256;
    - VMC-minSR at eta 0.01 and 0.003, lambda 1e-3, trust radius 0.01, 20k steps, B = 256.
- **Checkpoint selection, identical for every arm.**
  - Parameters are stored every 1000 steps.
  - The pre-registered checkpoint is the one with the lowest held-out own objective (symmetric f, 20% held-out chains):
    the edge loss for FN arms, the reweighted energy for VMC arms.
  - The final checkpoint and the 10k checkpoint are also reported (convergence evidence), together with the
    held-out/training loss curves.
- **Referee (paired VMC, independent of training).**
  - Fresh chains at a^2 with their own seeds.
  - For every sample x the guide is evaluated on x and on all valid neighbours y together with their one-hop shells
    (the full two-hop set). E_L^b(x) is therefore exact over all neighbours; there is no bond subsampling.
  - Delta = <H>_b - <H>_a = sum w E_L^b / sum w - mean E_L^a, with w = exp(2 f).
  - The SE comes from per-chain means of the influence w(E_L^b - E_b) - (E_L^a - E_a). Chains are independent;
    the pilot measured a lag-1 autocorrelation of E_L of -0.01 at 1 sweep.
  - **Target SE(Delta_FN) <= 7e-6/site.** If the first referee batch gives a larger SE, more independent chains are
    added until the target is met or the 20 GPU-h budget is reached. The decision depends on the SE only, never on
    Delta.
  - Free by-products of the two-hop data:
    - the unprojected target b_T = a exp(T): the gain contained in the target, and capture = Delta_FN / Delta_T;
    - one Lanczos step on the guide (moments <E_L>, <E_L^2>, <E_L (H^2 psi)/psi>, jackknife over chains): the external
      baseline at the same two-hop cost;
    - the absolute E/N of the guide and of the ViT with its complex phase.
- **FN-DMC referee** (calibrated protocol: tau_max 0.025, M = 512, beta window 0.8-2.4, beta-time-averaged mean).
  - Run only if the budget allows after the above.
  - Estimated cost on the written-back guide: each walker step needs b on N1, i.e. the guide on two hops (~1e4 guide
    evaluations, x16 projected). That is ~7e8 evaluations per population, about 2 h per population on an A40 for the
    translation-invariant guide, so >= 500 populations for SE 3e-6 is far beyond 20 GPU-h.
  - Expected: not run. The final report will say so with the measured numbers.

## Pass / fail (pre-registered)
Delta is measured relative to the guide (the projected ViT if the cost rule allows, else the translation-invariant
ViT), per site, at the pre-registered checkpoint.
- **PASS** if both hold:
  - (i) Delta_FN < 0 with |Delta_FN| > 2 SE(Delta_FN): the FN arm FN-Adam lowers <H_ph> by more than 2 sigma;
  - (ii) |Delta_FN| >= 1.5 max(0, -Delta_ctrl), where Delta_ctrl is the best control arm (most negative Delta at its
    pre-registered checkpoint; choosing it by the referee is conservative for us). If the best control raises the
    energy, (ii) holds.
  - The paired difference Delta_FN - Delta_ctrl and its SE are reported alongside.
- FN-minSR is reported but cannot rescue a failing FN-Adam: the primary arm is fixed.
- **Otherwise:** "inconclusive (cause: X)". The cause is diagnosed with the target diagnostic (Delta_T), the loss
  curves, the held-out gap and the per-arm ESS. A route is called failed only if a conceptual limit is shown.
- Absolute E/N of the guide and of the written-back guide, with errors, are reported next to:
  - RBM+PP -0.498886(1);
  - Hu VMC p=2 -0.49886(1);
  - our ViT -0.498823(35);
  - extrapolations -0.49906(1) / -0.4992(1).
- The Lanczos step on the guide is reported as the external baseline.

## Cost accounting
Stage times from the job logs, as GPU-h on the GPU type used:
- probe;
- pool (sampling and one-hop data);
- training (per arm);
- referee data;
- referee evaluation.
Also reported: ViT evaluations per sample for each stage. Budget <= 20 GPU-h.

## Files
- `l8_core.py`: lattice, ViT guide (vit_dt fp32, highest matmul precision), one-hop data, sampler.
- `l8_probe.py`: stage 0.
- `l8_pool.py`: stage 1.
- `l8_train.py` (with `l8_net.py`): stage 2.
- `l8_ref.py`: stage 3.
- `l8_eval.py`: stage 4.
- `l8.sbatch`: job script.
- Run data: ws1 `/project/theorie/a/A.Otaifi/chatty_loop8/runs/`.
- Results, table and figure: `results/loop_8x8/`.

## Amendment 1 (2026-10-09 12:05, after the probe, before any pool/referee/training run): sizes and the guide decision
**Probe** (job 17092583, A40-24gb slice, 0.2 GPU-h; `results/loop_8x8/probe.json`):
- Throughput:
  - ViT (translation-invariant, 4 patch offsets) 8.5e4 evaluations/s at batch 8192;
  - D4 x flip projected 5.2e3/s (16.4x).
- Pilot <H> at a^2, 8 samples x 1024/512 chains:
  - translation-invariant -0.498853(32) (complex phase) / -0.498853(32) (binarised sign), consistent with the paper's
    -0.498823(35);
  - projected -0.498871(52) / -0.498871(52).
- Global phase phi = 0.0560; the sign leak (mean |sin|) is 4e-4 / 1.3e-4, so binarising the sign costs nothing.
- One hop: 145 valid neighbours. Mean kept weight W = 27.0 and violating weight V = 5.7: 16.6% of the off-diagonal
  weight violates the ViT sign.
- Two hop: 1.0e4 distinct configurations per sample.
- E_L autocorrelation at 1 sweep: -0.01 / +0.003.
- E_L sd per sample: 2.9e-3 / 3.3e-3 per site.

**Guide decision (rule above):**
- Cost of the projected guide at the measured throughput:
  - pool 2.7 h (sampling 0.4 + one-hop data 2.3);
  - referee 6.6 h for 12288 samples (1.9 s per sample);
  - training about 2.5 h.
- Total about 12 GPU-h, which is <= 14 GPU-h, so **the guide is the D4 x flip projected ViT psi_P**.
- E_a = 64 x (-0.4988714) = -31.927772 (projected pilot); phi = 0.056006.

**Sizes:**
- Pool:
  - 2048 chains at beta = 1 and 2048 at beta = 1/2, burn 20 sweeps, 8 samples 1 sweep apart: P = 32768 (seed 1000);
  - K = 8; held-out 20% of chains.
- Training:
  - all arms 20k steps, B = 256, checkpoints every 1000 steps;
  - minSR arms capped at 1.5 h wall each (the reached step is reported).
- Referee:
  - 4 independent jobs x 768 chains x 4 samples (burn 20 sweeps, thin 1 sweep; seeds 5000, 6000, 7000, 8000):
    R = 12288;
  - extension by further independent jobs only if SE(Delta_FN) > 7e-6/site, within the 20 GPU-h budget.
