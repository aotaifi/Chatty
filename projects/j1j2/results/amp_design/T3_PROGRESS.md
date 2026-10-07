# T3: 6x6 loop at the ViT (progress log)

Goal: from the 6x6 ViT (VMC <H> = -0.503654(21)/site; guide FN at M=512 = -0.503666(8)), run the 4x4 T2 loop.
Each iteration: one sign step, then 3 fixed-sign RGN amplitude steps with a trust region verified on fresh samples.
Pass: FN of the final guide is >= 2 sigma below the ViT-guide FN.

Hard ceiling: 30 GPU-h. Stop rule: if after 2 iterations the cumulative <H> gain is < 2 sigma, stop and report.

- Code: `experiments/amp_design/` — `t3_rgn.py` (amplitude stage), `t3_distill.py` (composite sign net), `t3_gpu.sbatch`.
- Reused from `learned_loop_6x6_it2`: `it2_sign.py` (sign step with robust T), `it2_fn.py` (FN referee).
- Run dir: ws1 `~/chatty_t3/`.

## Design decisions (2026-10-06 21:30)
- **Start sign: G1 = K2vit = ss6 K1 net (L 48x6, b0.3) + one exact hop on |ViT| (T = -7.74). ED w_s = 6.0(1.2)e-5.**
  - Why not the it2 D2 net: D2 = K1net x net2 is a stacked factor net (w_s 4.9e-4). Angle 2 forbids stacking factor nets.
  - The K1 net's target, c1 relative to Marshall, *is* the composite sign at k = 1. So K1 net + hop is the composite + hop form.
- **Later sign steps:**
  - First distil the current hop sign into ONE new composite net (`t3_distill.py`): labels s * Marshall, beta 0.5 samples of the current amplitude, all one-hop neighbours, 8e4 Adam steps, L 48x6.
  - Then one exact hop on the current amplitude, with T from `it2_sign.py`.
- **Amplitude stage (`t3_rgn.py`):** the T2 recipe, adapted to P = 155k parameters.
  - RGN curvature is projected on a Krylov subspace {S^+ g, (S^+ M)^j S^+ g}, k = 8, with S^+ = minSR on the training samples.
  - Neighbour terms are computed by jvp/vjp on a curvature subset (2048 samples, all edges); Jacobians are never stored.
  - Damping = kept-edge Laplacian + S. The trust region is checked on an independent fresh sample set, with adaptive lambda/N.
  - beta = 1/2: the 6x6 pre-check found beta = 1/2 the best tempering for the plain estimator, and beta = 1/4 worse.
- **Referee:** `it2_fn.py`, M = 512, beta 2.4, burn 0.8, tau 0.025, seeds 72001-032 and 75001-032. These are the same seeds as the ViT-guide references -0.503654(10) and -0.503677(11), combined -0.503666(8).

## Log
- 21:38 smoke test 16851444 (2080Ti): the pipeline runs end to end.
  - The projected M was indefinite with only 256 curvature samples (min eigenvalue -450), so no step was taken.
  - Fixed: damping matrix = kept-edge Laplacian + S, lambda raised to 1.5x the smallest PSD value. Smoke 2: 16851490.
- Local data with the hop sign: about 39 ms/sample on the 2080Ti (level 2). This dominates the cost, as in it2.
- Certificate noise at the start (2048 samples): per-sample SD of the paired <H>_G - E_ViT = 2.0e-3/site.
  - So 3.2e4 samples give SE ~1.1e-5 (chain effects aside).
  - At the start the ViT reweighting is exact, so ESS = 1.
- 22:00-23:30 **The 6x6 curvature is noise-dominated** (`t3_rgn.py --diag`, 2080Ti, N = 1024-2048, beta 0.5 and 1).
  - The projected RGN Hessian has large *negative* eigenvalues (-100 to -980). The violating-edge term enters with negative weight.
  - The two halves of the curvature sample disagree by 3-30x. The top eigenvalues of A_+ are 2795 vs 7215; one heavy-tailed sample dominates. The diagonal (E_loc - E) term is small.
  - On 4x4 (full basis, exact sampler) none of this happened.
  - Changes:
    - Curvature weights are winsorised at the 0.999 quantile.
    - The damping matrix includes S.
    - minSR uses eigh (the float32 Gram matrix is not exactly PSD; Cholesky failed at N = 2048).
    - The Jacobian is kept in host memory, so 2080Ti GPUs fit N = 8192. Full A40s were unavailable all evening (queue priority).
  - **Mode 'ls'**: the trust region is verified on fresh samples over a candidate set. The candidates are RGN subspace solutions at 4 damping values plus the minSR direction at 4 step sizes. The best candidate is accepted only if its paired decrease on the independent verification states exceeds 2 SE (chain jackknife).
- 23:15 2080Ti probes, N = 2048, iteration-1 guide:
  - **No candidate is resolved.** minSR candidates measure -1.0..-2.0e-5/site with SE 0.9-2.3e-5. RGN candidates are tiny (heavily damped).
  - **The minSR step generalises poorly:** RMS(delta log a) is 0.03 on the verification states vs 0.005 on the training states (6x). A larger minSR shift (eps 0.1) does not change this.
  - Next: N = 8192 (job 16852278), with the Stage-0 certificate at 32768 samples before and after.
- 23:20 **T2 control tuning (CPU, 4 runs x ~5 CPU-h, about 20 CPU-h).** Complex-ViT RGN-TR with adaptive N; damping S or S + sign-aware kept-edge Laplacian; lam0 0.03 or 0.1; seed 2.
  - Best run: 4.8e-5 at 4.45 CPU-h. Sign-aware damping is no better.
  - Equal-cost factor vs the BEST of all 7 complex runs at each CPU-h (`vit_t1t2_table.md`): 3.3x at 0.5 CPU-h, 3.8x at 1, 5.0x at 2, 7.2x at 3. **It survives the tuned control.**
- 2026-10-07 00:00 **Stage 0 (noise budget): GO.**
  - Certificate at the start (G1 = |ViT| + K2vit sign, 32768 fresh |b|^2 samples): <H>_G1 - E_ViT = +0.46(0.79)e-5/site, so the absolute value is -0.503650(23).
  - The per-sample SD gives SE = 7.9e-6 at 3.3e4 samples, i.e. 3.2e-6 at 2e5.
  - A 3-sigma certificate therefore needs a cumulative gain of only >= 9.6e-6/site at 2e5 samples. The measured per-step gains below are 1-2e-5/site, so 3e-5 would need about 2e4 samples.
- 00:40 **At 6x6 the step direction decides; curvature does not.** Iteration-1 job 16852278 and probe 16852616, N = 8192, 2080Ti.
  - minSR with a small shift (1e-3 of the mean eigenvalue of K) **raises** the energy on the verification states: +1.3e-5 to +3.3e-3/site.
  - Its RMS(delta log a) is 11x larger on verification states than on training states (0.056 vs 0.005). With P = 155k >> N = 8192 it interpolates the training samples and does not generalise.
  - The plain gradient and strongly shifted SR (shift = 1 or 10 x mean eigenvalue) **generalise**: drms_va / drms_tr is about 4-6, and the energy decrease is resolved:

    | direction | step RMS (train) | decrease |
    |---|---|---|
    | sr_eps1 | 0.005 | -1.04(0.37)e-5/site |
    | sr_eps1 | 0.010 | -1.80(0.71)e-5/site |
    | gd | 0.005 | -0.71(0.33)e-5/site |

  - RGN subspace solutions are tiny, because the noise-dominated curvature forces heavy damping.
  - Changes: the Krylov basis now starts from shifted SR (eps_k = 1). The candidates are RGN(4) + sr_eps1 / sr_eps10 / gd at RMS 0.003-0.024. The accepted candidate is the most negative one among those with z > 2.
  - Iteration 1 was rerun as 16852730 (3 steps, N = 8192, certificate at the end).
- 02:40 **Iteration 1 done** (16852730, 2080Ti, 2.08 GPU-h: local data 1.26, Jacobian 0.24, curvature 0.03, trust region 0.09, certificate 0.46).
  - Step 1 (N = 8192): no candidate had z > 2. The best was gd at -5.6(4.2)e-6. N doubled to 16384.
  - Step 2: accepted gd, RMS 0.006, -8.3(3.9)e-6/site on the verification states.
  - Step 3: not accepted, -1.2(2.4)e-6.
  - **Certificate at the end (32768 fresh samples): <H>_G - E_ViT = -0.88(3.38)e-5/site, so <H> = -0.503663(40). Not resolved.**
  - The per-sample SD of the paired estimator grew from 1.35e-3 (start) to 8.4e-3/site, although the ViT-weight ESS is still 0.9999.
    The small amplitude update (RMS 0.03 on beta = 0.5 states) changes local energies strongly on a few low-amplitude configurations, so the certificate becomes heavy-tailed.
    **The Stage-0 noise estimate (made at b = |ViT|) was 4x too optimistic.** At 2e5 samples the SE would now be about 1.4e-5.
  - Chain submitted with dependencies: distil s1 into a composite net (16852732), sign step 2 (16852733), iteration 2 (16852734).
- 04:05 **Composite distillation of s1** (16852732, 2080Ti, 1.44 GPU-h: labels 0.38, training 0.97).
  - Setup: one L 48x6 net trained on s1 x Marshall. The training set was 2.0e6 states (24k beta = 0.5 samples of a1 plus all neighbours); 41 epochs, about 8e4 Adam steps.
  - Validation error 0.9% (2.3% on flips). Held-out disagreement with s1: 0.9% at beta 0.5, 0 at beta 1.
  - **ED w_s: composite net 1.1(0.2)e-4 vs s1 (net + hop) 8(2)e-5, paired +3(2.4)e-5.** The composite target is learnable, as angle 2 predicts. The it2 stacked D2 net had w_s 4.9e-4.
- 04:47 **Sign step 2** (`it2_sign.py`, 16852733, 2080Ti, 0.69 GPU-h): composite net N1 + one exact hop on a1.
  - T = -5.20 (smoothed). Half-sample values -4.9 and -6.6; bootstrap 68% interval [-6.2, +1.6]. As in it2, the energy curve is flat.
  - Held-out dH = -3.1(1.4)e-5/site vs N1 alone.
  - **Paired vs ViT: <H>_(a1, N1+hop) - E_ViT = -0.21(0.55)e-5/site, so <H> = -0.503656(22): a tie with the ViT.**
  - **ED w_s = 2(1)e-5**, down from 1.1e-4 for N1 alone (paired -9.0(2.3)e-5). This is the best 6x6 sign so far (ViT 1.3e-4, it2 K3 4.3e-5).
  - Amplitude closeness: std(log a1 - log|psi0|) = 0.052 vs 0.048 for the ViT. The iteration-1 amplitude step moved slightly away from the exact amplitude.
  - Iteration 2 (16852734) is running: 3 steps, N 8192 -> 16384, certificate at the end.
- 06:51 **Iteration 2 done** (16852734, 2080Ti, 2.07 GPU-h): **0 of 3 steps accepted.** The best candidates were -0.6(1.3)e-6, -1.3(2.3)e-6 and -0.7(2.2)e-6/site.
  - Gradient norm per step: 4.5 -> 1.3 -> 0.9. Near the ViT, at fixed (now nearly exact) sign, the amplitude has no resolvable descent at N = 8-16k.
  - Certificate at the end, on the guide (a2 = a1, s2): <H>_G - E_ViT = -0.82(0.43)e-5/site (1.9 sigma).
  - The independent estimate on the same guide from the sign step was -0.21(0.55)e-5. **The combined value is -0.59(0.34)e-5 (1.7 sigma).**

## VERDICT (2026-10-07 07:00): STOP RULE TRIGGERED
After 2 iterations the cumulative <H> gain over the ViT is -0.6(0.3)e-5/site, which is < 2 sigma, so the run stopped as agreed.
No iteration 3 and no FN referee were run. Following the oracle result, FN at the ViT amplitude ties even with exact signs.

- Final guide G2 = (a1, N1 + hop(a1)): <H> = -0.503660(21)/site, vs ViT -0.503654(21).
- **ED w_s = 2(1)e-5**, the best 6x6 sign so far (ViT 1.3e-4).

Compute: about 9.7 GPU-h for all T3 jobs (2080Ti), of which 6.3 GPU-h were production (it1 2.08, distillation 1.44, sign step 0.69, it2 2.07) and the rest smoke tests and diagnostics. Full A40s were not available. The parallel CPU control tuning used about 20 CPU-h.

Why the 4x4 result did not transfer:
1. **Generalisation.** On 4x4 the update lives on the full basis. On 6x6 (P = 155k >> N = 16k) an SR-type step changes log a 4-11x more on unseen states than on the training states. Small-shift minSR steps raise the energy; only plain-gradient and heavily shifted SR steps generalise.
2. **Noise-dominated curvature.** The edge curvature, especially the negative violating-edge term, is heavy-tailed, so RGN must be damped to nothing.
3. **Little gain available.** Even with the better sign (w_s 2e-5) the ViT amplitude is already near-optimal for E_H at this capacity. Measured gains are <= 1e-5/site per step at 8-16k samples, below the per-step resolution.
   - This matches the oracle (exact signs + ViT amplitude ties the ViT under FN) and the it1/it2 frozen-H_FN results.
4. **Cost.** Local data for the hop sign is about 55 ms/sample on a 2080Ti. One step at N = 16384 takes about 0.7 GPU-h, so more samples per step are not affordable within the ceiling.

What did work on 6x6: composite distillation (angle-2 rule). One net trained on s1 x Marshall had w_s 1.1e-4, and with one hop on a1 it reached w_s 2e-5 (8x better than the stacked D2 route).

Figure `t3_6x6.png`/`.pdf`, table `T3_table.md`, raw outputs `t3/*.json`.
