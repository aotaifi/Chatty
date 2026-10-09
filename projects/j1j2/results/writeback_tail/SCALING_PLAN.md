# Taking the written-back FN/Krylov loop to 8x8: plan (2026-10-09, no runs yet)

**Goal.** 8x8, J2/J1 = 0.5: start from our symmetrised 8x8 ViT (E = -0.498823 per site). Reach an energy below
RBM+PP (-0.498886), i.e. gain >= 6.3e-5/site, with a referee error bar <= 1e-5/site.
**What 6x6 established** (README sections 6b, 6c):
- One realistic write-back keeps 42-76% of an ideal FN iteration.
- The loop passes RBM+PP at iteration 6.
- The scheme needs features of the current guide one hop out, and the semi-implicit target needs the same.

## The cost problem: guide evaluations, not training
Counted distinct configurations (`hopcount.json`; J1-J2 exchange hops):

| lattice | one-hop | two-hop | three-hop (estimate) |
|---|---|---|---|
| 6x6 | 77 | 2.8e3 | ~1e5 |
| 8x8 | 135 | 8.9e3 | ~4e5 |

**Throughput assumption, to be measured first.**
- 8x8 ViT single image (4 patch translations inside): ~1.1e4 evaluations/s per A40, from AMP6 (1e5 one-hop local
  energies per 20 A40-min).
- H100: ~2.5x that (unmeasured).
- The fully symmetrised guide costs 16x, so it is used only where stated.

## Options to avoid the k-hop recursion
Per training sample at 8x8, in base-ViT evaluations:

| option | how | cost per sample | k-dependence | main risk |
|---|---|---|---|---|
| A. current-guide features (as at 6x6) | f_k sees one-hop features of guide_k = stack of all previous nets | 135^k (1.8e4 at k = 2, 2.5e6 at k = 3) | exponential | infeasible beyond k = 2 |
| B. frozen-base features (itfit F) | every f_k sees one-hop features of the base only; the target of iteration k is the semi-implicit step of the CURRENT guide, so needs guide_k on N1(x) = base on N2(x) | 8.9e3 (pointwise loss); ~8e4 (edge loss, needs targets at neighbours) | none | per-iteration capture decays (6x6: 0.95 / 0.80 / 0.69) |
| C. periodic re-basing | every m iterations distil the stack into one fresh symmetric net (warm-started ViT + residual), which becomes the new base; then as A/B with k <= m | 135 (pointwise) / 1.2e3 (edge loss), plus distillation (stack values at pool configurations: ~135 each) | none | distillation is a write-back of the accumulated correction into a spin-only net, i.e. the original failure mode (17-31% kept at 6x6) |
| D. caching along MCMC chains | consecutive samples are one hop apart, so N2 sets overlap; cache guide values per step | B or C divided by the hit rate (unmeasured; plausibly 2-5x) | none | memory; complements B/C |

## GPU-h estimate (per loop iteration; pool of 5e4 training configurations, reused for ~20k Adam steps)
Base = translation-only ViT; symmetrisation is applied in the referee only.
- **B (pointwise):** 5e4 x 9e3 = 4.5e8 evaluations, about 11 A40-h or ~4.5 H100-h. Six iterations: ~27 H100-h.
- **C (edge loss, m = 2):** 5e4 x 1.2e3 = 6e7 evaluations plus distillation, ~1 H100-h. Six iterations: ~6 H100-h.
- **Referee:**
  - VMC <H> of the final guide at SE 1e-5/site: ~3e4 samples x one-hop guide values, ~5 H100-h for B, less for C.
  - Calibrated FN-DMC (`results/fn_calibration_6x6`): tau_max 0.025, M = 512, beta window 0.8-2.4, beta-time-averaged
    mean, >= 500 populations for 3e-6. Each walker step needs the guide on N1. A full protocol costs ~5e7
    walker-steps x 135 = 7e9 evaluations (B: x135 more), i.e. ~70-200 H100-h.
  - Plan: run FN-DMC only for the final guide with ~100 populations (SE ~7e-6, bias <= 0.2e-6 at M = 512), ~15-40
    H100-h. Run VMC <H> at every iteration.
- **Total:** C ~6 + 5 + 30 ~ 40 H100-h; B ~27 + 5 + 30 ~ 60 H100-h. Paderborn H100s if back (the nodes were drained
  on 2026-10-08); otherwise this is too large for the ws1 A40 queue at our fair-share.

## Order of work (each step decides the next)
1. **6x6 pre-tests (exact, < 2 A40-h).**
   - (a) Distillation fidelity: distil the 3- and 6-iteration guides of `wtLOOP8` into one symmetric net (warm ViT +
     FEAT-of-base residual) and measure the exact <H> lost. C is viable only if >= 70% of the accumulated gain survives.
   - (b) Option B with a pointwise vs an edge loss (frozen-base features, current-guide target). This decides whether
     neighbour targets, the 8e4-per-sample cost, are needed.
2. **8x8 measurements (< 1 H100-h).**
   - Symmetrised and translation-only ViT throughput on H100.
   - N2-cache hit rate along an MCMC chain.
   - Variance of the realistic semi-implicit target and of VMC <H> per sample.
3. **8x8 one-iteration test** (pre-registered as Step 1/2 here):
   - capture of one iteration from the ViT, judged by the VMC <H> gain with error bar;
   - same-sample fixed-sign VMC control;
   - pass if >= 2e-5/site gain and >= 1.5x the control.
4. **8x8 loop**, 4-6 iterations with the option chosen in 1-2. Final guide by the calibrated FN-DMC.
   Success = below -0.498886 with a 2-sigma margin.

## Risks
- The per-iteration gain at 8x8 is unknown. The 6x6 ideal FN iteration gains 4.8e-5/site from the ViT; at 8x8 the
  ViT is closer to converged, and the gain may be smaller. Step 3 measures it.
- Sign: Krylov sign steps need one hop of the guide; their cost is in the referee budget.
- The input-standardisation detail of section 6c (guide-only constants) is already built into `wt_loop.py`.
