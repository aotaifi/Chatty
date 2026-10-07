# Why the FN/Krylov loop stalls at the ViT level on 6x6 (J1-J2, J2/J1 = 0.5, periodic)

Date: 2026-10-07. All energies are per site, as dE = (E - E0)/N with E0 = -0.50380965 (exact ED).
Everything below is **exact**: no sampling enters any energy.
- All work is done in the fully symmetric sector (k=0, A1, flip+; 15.8M orbit representatives) on one GPU.
- The exact FN ground state, the Krylov sign step and the variational energies use the same conventions as the CPU exact loop. They agree with it to 7e-15.

Code: `experiments/stall_6x6/`. Log: `PROGRESS.md`. Figure: `stall_6x6.png` / `.pdf`.

## Verdict
**The stall is a projection problem, not an information problem.** The loop loses its gain when the improved amplitude is written back into a network.
- **Information is plentiful.** One exact FN iteration from the symmetric ViT lowers <H> by 4.8e-5. Three iterations already beat RBM+PP, and twelve reach 8e-6 (fig. a).
- **Representation accuracy is what fails.** The FN update δ = log φ_FN - log a is small: rms 0.011 in log-amplitude. It must be stored with a rough (configuration-wise) error well below that size:
  - Rough log-amplitude errors cost ΔE ≈ 0.33 σ² per site (fig. b).
  - At σ = 0.012 that cost equals the whole per-iteration gain.
  - This is exactly the error our L2 projection leaves, which is why the L2-projected loop does not move (fig. a, yellow).
- **Energy-aware projections onto small families capture almost none of the gain:**
  - 44 exactly-optimised symmetric 2/4/6-spin cluster corrections: 0.7% of the gain.
  - 7 one-hop local-energy features: 8%.
  - The full ViT under exact-verified SR: about 1e-6 per step.
- **Signs are not the bottleneck.** A Krylov step gains only 5e-6 at the ViT amplitude.
- **Missing symmetry is a cheap, real part of the gap.** Projecting the ViT onto D4 x spin flip gains 2.4e-5.

Whether the ViT architecture *could* represent φ_FN to the required accuracy is not settled: no optimiser we tried gets there. What is settled:
- The architecture's short-range symmetric extensions have no headroom.
- The amplitude change the loop needs is a smooth-looking but small (0.011 rms), high-dimensional correction. Its error tolerance is set by the gain of one iteration.

## Decomposition of the ViT -> E0 gap (1.56e-4, full-basis complex ViT, VMC -0.503654)

| part | dE | how measured |
|---|---|---|
| missing symmetry (D4 x flip projection) | 2.4e-5 | ψ_P = Σ_g ψ(gx): 1.56e-4 -> **1.319e-4** (E = -0.5036777) |
| sign (one exact Krylov step at the ψ_P amplitude) | 0.5e-5 | 1.319e-4 -> 1.266e-4, w_s 1.1e-4 -> 2.4e-5 |
| amplitude, remaining | 1.27e-4 | (ψ_P amplitude, Krylov sign) |
| of which recoverable with perfect information (exact FN loop from ψ_P) | 1 it: 4.8e-5; 3 it: 8.4e-5; 12 it: 1.18e-4 | <H> of the guide after the step: 7.8e-5 / 4.3e-5 / 8.2e-6 |
| of which captured by the best projection we found | ≤ 6e-6 | one-hop features (1.207e-4); 44 clusters 3e-7; full-ViT SR 1.2e-5 in 12 exact-verified steps (rep-ViT start) |

Sign part, two further views:
- With the exact amplitude, the ViT's own signs give <H> 1.41e-4 and E_FN 0.94e-4.
- With the ViT amplitude, exact signs are *worse* variationally than the ViT's own signs (1.56e-4 vs 1.32e-4 for ψ_P), while E_FN is about the same (1.05e-4 vs 1.02e-4).

So the ViT's amplitude and sign errors partly compensate. An additive split "sign + amplitude" only holds along a stated path, as in the table above.

Earlier DMC oracles are confirmed within about 1σ by exact FN:

| guide | DMC (M=128) | exact FN |
|---|---|---|
| \|ViT\| + ViT sign | -0.503670(8) | -0.5036617 |
| \|ViT\| + exact sign | -0.503682(21) | -0.5036622 |
| \|ψ0\| + ViT sign | -0.503731(15) | -0.5037157 |

## Tests

### Test 0: symmetry (`sym_test0.json`)
| state | dE |
|---|---|
| ViT evaluated on canonical representatives | 2.09e-4 |
| full basis (VMC) | 1.56e-4 |
| ψ_P, own complex phase | **1.32e-4** |

- The projected phase is binary: no cancellation, sin² spread 1.6e-6.
- Projecting only the amplitude (arithmetic or geometric mean) gives 1.41-1.44e-4.
- Lesson: our earlier "asymmetry" metric (mean std of log|ψ| over images, 0.006) hid an energetically relevant asymmetry. Always use the projection.

### Test 1: compression of the ViT
- **Supervised L2 / |w|-edge fits to |ψ0|** (`PROGRESS.md` 13:11): the amplitude error falls (0.049 -> 0.038) but the energy rises (2.34e-4 -> 2.7e-4).
  - At fixed sign, the energy is a signed quadratic form on edge differences of the log-amplitude error.
  - VMC places the ViT's errors in soft modes of that form; pointwise fits move them into stiff ones.
- **Exact-verified SR on the full ViT, exact signs** (`t1_fit_exact_vit.json`, 12 steps, each accepted only if the exact energy drops): 2.343e-4 -> 2.225e-4. Gains are about 1e-6 per step and still decreasing.
  - Gradient steps of 1e-5..1e-8 per parameter (RMS) raise E by up to 1.5e-2, so the trained ViT sits in an extremely stiff landscape.
- **With projection (ψ_P plus symmetric corrections, exact linear method)** (`feat_clusters44.json`, `feat_hop.json`):
  - 44 cluster functions: -3e-7.
  - Alternating with Krylov sign steps: 1.264e-4 -> 1.259e-4.
  - One-hop features: -4.9e-6 / -6.0e-6 (own / Krylov sign).

### Test 2: projected ideal loop (fig. a)
- **Exact loop.**
  - From the rep-ViT (`exact_loop_from_vit.json`): E_FN 1.48e-4 -> 8.6e-5 -> ... -> 6.6e-6 at it 15.
  - From ψ_P (`exact_loop_from_projvit.json`): 1.02e-4 -> 6.5e-5 -> 4.9e-5 -> ... -> 8.8e-6 at it 12.
  - Contraction 0.64, 0.75, 0.78, then slowly rising to 0.87.
  - The first ideal iteration is worth 3.7e-5 in E_FN and 4.8e-5 in <H>, which is 3-4x the 1.5e-5 assumed in the it2 verdict.
- **Projected loops** (projection = exact minimisation of the frozen FN energy, the target whose unrestricted minimum is φ_FN):
  - Clusters: E_FN 1.02e-4 -> 0.94e-4, then stuck; 0.7% and 0.1% of the gain captured.
  - One-hop features: 0.90e-4 after 6 iterations; 7.7%, 2.9%, 1.1%, ... of the gain captured.
  - L2 fit of the ViT to φ_FN (rep-ViT start, `proj_loop_l2.json`): E_FN 1.48e-4 -> 1.39e-4; guide <H> 2.06e-4 vs 1.07e-4 for the exact step.
- **The loop stalls where the projection error equals the per-iteration gain**, not at a capacity floor found in test 1.

### Test 3: which capacity is missing (`interp_fn_update.json`, fig. b)
- **The update is not a short-range spin correlation:** 44 symmetric clusters capture 0.7% of it.
- **It is only weakly a function of local energies:** one hop captures 8% at first.
- **It is nearly linear and can be extrapolated:**
  - Partial steps a·exp(tδ) gain in proportion to t: <H> 1.32 / 1.18 / 1.05 / 0.84e-4 at t = 0, 0.25, 0.5, 1.
  - Overshoot t = 1.25 is still better (7.6e-5).
- **Error tolerance.** Rough (iid per-orbit) log errors of rms σ cost:

  | σ | cost per site |
  |---|---|
  | 0.003 | 2.6e-6 |
  | 0.01 | 3.5e-5 |
  | 0.03 | 3.1e-4 |

  The cost is the same on φ_FN and on ψ_P, i.e. ΔE ≈ 0.33 σ².

  Keeping 90% of one iteration's gain therefore needs a projection error ≲ 0.004 rms, while the update itself has rms 0.011.

## What this implies for larger sizes
1. **Symmetrise first.** Project the ViT over the point group and spin flip before any loop. This is free energy: 2.4e-5 here, and 1.7e-4 for the public 10x10 ViT.
2. **The loop is not information-limited.** Exact FN iterations stay strong at the ViT level, with first-iteration contraction about 0.6.
   - A sampled loop has to deliver the per-iteration update with rough error σ ≲ sqrt(ΔE_iter / 0.33).
   - Here that is about 0.004 for ΔE_iter ≈ 5e-5.
   - Both ΔE_iter and the 0.33 coefficient are per-site (intensive), so the same tolerance applies at 8x8 and 10x10.
   - With histogram or walker estimates, the relative error per configuration is about 1/sqrt(N_samples p(x)). This, not FN information, is what limits sampled loops: compare the 4x4 1/N floor.
3. **Do not project with pointwise losses.** Use the frozen FN energy (or the variational energy) as the projection objective, as the energy-weighted metric demands.
4. **Exploit linearity and overshoot.** Anderson / extrapolated network updates are well-founded: t = 1.25 still lowers the energy.
5. **Optimizer stiffness is the practical blocker.** SR on the ViT gains only about 1e-6 per exact-verified step. Small symmetric add-on factors (clusters, one-hop features) cannot carry the update.
   - A larger or differently structured correction is needed: e.g. a symmetric network factor trained on the frozen FN energy with N >> P.
   - It must reach about 0.004 rms projection error before the loop can beat the ViT + projection baseline at 6x6.

## Compute
- About 4.9 GPU-h: RTX 2080 Ti 4.3 h, A40 0.63 h. Many of the 2080 Ti jobs were short memory-debugging runs; see `PROGRESS.md`.
- 1.2 CPU-h of CPU jobs (CSR build). The GPU jobs also allocated 6 cores each (about 29 core-h).
- Large data and runs: ws1 `/project/theorie/a/A.Otaifi/chatty_stall6/` (csr 7.2 GB, data, runs).
