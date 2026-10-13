# Deliverable C, round 1 (2026-10-13): ViT + Lanczos + variance extrapolation + FN-DMC at 8x8

Pre-registration and amendments 1-3: `PREREG.md` (each amendment was written before the run it governs).
Literature: `LITERATURE.md`. Costed scale-up plan: `PLAN.md`. Code: `experiments/deliverable_C/`.
Paderborn drafts (not submitted): `cluster/paderborn/deliverable_C/`.

Terms (per-site energies in units of J1, periodic 8x8, J2/J1 = 0.5):
- **psi_P**: our 8x8 ViT projected onto the D4 x spin-flip symmetry, with its own sign made real (binarised).
- **Lanczos step**: psi_1 = psi_P (1 + alpha (E_L - E_ref)), E_L = (H psi_P)/psi_P the local energy. Fixed numbers
  alpha = -0.160951, E_ref = -31.92677568. One step = one more application of H ("one hop", ~145 configurations).
- **Direct VMC**: Metropolis sampling of |psi_1|^2 and the mean of its exact local energy: a variational upper bound.
- **Variance**: sigma^2 = <H^2> - <H>^2 of a state, per site; zero for an exact eigenstate.
- **Variance extrapolation**: straight line through (sigma^2_p, E_p) for p = 0, 1 Lanczos steps, read at sigma^2 = 0
  (Hu et al. 2013). An estimate of the ground-state energy E0, NOT a bound.
- **FN-DMC**: lattice fixed-node projector Monte Carlo with a guide state; its energy is a rigorous upper bound on E0 and
  lies below the guide's own energy. "Calibrated protocol": M = 512 walkers per population, beta-time average over
  the imaginary-time window [0.8, 2.4] (results/fn_calibration_6x6).
- **SE**: standard error from independent chains or populations.

## Results table (8x8)
| quantity | E/N | type | source | new? |
|---|---|---|---|---|
| psi_P (projected ViT) | -0.4988474(106) | bound | results/loop_8x8 FU1a, re-derived from raw data | ours, not a record |
| **psi_1 = psi_P + one Lanczos step** | **-0.4989687(141)**; conservative late-time -0.498948(23) | bound | review REVIEW_lanczos8x8, re-derived | **yes: lowest 8x8 variational energy in the literature survey** |
| FN-DMC, guide psi_P | FN_RESULT | bound | this round | first NQS-guided FN energy for J1-J2 |
| variance extrapolation p = 0, 1 | **-0.498988(18)** (stat 1.7e-5, sys 8e-6) | estimate of E0 | this round, `extrap_p01.json` | yes |
| RBM+PP, Nomura-Imada 2021 | -0.498886(1) | bound | PRX 11, 031034 | |
| Hu et al. 2013, VMC p = 2 | -0.49886(1) | bound | PRB 88, 060402(R) | |
| Hu et al. 2013, variance extrapolation | -0.49906(1) | estimate | same | |
| DMRG torus, truncation-error extrapolation | -0.4992(1) | estimate | Gong et al. 2014 | |
| HQT transformer (arXiv:2607.00398) | -0.5001(1) | claimed bound, below every E0 estimate | not credible as a bound | |

## What was done (steps, in plain words)
1. **Literature check** (`LITERATURE.md`). Read Chen et al. 2022 (6x6 and 10x10 only), Wang-He-Lu 2026 (6x6 p <= 2,
   10x10 p = 1, CNN, no extrapolation), Roth et al. 2023 (Lanczos only on the triangular lattice), Chen-Heyl 2024
   (Lanczos inside their 10x10 zero-variance extrapolation; 8x8 not tabulated), Hu et al. 2013, Nomura-Imada.
   Searches for NQS-guided FN/GFMC on J1-J2 and for NQS + Lanczos at 8x8 found nothing. The method is known; our 8x8
   numbers are new.
2. **Variance extrapolation from existing raw data** (no new sampling). Inputs: 98304 VMC samples of psi_P (2048 chains)
   and the 10240 direct samples of psi_1 (3584 chains) of the review. Variances per site: sigma^2_0 = 7.08(1.02)e-4,
   sigma^2_1 = 9.81(1.54)e-5 (ratio 0.14). Line through both points -> **E0 = -0.498988**, chain-bootstrap SE 1.7e-5.
   - Calibration of the method on exact 6x6 data (`p2rep_6x6.json`, results/lanczos_baseline_6x6): from psi_P the same
     two-point line lands 2.2e-6 above the exact E0 (ratio 0.18 there); from the worse ViT(rep) start 7.7e-6 above.
     Systematic +-8e-6 (pre-registered). Panel (b) of the figure.
   - Heavy tails: 5 of 98304 psi_P samples carry 26% of sigma^2_0. Removing the 5 / 10 / 30 most extreme samples of
     each set moves the extrapolation by +0.3 / +0.5 / +1.1e-5 (E and sigma^2 move together). Using only the late
     psi_1 samples (sweep >= 12, 5120 samples) gives -0.498966; the latest 2048 alone -0.498947 (the review's
     conservative subset). A second estimator of sigma^2_0, <(H^2 psi)/psi> - <H>^2 (15% larger on the referee
     samples), moves the result by 1.2e-6.
   - To reach Hu's -0.49906 the psi_1 variance would have to be 4.7x larger than measured with sigma^2_0 fixed.
   - Pre-registered comparison: |-0.498988 - (-0.49906)| = 7.2e-5 = **3.4 combined sigma: in tension with Hu's
     extrapolation**; 2.1 sigma from the DMRG torus value -0.4992(1). Our extrapolation starts from a state 9e-5 below
     Hu's best p = 2 state, so it extrapolates over a ~5x shorter distance. Hu's extrapolation also lies below the
     exact/best values at 6x6 (-0.50382(1) vs exact -0.503810) and 10x10 (-0.49781(2) vs Chen-Heyl ZVE -0.497715(9)).
3. **FN-DMC with the psi_P guide** (new code `dc_fn.py`; same update rules as the calibrated 6x6 referee, the projected
   ViT is evaluated once per new walker configuration and cached).
   - Gate V1 on 4x4 with exact FN energies (two deliberately bad toy guides): the code reproduces the exact FN energy
     after extrapolating the population-control bias in 1/M (a = +3.0(1.9)e-5 and +2.4(3.4)e-5 per site; bias b/M with
     b proportional to the guide's energy variance). For psi_P at 8x8 (74x smaller variance than the toy guide) the
     predicted bias at M = 512 is ~2.5e-7 per site. The first gate run failed by the letter; amendments 1-2 record the
     diagnosis (projection transient, then 1/M population bias) before the 8x8 run.
   - 8x8 run: FN_STEPS
4. **p = 2 (two Lanczos steps)**: pilot only (pre-registered). The local energy of psi_2 needs psi_P on the three-hop
   ball, 4.44e5 configurations per sample (86 s on an A40 slice). 64 samples: the optimal 3x3 Ritz coefficients are not
   estimable from so few samples (the tail decides them), so psi_2 is fixed to the repeated step psi_P f^2
   (amendment 3). Exact 6x6: the repeated step keeps 78% of the optimal p = 2 gain (1.23e-5 vs 8.6e-6 vs p = 1 2.55e-5;
   a Lanczos step on psi_1 keeps 86%); with it, the three-point extrapolation at 6x6 is within 1.5e-6 of E0.
   The 8x8 paired SE 5e-6 needs ~1.9e4 samples (~590 slice-h): Paderborn draft `p2_prod_h100.sbatch`. Expected gain
   ~1-2e-5 per site, i.e. the p = 2 number would add a third extrapolation point more than a new bound.
5. **10x10 preparation**: the Hugging Face checkpoint uses the same ViT code as ours; our 10x10 instance has 434,760
   parameters, exactly the card's number, and the symm_trxy_ising revision is the same projection (sum of amplitudes
   over 32 permutations x spin flip) as our psi_P. Measured cost: 2.2e3 symmetric evaluations/s on an RTX 2080 Ti
   (8x8 psi_P: 4.6e3/s on the same card). Download of the weights not done (needs the user's OK). Loader and VMC gate:
   `dc_vit10.py`.

![figure](fig_deliverableC_8x8.png)

Figure: (a) 8x8 energy against variance: psi_P (p = 0) and psi_1 (p = 1) with bootstrap errors, the two-point line
and its sigma^2 = 0 value; bands: RBM+PP bound and Hu et al.'s extrapolation; FN-DMC with the psi_P guide.
(b) exact 6x6 calibration of the same procedure: optimal p = 0, 1, 2, the repeated-step p = 2 state, and the exact E0.

## What is new and what is not
- Not new: the method (NQS + Lanczos: Chen et al. 2022, Wang-He-Lu 2026; variance extrapolation: Hu et al. 2013;
  lattice FN-GFMC: ten Haaf et al. 1995).
- New: (i) the 8x8 NQS + one Lanczos step bound -0.498969(14), 8.3e-5 below RBM+PP (conservative late-time value -0.498948(23): 6.2e-5 below); (ii) an 8x8 E0 estimate
  -0.498988(18) that disagrees with Hu's -0.49906(1) at 3.4 sigma; (iii) the first FN-DMC energy with a deep-NQS guide
  for this model (FN_NEWNESS).

## Open problems (each with its cause and the test that settles it)
- The extrapolation rests on two points. A third (p = 2) needs ~1e10 projected evaluations at 8x8 (Paderborn).
  Risk: if the curvature at 8x8 differs from 6x6, the two-point value can be off by more than 8e-6; the p = 2 point
  settles it (3-point linear vs quadratic, pre-registered).
- Heavy tails: variances are carried by <0.1% of the samples. The bootstrap SE assumes the tail is sampled; the trim
  test shows the extrapolation is insensitive (E and sigma^2 move together), but an unseen tail in sigma^2_1 alone
  would lower the estimate. Monitor: the p = 2 run measures sigma^2_1 again on fresh samples.
- FN-DMC with the psi_1 guide (the candidate for the best 8x8 bound) costs ~70x the psi_P run: Paderborn draft.
- 10x10: waiting for permission to download the public checkpoint; Lanczos at 10x10 is Paderborn-scale (PLAN.md).

## Cost (ws1, this round)
COST_TABLE
