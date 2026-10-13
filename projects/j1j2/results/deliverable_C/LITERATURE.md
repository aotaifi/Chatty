# Deliverable C, step 1: what is published for "NQS (or VMC) + Lanczos steps (+ variance extrapolation, + FN/GFMC)"

Model: spin-1/2 J1-J2 Heisenberg model on the L x L square torus, J2/J1 = 0.5. Energies are per site in units of J1.
Compiled 2026-10-13, building on `results/literature/J1J2_BENCHMARKS_COST.md` (the survey; numbers there were read from
the papers). New checks for this file are marked "checked 2026-10-13" (full text read for Chen 2022, Wang-He-Lu, Roth;
abstract/HTML for the rest).

Terms:
- **Lanczos step (p steps).** From a variational state psi, form psi_p = P_p(H) psi with a polynomial of degree p whose
  coefficients minimise the energy (Ritz value in span{psi, H psi, ..., H^p psi}). Each step costs one more application of H,
  i.e. one more "hop" (all ~145 configurations reachable by one spin exchange at 8x8). The energy of psi_p is still a
  variational upper bound when it is measured by sampling |psi_p|^2 with fixed coefficients.
- **Variance extrapolation.** Fit the energies E_p against the energy variances sigma^2_p = <H^2> - <H>^2 of the same states
  and read off the energy at sigma^2 = 0 (exact eigenstate). Not a bound; an estimate of E0 whose error is set by the
  curvature of E(sigma^2) and by the statistical error of the variances.
- **Zero-variance extrapolation (ZVE).** Same idea across different network sizes instead of Lanczos steps.
- **FN-GFMC / FN-DMC.** Lattice fixed-node projector Monte Carlo (ten Haaf et al. 1995): the guide's signs define an
  effective sign-free Hamiltonian whose ground energy is a rigorous upper bound to E0 and lies below the guide's own
  variational energy.

## 1. Who did what (J2/J1 = 0.5, square torus)

| Work | Base state | Lanczos steps | Sizes | Extrapolation | Key numbers (E/N) |
|---|---|---|---|---|---|
| Hu, Becca, Parola, Sorella, PRB 88, 060402(R) (2013), arXiv:1304.2630 | Gutzwiller-projected fermionic (Z2 spin liquid) + Jastrow, few parameters | p = 0, 1, 2 | 6x6 ... 18x18 | variance, linear in sigma^2 | 6x6 p2 -0.503571(3), EXT -0.50382(1) (exact -0.503810); **8x8 p0/p1/p2 -0.49656 / -0.49855 / -0.49886(1), EXT -0.49906(1)**; 10x10 p2 -0.497549(2), EXT -0.49781(2); 14x14 p1 -0.49638(1), EXT -0.49722(2); 18x18 EXT -0.49717(2) |
| Chen, Hendry, Weinberg, Feiguin, NeurIPS 2022, arXiv:2206.14307 (checked 2026-10-13) | symmetry-projected RBM | p = 0, 1, 2 | **6x6 and 10x10 only** (8x8 only at J2 = 0, Table 3) | mentioned, not done | 6x6 p0/p1/p2 -0.50364(2) / -0.50376(3) / -0.50378(4); 10x10 p0 -0.49580(2), p1 -0.4968(4) |
| Wang, He, Lu, PRB 113, 085120 (2026), arXiv:2502.01264 (checked 2026-10-13) | CNN ("aCNN", sign + amplitude nets), Lanczos states re-learned by supervised learning + VMC | 6x6 p = 1, 2; 10x10 p = 1 | 4x4 (J2 = 0.55), **6x6, 10x10** | none | 6x6 p2 (VMC-Lanczos) -0.503178(4); 10x10 p1 -0.496102(7). Weaker than the ViT/ResNet records. No 8x8. |
| Chen & Heyl, Nat. Phys. 20, 1476 (2024), arXiv:2302.01941 | deep ResNet, MinSR | one Lanczos step inside the ZVE | 10x10 (energies), 8x8 / 12x12 / ... only gaps and figures | ZVE (2 network sizes + Lanczos) | 10x10 best variational -0.4976921(4) (no Lanczos), ZVE -0.497715(9); 16x16 -0.4967163(8). No tabulated 8x8 energy |
| Roth, Szabo, MacDonald, PRB 108, 054410 (2023), arXiv:2211.07749 (checked 2026-10-13) | GCNN | one Lanczos step, **triangular lattice only** (Tables III-IV) | square: 10x10, 12x12, 16x16 without Lanczos | none | square 12x12 -0.496769(9), 16x16 -0.496509(6) (plain VMC) |
| Nomura & Imada, PRX 11, 031034 (2021), arXiv:2005.14142 | RBM + pair product, quantum-number projections | none (mVMC) | 6x6 ... 18x18 | none for J2 = 0.5 energies | 6x6 -0.503765(1); **8x8 -0.498886(1)**; 10x10 -0.497629(1); 12x12 -0.496791(4) (4x4-sublattice PP) |
| Rende, Viteritti, Bardone, Becca, Goldt, Commun. Phys. (2024), arXiv:2310.05715; HF checkpoint nqs-models/j1j2_square_10x10_05 | ViT, SRt | none | 10x10 | none | -0.497634(1) (paper); checkpoint card -0.497676335 (no error bar) |
| Lin, He, Lu, Chin. Phys. B 31, 080203 (2022); Lin et al., PRB 109, 235133 (2024), arXiv:2406.12207 | RBM- and PEPS-guided **GFMC** | - | 6x6, 8x8, 10x10 | D-extrapolation | 10x10 PEPS(D=5)-GFMC -0.4957(2): ~2e-3 above the NQS bounds. Not competitive at 8x8/10x10 |
| Guo et al., arXiv:2607.00398 (HQT) | transformer | - | 8x8, 10x10 | - | 8x8 -0.5001(1), 10x10 -0.49782(3): below every extrapolated E0; not credible as variational numbers, but must be cited and discussed |

Searches done 2026-10-13 (web, arXiv, Google-Scholar-type queries "neural quantum state guided fixed-node GFMC J1-J2",
"J1-J2 8x8 neural network Lanczos"): no NQS + Lanczos energy at 8x8, no fixed-node/GFMC calculation with a deep-NQS
guide for J1-J2 at any size, no NQS energy with error bar at 14x14. (NQS + Lanczos at 8x8 exists for the Hubbard model,
neural backflow, arXiv:2510.26906 / 2609.26544; not our model.)

## 2. What is known as a method (so: not new)
- NQS + Lanczos steps: Chen et al. 2022; Wang-He-Lu 2026; Roth et al. 2023 (triangular); Chen-Heyl 2024 (inside ZVE).
- Lanczos steps + variance extrapolation: Sorella 2001 (PRB 64, 024512), Hu et al. 2013 (the J1-J2 reference),
  Becca-Sorella textbook.
- Lattice FN-GFMC with variational (Gutzwiller, Lanczos-improved) guides is a standard tool of the Sorella/Becca school
  (ten Haaf et al. 1995; Becca-Sorella textbook). For J1-J2 at 0.5 we found only RBM- and PEPS-guided GFMC (Lin et al.).
  NQS -> FN-GFMC exists for Hubbard / t-J (not J1-J2).
- Our method is therefore the published baseline: ViT (projected) + 1-2 Lanczos steps + variance extrapolation + a
  calibrated lattice FN-DMC referee. The novelty can only be in the numbers, the sizes and the error control.

## 3. Which of our numbers would be new (and the exact bar for each)

| Number | Status | Bar / comparison | New? |
|---|---|---|---|
| 8x8 psi_P + 1 Lanczos step, direct VMC -0.498969(14) pooled, conservative late-time -0.498948(23) (reviewed, `results/loop_8x8/REVIEW_lanczos8x8.md`) | done | best published 8x8 bounds: RBM+PP -0.498886(1), Hu p2 -0.49886(1) | **yes**: lowest published-class 8x8 variational energy (first NQS + Lanczos at 8x8). HQT -0.5001 must be discussed (below every E0 estimate) |
| 8x8 p = 2 | to compute (expensive, see PREREG) | must beat p = 1 by a resolved paired gain | yes if resolved |
| 8x8 variance-extrapolated E0 from p = 0, 1 (2) | this round | Hu EXT -0.49906(1), DMRG torus -0.4992(1) | **yes**: a second, independent extrapolation from a far better start (our p = 1 is 9e-5 below Hu's p = 2, so the extrapolation distance is ~5x shorter). It bears on the Hu-vs-DMRG disagreement |
| 8x8 FN-DMC with the NQS guide psi_P | this round | first NQS-guided FN energy for J1-J2; must be below psi_P | yes (method known, number new); expected above psi_1, so not a new best bound |
| 8x8 FN-DMC with the psi_1 guide | Paderborn | would be the best 8x8 upper bound if below -0.498969 | yes if done |
| 10x10 HF ViT + 1 Lanczos | needs the checkpoint (download not yet approved) and Paderborn-scale sampling | must beat -0.4976921(4) (Chen-Heyl) to be a record; else only a cross-check of the ZVE | maybe |
| 12x12 / 14x14 NQS + Lanczos (+ FN) | plan only | 12x12: RBM+PP -0.496791(4), GCNN -0.496769(9); 14x14: only Hu p1 -0.49638(1), EXT -0.49722(2) | **yes** (first deep-NQS energies with error bars there) |

Caveat that a referee will raise and that we must answer with numbers: a fixed number of Lanczos steps is not size
consistent (Chen et al. 2022 note this). Our exact scaling test (results/lanczos_vs_krylov_scaling, local_krylov F1)
found no shrinkage of the one-step gain fraction up to 36 sites and an estimated halving size N_1/2 of order 1e3-1e4
sites for ViT-quality guides, so the step should still carry most of its gain at 12x12-14x14 (144-196 sites).
