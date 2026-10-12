# RBM+PP as the guide of the FN/Krylov loop (exact 6x6, J2/J1 = 0.5, PBC)

Pre-registration, written 2026-10-12 22:10, after the correctness tests and two 20-step smoke runs, before any
decisive run. Results go to `results/rbmpp_6x6/`. Exact ground state E0/N = -0.50380965.

## Why
Two pitches (lead brief 2026-10-12):
- **(A) add-on:** start the loop from the best guide on the market, a converged RBM+PP (Nomura & Imada, PRX 11,
  031034 (2021), arXiv:2005.14142; published 6x6: -0.503765(1), i.e. 4.5e-5/site above E0, relative error 9e-5).
- **(B) method:** start from a moderate guide (an early RBM+PP checkpoint, and the PP alone) and let the loop do the
  work.

## Terms
- **RBM+PP:** Psi(sigma) = phi_RBM(sigma) x psi_PP(sigma). phi_RBM = prod_k 2 cosh(b_k + sum_i W_ik sigma_i) is a
  restricted Boltzmann machine (16 hidden units, complex b, W). psi_PP is a pair-product (geminal) state
  P_G (sum_ij f_ij c+_{i up} c+_{j dn})^{N/2}|0> mapped to spins (P_G removes doubly occupied sites); its amplitude
  is a sign times det f[up sites, down sites], with f a real 36 x 36 matrix.
- **Projection:** the sum psi_P(sigma) = sum_g Psi(g sigma) over a set of lattice symmetries g and the global spin
  flip. It removes the wrong-symmetry parts. 'simple' = the published one (36 translations x {identity, 90-degree
  rotation} x spin flip, 144 terms); 'full' = all 288 space-group elements x spin flip (576 terms), the exact
  projector onto our sector (momentum 0, A1, spin-flip even).
- **SR (stochastic reconfiguration):** the natural-gradient / imaginary-time optimiser of VMC: each step solves
  S d = -lr F, with S the covariance of the log-derivatives O_k = d log psi / d theta_k and F the covariance of O_k
  with the local energy. Not Adam.
- **VMC energy:** <H> estimated from Metropolis samples of |psi|^2 (local energy E_loc averaged).
- **dE/site:** E/N - E0/N (absolute). **Relative error:** dE/site / |E0/N|.

## Route: JAX implementation, not mVMC (decision and why)
- The loop needs the guide evaluated on all 15.8M orbit representatives in our own code (step 2). With mVMC we would
  have to re-implement its conventions anyway (Pfaffian/fermion ordering, RBM layout, projection weights) and a
  convention mismatch would only show up as an energy disagreement. Writing the ansatz once in JAX gives one code
  path for training, sampling and the exact table.
- GPU: the 6x6 problem is small (2480 real parameters); one SR step with 2048 samples takes ~1.5 s on an
  RTX 2080 Ti (measured, smoke `rpSMr`). mVMC on CPUs would also work; it is not slower in principle, but the
  conventions risk above decides.
- Whether the mVMC release contains the RBM correlator was not checked; it does not affect the decision.

Code (this directory):
- `rbmpp_lib.py`: ansatz exactly as published (16 complex hidden units, no symmetry imposed on W; f real, full
  36 x 36, no sublattice), projections, local energy with fast determinant updates, Metropolis sampler (exchange
  moves on anti-aligned NN/NNN bonds, Hastings factor, single-precision acceptance).
- `rbmpp_train.py`: SR training, history and parameter checkpoints every 50 steps, final VMC energy with error bars.
- `rbmpp_table.py`: full-projection table on all reps, `sym_tables_<tag>.npz` (lP float64 log|psi|, sP int8 sign,
  the format of `chatty_stall6/data/sym_tables.npz`), exact <H>, w_s, FN energy, guide + 1..3 Lanczos steps.
- `rbmpp_test.py`: correctness tests. **All passed** (2026-10-12, ws1 CPU): same space group and bonds as
  `st6_sector`; fast-update local energy = brute force (max |dE| 4e-9, 4x4 and 6x6, all projections); symmetric f
  gives an exact singlet (<S^2> = 8e-15, checks the fermion sign); 4x4 exact identity
  sum |psi|^2 E_loc / sum |psi|^2 = <H> to 1e-10; 4x4 ED E0/N = -0.52862021; Metropolis <H> agrees with the
  exact value (simple projection; 1.1 sigma).

## Training plan (decided now)
All runs: 2048 Metropolis chains, one sample per chain per SR step after a sweep of 36 exchange proposals; SR with
relative diagonal shift 1e-3 in the diagonally rescaled basis, step cap sqrt(d S d) <= 0.3; training with the
published 'simple' projection; parameter checkpoints every 50 steps.
- **T1 (PP alone):** f random as published (uniform in [-fmax, fmax], fmax ~ R^-2), RBM off; lr 0.05 -> 0.01
  (cosine), 3000 steps.
- **T2 (RBM+PP, staged):** f from T1, RBM fresh random (Re, Im uniform in [-0.05, 0.05], as published);
  lr 0.05 -> 0.005, 6000 steps.
- **T3 (RBM+PP, from scratch = published recipe):** f and RBM random; same lr schedule, 6000 steps.
- More steps or samples only if the convergence check below fails, recorded as an amendment.
- Final energies: VMC with >= 4e5 samples, error from 2048 independent chain means, for the 'simple' projection
  (comparison with the published number) and the 'full' projection (the state that goes into the table).
- Cost: wall-clock of every step and of the whole run, converted to GPU-h (GPU type recorded).

**Convergence evidence (required for the (A) guide):** the mean energy of the last 500 steps and of the 500 steps
before agree within 2e-6/site (or within 2 sigma of their difference), E_loc variance flat over the same windows,
plus the high-statistics final VMC energy.

**Target (pre-registered):** the converged RBM+PP reaches E/N <= -0.503745 with the published projection (i.e. the
published -0.503765 within 2e-5/site). If not reached: report the cause (optimizer step/shift, samples, steps,
local minimum, seed), with the evidence for it; the run is "inconclusive (cause: X)", not "RBM+PP fails".
The guide for pitch (A) is the lower of T2/T3 (final VMC energy, 'simple' projection).

**Moderate guides for pitch (B), selection rule fixed now:**
- B1: the first saved checkpoint of the chosen RBM+PP run whose 50-step running mean energy is below
  dE/site = 5.0e-4 (relative error ~1e-3);
- B2: the first saved checkpoint below dE/site = 1.5e-4 (relative error 3e-4; this is the ViT start, 1.32e-4 / site,
  so B2 vs the ViT compares two guides of about the same energy);
- PP: the final T1 state (PP alone).
The exact energy of each table (full projection) is what counts; the running-mean rule only selects the file.

## Exact table check (step 2)
For every table: the exact <H>/N of the fully projected complex state must agree with its VMC energy ('full'
projection, same parameters) within 3 sigma of the VMC error. The real guide written to the table (best real
combination of Re psi and Im psi) can only be lower or equal in energy; both numbers are reported.

## Loops (step 3, code unchanged except the parameterisation)
- `writeback_tail/wt_loop.py` (table loop, 8 iterations, as wtLOOP8: 20k Adam steps x 256 x 8 bonds, lr 3e-3), with
  `lanczos_eval` (each iteration's guide + 1 and + 2 exact Lanczos steps; new option `lanczos_p`, default 1 =
  old behaviour) and `ws` (w_s); the guide + 0..3 Lanczos steps is recorded once at the start.
- `compress_6x6/cx_chop2.py` (network loop, round 4), 8 iterations, seeds 0 and 1. `cx_lib.py` now reads the
  table-loop reference from `CX_LOOP8` and refuses to run a non-ViT guide without it; for a non-ViT guide H_START
  and the Lanczos references are computed exactly from the guide (the ViT constants are used only for the ViT).
  w_s of the network loop is computed afterwards from its saved signs.
- Per iteration: <H>, E_FN (exact FN energy), w_s (wrong-sign weight against the exact ground state), guide + 1
  and + 2 exact Lanczos steps.
- Order: (A) converged guide table loop + 2 network-loop seeds; (B2) the same; (PP / B1) table loop only if the
  budget allows.

Readings (pre-registered):
- Gain: fraction of the guide's error removed after 8 iterations, (H_guide - H_8) / H_guide, table and network.
- Equal-hop comparison as in round 4: network guide k + 1 Lanczos step (2 hops) against guide + 2 Lanczos steps;
  the loop is "useful as an add-on at equal hops" only if it wins there (it lost 2x for the ViT).
- (A) add-on: the loop is worth something on top of RBM+PP if after 8 iterations the network-loop guide (mean of
  2 seeds) is below the RBM+PP guide by >= 30% of the guide's error (exact <H>), and reported against
  guide + 1 Lanczos step (the one-hop alternative).
- (B) method: from B2 (same energy as the ViT start) the loop is compared with the ViT trajectory (table 3.16e-5,
  network 4.2-5.1e-5 after 8 iterations); a guide-independent loop should remove a similar fraction.

## Budget and hardware
<= 15 GPU-h (lead). The cip partition allows 3 GPUs per user and is fully used by other sessions; this work runs on
inter RTX 2080 Ti (fp64 1/32 of fp32, comparable to an A40 in fp64) or A40 when free.

## Amendment 1 (2026-10-13 00:00, during T1/T3, before any decisive result)
- **Bug found and fixed (no design change).** The first T1 (job 17160329) stalled at E/N = -0.157. Cause: the
  PP-only parameter mask assumed f comes first in the flattened parameter vector, but `ravel_pytree` orders the keys
  alphabetically (Wim, Wre, bim, bre, f), so 1184 of the 1296 f entries were frozen. Found with an exact
  (enumerated) SR run on 4x4 that converged and a sampled 4x4 run that showed |F| = 0. The mask is now built on the
  pytree. RBM+PP runs were not affected (mask all true). T1 and T3 were restarted (17163493, 17163494).
- **Sampler/estimator bias check (4x4, PP, unprojected, SR to -0.5230):** VMC -0.522956(69) vs exact enumeration
  -0.522964: agreement within 0.13 sigma.
- **First exact 6x6 table (check of `rbmpp_table.py`):** T1 checkpoint at step 250 (PP alone), full projection:
  exact dE/site = 8.35e-5 (E/N = -0.5037262), w_s = 3.8e-5, table time 21 min on an RTX 2080 Ti. The VMC running
  mean of the same run with the published (simple) projection is -0.50365 (dE 1.6e-4) at that point: the full
  point-group projection is worth ~8e-5/site. Both numbers will be reported; the published comparison uses 'simple'.
- Table script: a numerically real state (PP alone) now uses Re v directly (the 2x2 combination problem is singular
  when Im v = 0).
