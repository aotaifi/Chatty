# Independent adversarial review: one Lanczos step on the projected 8x8 ViT

Reviewer: Claude (Opus, review session), 2026-10-12. Claim under review (README FU1, commit b176c90):
psi_1 = psi_P (1 + alpha (E_L - E_ref)), alpha = -0.160951, E_ref = -31.92677568, gives
E/N = -0.4989530(164) (combined: VMC of psi_P + paired moment gain) and -0.4989476(231) (direct VMC of psi_1),
below RBM+PP -0.498886(1).

Terms used below:
- psi_P: the D4 x spin-flip projected ViT with its binarised own sign. E_L = (H psi_P)/psi_P, its local energy.
- factor f(x) = 1 + alpha (E_L(x) - E_ref); psi_1 = psi_P f. |psi_1|^2 vanishes where f = 0 ("nodes of the factor").
- E_L1 = (H psi_1)/psi_1, the local energy of psi_1; it needs psi_P on all two-hop configurations.
- "Direct VMC": Metropolis sampling of |psi_1|^2 and the mean of E_L1. It is a variational upper bound for any fixed
  alpha (no optimisation bias), provided the chains are equilibrated and the error bar is right.
- "Moment estimate": the 2x2 Ritz value from sampled moments of psi_P. Not variational on finite samples.
- SE: standard error of a mean, from per-chain means (independent chains).
- Heavy tail: a distribution whose tail falls like x^(-k). With tail index k < 2 the variance is infinite and the usual SE
  is meaningless; with k < 3 the SE of the SE is unreliable.

## Pre-registration (written before any new computation; committed before the new runs)

### Loopholes to be checked
L1 Sampling of psi_1 (heavy tails near factor nodes, equilibration from psi_P starting points, 4 sweeps burn-in,
   512 chains x 4 samples, accept 0.147).
L2 Correctness of E_L, (H^2 psi_P)/psi_P and E_L1 (bonds, J2 diagonals, PBC, diagonal term, sign convention, Sz = 0,
   symmetrisation of psi_P).
L3 Statistics of the combined estimate (heavy tails of m2 and m3; alpha chosen on the same samples; the 1.6e-5 spread
   between the two Ritz estimators).
L4 psi_P's own energy (equilibration of the plain VMC; consistency with the referee batches and the earlier ViT
   number; single-valuedness of psi_P under fp32 numerics, i.e. the same configuration always gets the same sign).
L5 Literature: an NQS + Lanczos 8x8 energy below RBM+PP already published?

### Tests and pass/fail criteria
T1 (L2, exact formulas). 4x4 PBC J1-J2 (J2 = 0.5), the l8_core functions (neighbors, diag_vec, onehop) rebuilt with
   L = 4 and the E_L2 / E_L1 / Ritz formulas copied verbatim from l8_direct.py and l8_eval.py, with a toy state that is
   not an eigenstate (exact ground state times a random positive perturbation, with the Marshall-like sign of the
   ground state). Exact enumeration of the estimator over all 12870 Sz = 0 configurations versus sparse-matrix
   <psi_1|H|psi_1>/<psi_1|psi_1> and the exact 2x2 Ritz value. PASS: agreement to 1e-9 relative.
T2 (L2, the real 8x8 lattice). For >= 16 referee samples, an independent implementation (own bond list from
   coordinates, own diagonal term, own D4 x flip images built from coordinates, unprojected network summed over the 16
   images) recomputes psi_P on the full two-hop set and gives E_L, E_L2, E_L1. PASS: |difference| < 1e-6 (total
   energy units) against the stored referee values for E_L and E_L2; the stored log|psi_P| and sign equal the fresh
   ones (|d log a| < 1e-4, identical signs); psi_P invariant under random translations x D4 x flip (|d log a| < 1e-4).
   Also: every stored configuration has 32 up spins.
T3 (L4, single-valuedness). Configurations that occur several times in the referee data (as sample or neighbour,
   evaluated in different batches) must carry the same sign; report max |d log a| and the number of sign
   disagreements. PASS: no sign disagreement among repeated configurations, or a disagreement rate < 1e-6 per
   evaluation (then shown to be energetically irrelevant).
T4 (L1, L3, L4 on existing data; diagnostics, no pass/fail by itself).
   - Hill estimator of the tail index of: |E_L - mean| (psi_P samples), the per-sample terms of m2 and m3, f^2
     under psi_P, |E_L1 - E1| in the direct samples. Fraction of m2, m3 carried by the top 0.1 % / 1 % samples.
   - Bound on the non-equilibration bias of the direct run: the plain mean of E_L1 over psi_P samples (zero
     equilibration) versus the reweighted psi_1 energy; the direct-run estimate split by sample time (t = 1..4 sweeps
     after burn-in).
   - Optimisation bias of the moment estimate: choose alpha on one half of the referee chains, evaluate the
     reweighted energy at that fixed alpha on the other half (both ways), compare with the same-sample Ritz value.
   - Burn-in check of the psi_P VMC: first versus second half of each chain's 48 samples; blocking of the per-chain
     means.
T5 (L1, decisive). A new direct VMC of psi_1 with the same fixed alpha, E_ref (so no optimisation bias), independent
   seeds, chains started from psi_P samples of referee batches 7000/8000/10000 (direct run a used batch 5000),
   3 jobs x 1024 chains, burn-in 8 sweeps (2x the original), 2 samples per chain 2 sweeps apart (6144 samples,
   3x the original). Sampler: delayed-acceptance Metropolis (stage 1 on |psi_P|^2 with one amplitude, stage 2 accepts
   with f(y)^2 / f(x)^2; exact detailed balance for |psi_1|^2), checked in T1-style on 4x4 against exact |psi_1|^2
   histograms before use. The chain-average of E_L - E_ref and of log f is recorded every sweep (equilibration trace).
   Criteria:
   (a) CONFIRMED if E1_new + 2 SE_new < -0.498886 (the README's own rule, now on independent data), and
   (b) |E1_new - E1_direct_a| < 2 sigma and |E1_new - (-0.4989530)| < 2 sigma (consistency), and
   (c) no drift: the two sample times agree within 2 sigma and the equilibration trace of <E_L - E_ref> after burn-in
       has no trend beyond 2 sigma, and
   (d) the Hill tail index of |E_L1 - E1| on the pooled direct samples is > 2.5 (finite variance with margin).
   If (a) fails while E1_new < -0.498886: NOT CONFIRMED at 2 sigma (lead stays a lead). If (c) or (d) fails: the
   error bar is not trustworthy; NOT CONFIRMED (cause named).
   The number to publish is the pooled direct VMC (run a + new runs, both fixed alpha, chain-level SE), because it is
   the only estimator that is variational on finite samples; the combined moment estimate is quoted as a cross-check.
Budget: T2 ~0.1 GPU-h, T5 ~6-7 GPU-h on A40 slices (<= 10 GPU-h total).

## Results
(to be filled after the runs)
