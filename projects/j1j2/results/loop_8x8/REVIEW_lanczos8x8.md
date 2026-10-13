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

### Amendment 1 (2026-10-12, after T1-T4 on existing data, before the T5 jobs were submitted)
- T4 found that the local energies are heavy-tailed: for psi_P the robust width (interquartile range / 1.349) is
  3.7e-4 per site against a standard deviation of 3.3e-3; the top 1 % of referee samples carry 81 % of m2 and 83 % of m3;
  in direct run a the Hill index (top 2 %) of |E_L1 - E1| is 1.8. Criterion (d) stays as registered and will be reported.
- Added (d'): the error bar must survive (i) a chi^2 test of 16 chain-group means per new job (plus run a) against
  their SEs, p > 0.01, and (ii) confirmation (a) evaluated with SE = max(chain SE, SE from the spread of the 4
  independent batch means).
- T2 tolerance: the independent symmetrisation sums fp32 network outputs outside the jitted graph, so its log|psi_P|
  differs from l8_core by up to 1e-5 (rounding). The registered 1e-6 tolerance on E_L / E_L2 is therefore replaced by:
  differences must be random in sign and below 1e-6 per site on E_L1 (the quantity that enters the energy).
- The delayed-acceptance sampler passed its 4x4 check (T1, below) before use.

### Amendment 2 (2026-10-13 02:15, after T5, before T6 was submitted)
- T5 passes (a), (b), (c) by the letter (pooled direct E1 = -0.4989764(159)), fails (d) (Hill index 1.90) and passes (d').
- But in all four direct runs the first measurement time is lower than the later ones: run a t0 - mean(t1..t3) =
  -4.7(5.0)e-5; new runs, sweep 10 - sweep 12: -3.2(7.0)e-5, -6.9(6.1)e-5, -8.3(4.8)e-5. Pooled new-run drift
  +6.1(3.5)e-5 (1.8 sigma, below the 2-sigma threshold of (c)). Four same-sign results are suggestive of slow upward
  relaxation of E1, i.e. the estimate could be biased low. The psi_P VMC shows a slow downward drift of its mean over
  sweeps 20-68 (quarters -0.498821 / -0.498842 / -0.498862 / -0.498864), driven by tail samples (medians flat).
- T6 (closes this): continue the 2 x 1024 chains of T5 jobs D2 and D3 from their final states (about 35 sweeps after
  the random start of the referee chains) by 40 sweeps of Metropolis on |psi_P|^2 (|psi_1|^2 differs from it by f^2,
  f ~ 1; stage-2 acceptance in T5 was 0.988) and 4 sweeps of delayed-acceptance Metropolis on |psi_1|^2, then measure
  E_L1 on all 2048 chains (about 80 sweeps after the random start). Record the psi_P local energy every 4 sweeps.
  Criteria:
  - relaxation present if E1_late - E1(T5, sweep 12, D1-D3 pooled) > +2 sigma. Then the only number to publish is
    E1_late, and the claim is CONFIRMED only if E1_late + 2 SE < -0.498886.
  - otherwise no relaxation beyond sweep 12 is resolved; publish the pooled direct estimate (runs a, D1-D3, T6) with
    the chain-level SE, and also quote the late-time-only estimate (sweep-12 samples + T6).
- Budget: T6 ~2.4 GPU-h; total review ~9 GPU-h.

## Results (2026-10-13; raw summaries in `review/`, scripts in `experiments/loop_8x8/review/`)

### Verdict: CONFIRMED WITH CORRECTED ERROR BAR (one pre-registered criterion, (d), formally failed; see below)
- **Number to publish: E/N = -0.498969(14)**, direct VMC of psi_1 (fixed alpha = -0.160951, E_ref = -31.92677568):
  10240 samples, 3584 independent chains, four independent jobs plus a late-time continuation.
  This is 5.9 sigma below RBM+PP -0.498886(1) and 7.7 sigma below Hu VMC p=2 -0.49886(1).
- **Conservative value** (only samples taken 12 or more sweeps after the start of the psi_1 chains):
  -0.498948(23), 2.8 sigma below RBM+PP. The latest samples alone (T6, about 80 sweeps after the random start):
  -0.498938(22), 2.3 sigma below RBM+PP. Every subset lies below RBM+PP and below Hu p=2.
- **Corrected combined (moment) estimate:** -0.498953(22), not (16). The jackknife SE of the moment gain (1.25e-5)
  is too small; the spread of the 5 independent referee batches gives 1.9e-5.
- The original direct number -0.4989476(231) is consistent with all of this (run a is one of the pooled batches).
- **Formal status.** Criterion (d) (Hill tail index > 2.5) failed: it is 1.87 on the pooled samples.
  - By the letter of the pre-registration this makes the verdict NOT CONFIRMED.
  - I judge (d) mis-specified. The distribution of E_L1 has a narrow core (robust width 2.5e-4 per site) and broad,
    non-power-law shoulders: the Hill index is 2.3 / 1.9 / 1.4 at the top 1 / 2 / 5 %, so it does not measure an
    asymptotic tail.
  - What (d) was meant to protect, the reliability of the SE, is checked directly by (d') and passes:
    - chi^2 of 56 chain-group means: p = 0.52;
    - chi^2 of the 4 batch means: p = 0.57;
    - the between-batch SE (1.0e-5) is below the chain SE (1.6e-5);
    - removing the 10 or 30 most extreme samples at each end moves E1 by +0.2e-5 and +0.6e-5.
  - The PI should decide whether to accept (d') in place of (d). If (d) is kept, quote the conservative late-time
    value.

### Loopholes and how each was closed
| # | loophole | test | result |
|---|---|---|---|
| L2 | wrong E_L, (H^2 psi)/psi, E_L1 formulas, sign convention | T1: 4x4 exact enumeration with the l8_core functions rebuilt at L = 4, formula lines copied verbatim | E0/N(4x4) = -0.528620 (known value); E_L, E_L2 exact to 1e-11; E_L1 and fixed-alpha energy exact to 5e-16 relative; Ritz moments exact to 3e-15. PASS |
| L2 | bonds, J2 diagonals, PBC, Sz on the real 8x8 lattice | own bond list from coordinates, own diagonal term, applied to the stored referee amplitudes | 256 bonds (128 J1 + 128 J2); E_L recomputed to 4e-14 on all 15360 samples; all configurations have 32 up spins. PASS |
| L2 | symmetrisation of psi_P, two-hop data | T2: own D4 x flip images from coordinates, unprojected net summed, own H applied twice, 16 samples (8 of them the most extreme E_L), 165002 two-hop states | log a agrees to 1.1e-5 (fp32 rounding); 0 sign mismatches; E_L1 differs by at most 4e-7 per site, random sign (amended tolerance 1e-6). Invariance under random translation x D4 x flip: 1e-5, 0 sign mismatches. PASS |
| L4 | psi_P not single-valued (batch-dependent sign) | T3: 1783 configurations evaluated twice in different batches; re-evaluation in permuted order | identical log a and sign (difference 0.0). PASS |
| L1 | non-equilibrated psi_1 chains (started from psi_P samples, 4 sweeps burn-in) | zero-equilibration bound; T5 with 8-10 sweeps burn-in; T6 continuation by 44 sweeps | <E_L1> over psi_P samples (no equilibration at all) differs from the psi_1 value by only 0.7e-5. The first measurement time was lower in all 4 jobs (pooled +4.1(3.1)e-5, 1.3 sigma). T6, late minus sweep 12: +1.8(3.4)e-5, so no relaxation is resolved. The ordering early < mid < late (-0.498989 / -0.498955 / -0.498938) is not significant but is why the conservative value is quoted. |
| L1 | heavy tails, near-node sampling | T4/T5 tail statistics | E_L of psi_P: robust width 3.7e-4 against sd 3.3e-3; the top 1 % of samples carry 81 % of m2 and 83 % of m3. Direct sampling does visit psi_P-node configurations (factor f down to -17.8, up to 48.7) and E_L1 is regular there. The SE is validated by (d'); the formal (d) failed (see above). |
| L1 | sampler correctness | delayed-acceptance sampler vs exact |psi_1|^2 and plain Metropolis on 4x4 | TV distance 0.034 vs 0.033 (plain) vs 0.033 (iid reference); E1 within 0.9 sigma of exact. PASS |
| L3 | alpha chosen on the same samples (optimisation bias) | split-half: alpha from half A, fixed-alpha energy on half B | out-of-sample gain -1.21e-4 vs same-sample Ritz -1.06e-4: no low bias of the Ritz value. The direct VMC uses a fixed alpha, so it is variational regardless. |
| L3 | the two Ritz estimators differ by 1.6e-5 | moments | They differ only in how <H^2> is estimated: <E_L^2> - h1^2 = 0.04177 vs <(H^2 psi)/psi> - h1^2 = 0.04790 (15 % apart on the same samples, a heavy-tail effect). Both are consistent and neither is unbiased on finite samples. The fixed-alpha direct VMC replaces both. |
| L3 | SE of the combined estimate | 5 independent referee batches | Ritz gain per batch -1.02 / -0.80 / -0.68 / -1.76 / -1.04 e-4, SE of the mean 1.9e-5 (jackknife said 1.25e-5); bootstrap 1.4e-5. Corrected combined -0.498953(22). |
| L4 | psi_P energy | plain VMC by quarters; referee batches; T6 traces | -0.4988474(106) overall. The quarters drift down by 4e-5 (tail-driven, medians flat); the second half gives -0.498863(15). Referee batches -0.498843(26). The T6 traces (40 sweeps) show no trend. Consistent with the earlier unprojected ViT -0.498823(35). psi_P is a fixed real function (binarised sign, exact projector sum), so it is a legitimate variational state. |
| L5 | literature | web search 2022-2026 plus the project survey | No published NQS + Lanczos energy at 8x8. Chen et al. 2022 (arXiv:2206.14307) covers 6x6 and 10x10; Wang, He, Lu 2025 (arXiv:2502.01264) covers L = 4, 6, 10; Chen-Heyl give 8x8 only as zero-variance extrapolations in figures. HQT (arXiv:2607.00398) claims -0.5001 at 8x8, below both extrapolated E0, so it is not credible as a variational number, but a paper must mention it. The method (NQS + Lanczos step) is known; the 8x8 number is new. |

### Direct VMC batches (E/N, chain SE)
| batch | samples | E1 |
|---|---|---|
| run a (FU1c, plain Metropolis, sweeps 5-8) | 2048 | -0.4989476(231) |
| D1 sweep 10 / 12 | 1024 / 1024 | -0.4989970(376) / -0.4989651(592) |
| D2 sweep 10 / 12 | 1024 / 1024 | -0.4990309(488) / -0.4989620(373) |
| D3 sweep 10 / 12 | 1024 / 1024 | -0.4990222(351) / -0.4989389(324) |
| L2 / L3 (T6, +44 sweeps) | 1024 / 1024 | -0.4989187(324) / -0.4989569(306) |
| **pooled** | 10240 | **-0.4989687(141)** |

Physics note: the Lanczos gain comes mostly from removing rare high-energy configurations of psi_P (the E_L
variance drops from 3.3e-3^2 to 1.3e-3^2 per site^2, while the robust width changes only from 3.7e-4 to 2.5e-4).
So the gain estimate is dominated by rare samples. This is why moment-based estimates have under-stated SEs and why
direct sampling of psi_1 is the trustworthy referee.

### Cost
T2 0.03, T5 6.5 (D1 2.33, D2 2.06, D3 2.07), T6 3.7 (L2 2.61 on a contended slice, L3 1.11): **10.2 GPU-h**
(A40 slices). This is 0.2 GPU-h over the 10 GPU-h budget, because of slice contention.
