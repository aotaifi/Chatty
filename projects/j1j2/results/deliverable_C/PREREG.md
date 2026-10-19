# Deliverable C: pre-registration (written 2026-10-13, before any new GPU run of this deliverable)

Deliverable C (PI-approved): a publishable result with a KNOWN method. Our projected ViT + 1-2 Lanczos steps + variance
extrapolation (the published baseline, Hu et al. 2013) + our calibrated lattice FN-DMC referee, at 8x8 now, 10x10 next,
12x12/14x14 planned for Paderborn. Literature: `LITERATURE.md`. Code: `experiments/deliverable_C/`.

Definitions (periodic 8x8, J2/J1 = 0.5, per-site energies in J1):
- psi_P: the 8x8 ViT projected onto D4 x spin flip, binarised own sign (phi = 0.056006), as in `results/loop_8x8`.
- E_L(x) = (H psi_P)(x)/psi_P(x); eps = E_L - E_ref; E_ref = -31.92677568 (total units).
- psi_1 = psi_P f, f = 1 + alpha eps, alpha = -0.160951 (fixed numbers, reviewed).
- psi_2 = psi_P g, g = 1 + b1 eps + b2 q, q(x) = ((H - E_ref)^2 psi_P)(x)/psi_P(x). (b1, b2) fixed before the measurement.
- var_p = <psi_p|H^2|psi_p>/<psi_p|psi_p> - E_p^2 per site = variance of the local energy of psi_p under |psi_p|^2
  (exact identity; it needs no more hops than the energy).
- Cost unit: one "projected evaluation" = psi_P at one configuration (64 ViT passes). One hop ball at 8x8 has ~145
  configurations, two hops ~1.0e4, three hops ~4.4e5 (counted on three psi_1 samples: 442776 / 425398 / 444156).

## Disclosure (what was computed before this file)
- `dc_extrap.py` on the existing raw data (no new runs): E_0 = -0.4988474, var_0 = 7.08e-4, E_1 = -0.4989687,
  var_1 = 9.81e-5, two-point extrapolation -0.498988, chain-bootstrap SE 1.7e-5 (`extrap_p01.json`).
  Its error model and the comparison rule below are fixed now, after seeing this point estimate; they are taken from
  the exact 6x6 calibration (done 2026-10-07, before any 8x8 number existed), not tuned to it.

## E1. Variance extrapolation (p = 0, 1 now; p = 0, 1, 2 when p = 2 exists)
- Primary: linear fit E = E_inf + A var through (var_0, E_0), (var_1, E_1) (Hu et al. 2013).
- Statistical error: bootstrap over independent chains inside each data set (B = 1000).
- Systematic error: +-8e-6, the largest deviation of the two-point extrapolation from exact E0 at 6x6 for starts with
  var_1/var_0 <= 0.25 (psi_P +2.2e-6, ViT(rep) +7.7e-6; `results/lanczos_baseline_6x6`). Our 8x8 ratio is 0.14.
  Starts with a larger ratio (0.35, Krylov start) overshot by -1.2e-5 at 6x6; if any 8x8 data set has ratio > 0.25 the
  systematic is +-1.5e-5.
- Heavy tails: report the extrapolation after removing the k = 5, 10, 30 most extreme samples of each set. If the
  k = 10 value differs from the full one by more than the statistical error, the statistical error is doubled.
- Comparison rule (fixed now): "in tension with Hu's -0.49906(1)" if |E_inf - (-0.49906)| > 3 x sqrt(stat^2 + sys^2 +
  1e-5^2); same rule against the DMRG torus extrapolation -0.4992(1).
- With p = 2: three-point linear and quadratic fits; quote the linear one, the quadratic-minus-linear difference as an
  added systematic.
- The extrapolated number is an estimate of E0, NOT a bound, and is labelled as such everywhere.

## E2. FN-DMC with the psi_P guide (8x8)
- Algorithm and estimator: the calibrated referee of `results/fn_calibration_6x6` unchanged: propagator
  1 - tau (H_FN - E_ref,pop), tau = min(0.025, 0.8 / max_walkers(d_FN - E_ref,pop)), one move per walker per step (stay
  or one kept hop), systematic resampling every step, M = 512 walkers, mixed estimator = population mean of the guide
  local energy, BETA-TIME-AVERAGED over beta in [0.8, 2.4] per population; result = mean over populations, SE = sd of the
  population means / sqrt(n_pop).
- Initial walkers: distinct psi_P^2 samples of the referee batches (l8ref*), random per population.
- New implementation (`dc_fn.py`): the guide is the projected ViT, evaluated once per new walker configuration and cached
  (a walker that stays or a resampled copy reuses its one-hop data), otherwise identical update rules.
- Number of populations (deviation from ">= 500"): the 6x6 rule was set to reach SE ~3e-6; the bias at M = 512 is
  <= 0.2e-6 independently of n_pop. At 8x8 one population costs ~0.2 GPU-h on an A40 slice (estimate), so 500 are out of
  reach (~100 GPU-h). Rule: run populations in batches; stop at SE <= 1.0e-5 per site, at least 16 populations, cap
  9 GPU-h. The result is quoted with its SE; no 1e-5-level claim is made with SE > 5e-6.
- Gate V1 (before the 8x8 run): the same code with a 4x4 lattice and table guides (two guides: exact amplitude x random
  log-noise of rms 0.3 with exact sign; same amplitude with 10% of configurations sign-flipped) must reproduce the exact
  lattice FN energy (sparse diagonalisation of H_FN in the Sz = 0 sector) within 3 SE, with SE <= 1e-5 per site.
  Failure stops E2.
- Reading: E_FN is a rigorous upper bound on E0 (lattice FN). "FN gain resolved" if E_FN + 2 SE_comb < E_0
  (= -0.4988474(106), independent samples). "New best 8x8 bound" only if E_FN + 2 SE < E_1 = -0.498969(14) (not expected:
  at 6x6 the psi_P-guided FN removed 23% of psi_P's error, one Lanczos step 81%).

## E3. p = 2 (8x8): pilot on ws1, production on Paderborn
- Why not on ws1: E_L of psi_2 needs psi_P on the three-hop ball (4.4e5 projected evaluations, ~85 s per sample on an
  A40 slice at the measured 5.2e3 evaluations/s). The expected p1 -> p2 gain is ~1-2e-5 per site (6x6: 2/3 of the
  remaining error; our 8x8 extrapolation puts psi_1 ~2e-5 above E0), so the paired SE must be <= 5e-6, i.e. thousands
  of samples (hundreds of slice-hours). Any fixed (b1, b2) gives a variational state, so the choice can be made on
  pilot data.
- Pilot P2-0 (ws1, <= 3 GPU-h): 64 configurations from the existing psi_1 direct samples (D1-D3, sweep 12, distinct
  chains). For each: psi_P on the three-hop ball, E_L, (H^k psi_P)/psi_P for k = 1, 2, 3 at x, and E_L, q on the one-hop
  ball. Outputs (no energy claim): seconds per sample on the slice/A40; (b1, b2) from the 3x3 Ritz problem in
  span{psi_P, A psi_P, A^2 psi_P} estimated on these samples (weights 1/f^2), shrunk halfway toward the repeated step
  g = f^2 if the two Ritz estimators disagree by more than 30%; the sd of E_L[psi_2] and of the paired difference
  E_L[psi_2] - E_L[psi_1]; this sizes the Paderborn job.
- Gate V2 (before the pilot): the three-hop formulas, rebuilt for L = 4, must reproduce the exact
  <psi_2|H|psi_2>/<psi_2|psi_2> and the local energies of an arbitrary psi_2 to 1e-9 relative on all 12870 states.
- Production estimator (Paderborn, fixed now): sample the mixture pi ∝ psi_P^2 (f^2 + g^2) with delayed-acceptance
  Metropolis (stage 1 on psi_P^2, stage 2 with the ratio of f^2 + g^2), chains started from psi_1 samples, burn-in 4
  sweeps. On each sample compute E_L[psi_1], E_L[psi_2] exactly (three hops) and the bounded weights
  w_k = psi_k^2 / pi in [0, 1]. E_k = sum w_k E_L[psi_k] / sum w_k (ratio estimators, both finite-variance: at a node
  of psi_k the weight vanishes like the square of the divergence). var_2 from the same samples. SE by chain blocks.
  Why the mixture: reweighting psi_1 samples to psi_2 has weights (g/f)^2 with an infinite-variance tail (tail index
  3/2 at the nodes of f); the mixture removes it for both states and gives a paired E_2 - E_1.
- Claim rule: "p = 2 lowers the energy" if (E_2 - E_1) + 2 SE < 0 (paired); "new lowest 8x8 bound" if E_2 + 2 SE <
  -0.498969 (our p = 1, direct).

## E4. 10x10
- Checkpoint: nqs-models/j1j2_square_10x10_05 (Hugging Face, Apache-2.0; model.safetensors 3.49 MB + config/code, three
  revisions main / symm_t / symm_trxy_ising). Not yet on ws1; downloading it needs the user's explicit OK (safety rule on
  downloads), requested in the report. Architecture = the same ViT family as our vit_dt.py (L_eff, num_layers = 8,
  d_model = 72, heads = 12, b = 2), so it loads into our code.
- Reproduction gate: plain VMC of each revision (>= 4096 samples, 1024 chains, 20 sweeps burn-in) must agree with the
  card value within 3 SE (main -0.497505103, symm_t -0.49760546, symm_trxy_ising -0.497676335). If not, stop and find
  the cause before any Lanczos step.
- Lanczos step: as at 8x8 (alpha from two-hop moments on psi^2 samples, then direct VMC of psi_1 with fixed alpha).
  Record claim only if E_1 + 2 SE < -0.4976921 (Chen-Heyl).

## Budget (ws1, this round, <= 25 GPU-h)
| item | cap (GPU-h) |
|---|---|
| V1, V2 gates (4x4, mostly CPU) | 0.5 |
| E2 FN-DMC psi_P guide | 9 |
| E3 pilot P2-0 | 3 |
| E4 10x10 reproduction (only after download approval) | 3 |
| reserve (reruns, slice contention) | 9.5 |

## Amendment 1 (2026-10-13, after the first V1 gate run, before any 8x8 FN-DMC run)
- First V1 run (4096 populations, window [0.8, 2.4]): DMC - exact E_FN = +2.1(0.4)e-4 (G1) and +3.8(0.5)e-4 (G2) per site,
  FAIL by the letter. Cause found before any change of code: the toy guides are far from the ground state
  (E_T - E_FN = 0.029 / 0.072 per site at beta = 0), so the mixed estimator still relaxes inside the window. The exact
  continuous-time transient (e^{-beta H_FN} from the psi_T^2 start, `dc_fn_gate4x4.py`) averaged over [0.8, 2.4] is
  +1.24e-4 / +3.16e-4, the same size as the measured offsets. The gate was mis-specified, not (yet) the code.
- New V1 (replaces the old one): DMC to beta = 8; (a) window [4, 8] must agree with the exact transient-corrected value
  (E_FN plus <= 1e-5 of residual transient) within 3 SE, SE <= 2e-5 per site (1e-5 at 4x4 would need ~6e4 populations;
  4x4 per-population sd is 2.5e-3, 10x the 8x8 expectation); (b) window [0.8, 2.4] must agree with the exact transient
  average within 3 SE + 15% of the transient (time-step propagator vs e^{-beta H}). Both guides must pass.
- Consequence for 8x8 (E2): the same transient can bias the [0.8, 2.4] window if psi_P's excess over E_FN does not
  decay by beta = 0.8. The 6x6 calibration saw a plateau from beta ~ 0.4 for ViT-quality guides, but the 8x8 gap is
  smaller. Therefore the 8x8 run goes to beta = 3.2 and stores every step. Primary value: window [0.8, 2.4]
  (calibrated protocol). Transient check: late window [2.4, 3.2] minus [0.8, 2.4], paired per population. If it is
  below -2 SE, the late window [1.6, 3.2] becomes the quoted value (with its SE) and the protocol value is reported as
  biased.

## Amendment 2 (2026-10-13 19:45, after the amended V1 run, before the 8x8 FN-DMC run)
- Amended V1 at M = 512, window [4, 8]: DMC - (E_FN + exact transient) = +1.12(0.21)e-4 (G1), +1.07(0.25)e-4 (G2):
  FAIL by the letter. Diagnosis run (G1, same code, M = 128 and 2048): +3.20(0.42)e-4, +1.12(0.21)e-4, +0.42(0.21)e-4.
  Fit a + b/M: a = +3.0(1.9)e-5 (1.6 sigma from 0), b = 0.038(6) per site x M, chi^2 = 0.27 (1 dof).
- Cause: population-control bias ~ 1/M, as in the 6x6 calibration (which fitted a + b/M too). It is large here because
  the toy guides are bad: variance of H in the 4x4 guide G1 = 3.34 (total units), 74x the 8x8 psi_P (0.045) and
  ~140x the 6x6 ViT guides (~0.024). In total-energy units b_total / var = 0.18 (4x4 G1) vs ~0.1 (6x6 calibration):
  the same law. Predicted 8x8 psi_P bias at M = 512: 0.18 x 0.045 / 512 / 64 ~ 2.5e-7 per site, negligible against the
  1e-5 target.
- Gate decision (fixed now): the implementation is validated by the 1/M extrapolation of G1 (a consistent with 0).
  G2 at M = 2048 is running in parallel; if its bias is not consistent with the G1 law (|bias_2048 - b_G2/2048| test
  using its M = 512 value, > 3 sigma from a = 0), the 8x8 FN result is withheld until the cause is found.
- The 8x8 run starts now: two jobs on A40-16gb slices (seeds 1, 2), M = 512, beta to 3.2, populations in batches of 4,
  each job stops at SE <= 1.4e-5 (combined ~1e-5) with >= 8 populations, or at its 4.5 GPU-h cap (total cap 9 GPU-h as
  registered). Analysis: dc_fn_analyze.py on the merged populations.

## Amendment 3 (2026-10-13 20:00, after pilot P2-0, before any p = 2 production run)
- Pilot P2-0 (job 17182266, 64 psi_1 samples, 1.53 GPU-h on an A40-16gb slice): 85.9 s per sample, 4.44e5 projected
  evaluations per three-hop ball (5.2e3 evaluations/s).
- The pre-registered coefficient rule fails: the 3x3 Ritz problem is not estimable from 64 samples (the "unbalanced"
  moment estimator gives a non-positive-definite overlap matrix; the "balanced" one gives b1 = -390, b2 = 48, i.e. noise).
  The optimal p = 2 polynomial is set by rare tail configurations, which 64 typical samples do not contain.
- Replacement (fixed now): psi_2 = psi_P f^2 (b1 = 2 alpha, b2 = alpha^2), the repeated first step. It needs no sampled
  moments and is variational. On the pilot samples g/f lies in [0.987, 1.030]; sd per site of E_L[psi_2] 7.5e-4,
  of E_L[psi_1] 6.3e-4, of the paired difference 6.9e-4 (tail-free samples, so a lower bound on the true sd).
  Exact 6x6 check of how much of the optimal p = 2 gain the repeated step keeps: job 17184225 (dc_p2rep_6x6.py).
- Production size: paired SE 5e-6 needs ~1.9e4 samples (>= 1.1e10 projected evaluations, ~590 A40-slice-h); SE 1e-5
  needs ~4.8e3 samples. Not affordable on ws1 this round; Paderborn draft p2_prod_h100.sbatch with b1 = -0.321902,
  b2 = 0.025905.

### Gate V1 outcome (2026-10-13 20:25): PASS under amendment 2
- G1, window [4, 8], bias vs M: +3.20(0.42) / +1.12(0.21) / +0.42(0.21) e-4 at M = 128 / 512 / 2048; a + b/M fit
  a = +3.0(1.9)e-5, b = 0.038(6), chi^2 = 0.27.
- G2: +1.07(0.25) / +0.45(0.25) e-4 at M = 512 / 2048; two-point a = +2.4(3.4)e-5, b = 0.042, same law as G1.
- The DMC code reproduces the exact FN energy after the population-control extrapolation; the bias follows b/M with
  b proportional to the guide's variance. For the 8x8 psi_P guide (variance 74x smaller than G1) the predicted bias at
  M = 512 is ~2.5e-7 per site.

## Amendment 4 (2026-10-13 20:45, before the run): late-time psi_1 samples for the extrapolation
- Observation (from existing data, `README.md` step 2): the extrapolated E0 depends on which psi_1 samples are used:
  all 10240 -0.498988, sweep >= 12 (5120) -0.498966, latest 2048 -0.498947. The review found no resolved relaxation of
  E_1 (T6: +1.8(3.4)e-5), but the extrapolation also uses sigma^2_1, and the trend is in the direction that would
  increase the tension with Hu.
- Run (cost ~1.2 GPU-h on an RTX 2080 Ti, reviewed code rv_late.py unchanged): continue the 1024 chains of review job
  D1 (not continued so far) by 40 psi_P sweeps + 4 delayed-acceptance sweeps on |psi_1|^2, then exact E_L1 for all
  chains. Same protocol as the review's T6 (L2, L3).
- Reporting rule (fixed now): quote the extrapolation with (i) all samples and (ii) late samples only (L2, L3 and the
  new L1; ~3072 samples, >= 80 sweeps after the random start). If (ii) - (i) > 2 sigma (bootstrap of the difference by
  chains), (ii) is the primary value; otherwise (i) is primary and (ii) is quoted as the conservative value. The Hu
  tension rule of E1 is applied to the primary value.

## Outcomes (2026-10-19), against the registered rules
- E1 (extrapolation, amendment 4 rule): all samples -0.498992(16) stat (8) sys; late-only -0.498974(27); late - all
  +1.8(2.4)e-5 < 2 sigma -> "all" is primary. Trim k = 10 moves it +6e-6 < stat -> no doubling. Hu tension: 3.4 sigma
  (> 3) -> "in tension with Hu's -0.49906(1)"; DMRG: 2.1 sigma (no tension by the rule).
- E2 (FN-DMC psi_P guide): 8 populations (both jobs stopped by the 4.5 GPU-h per-job cap; 7.2 GPU-h total, cap 9),
  E_FN = -0.498889(11). FN gain resolved: -4.2(1.5)e-5 (E_FN + 2 SE_comb < E_0). New best bound: no. Transient rule:
  late - protocol = +2.3(1.8)e-5, not below -2 SE -> protocol value quoted. The SE target 1e-5 was missed by 7%.
- E3: pilot done; production not run (Paderborn).
- E4: not run (download not approved).

## E5. 12x12 RBM+PP cost pilot on a Paderborn H100 (pre-registered 2026-10-19, before submission; lead decision)
- Purpose: measure what option (b) of PLAN.md costs at 12x12 before committing H100 time. No energy claim.
- Run: `cluster/paderborn/deliverable_C/rbmpp_L12_pilot_h100.sbatch`. PP alone (Nomura-Imada pair product, full
  real 144 x 144 f, no RBM yet), our JAX code from results/rbmpp_6x6 unchanged, 'simple' published projection
  (144 translations x {E, C4} x spin flip = 576 images), SR (natural gradient) lr 0.05 -> 0.03 (cosine), diagonal
  shift 1e-3, max |d|_S 0.3, 4096 samples per step, 300 steps, checkpoints every 25 steps, then a 16384-sample VMC
  evaluation. 1 H100, wall limit 2 h (cap 2 H100-h).
- Recorded: seconds per SR step (sampling, local energies, S solve), memory, E/N and Var/N against step.
- Checks (pass/fail, fixed now):
  (i) sanity: all energies finite; acceptance in [0.1, 0.8]; mean E/N over the last 50 steps below the mean over steps
      25-75 by more than 5 combined SE (the optimiser works at 12x12).
  (ii) cost projection: steps to plateau taken from 6x6 (PP alone flat after ~400 steps; production 3000 PP steps + 900
      RBM+PP steps + 1500 steps at 4x samples, results/rbmpp_6x6) times the measured seconds per step at 12x12 (4x
      samples -> 4x time for the last stage). Pass for option (b) if the projected PP + RBM+PP training cost is
      <= 50 H100-h; if it is 50-100 H100-h, report and propose cost cuts (projection only at the end, Nomura's 4x4
      sublattice PP); if > 100 H100-h, option (b) is not the cheap route and the plan is revised.
  (iii) reference point, not a criterion: published 12x12 RBM+PP (4x4-sublattice PP) -0.496791(4); GCNN -0.496769(9).

### E5 outcome (2026-10-19, job 3749014, 1 H100, 17:08-18:08, stopped by the 1 h limit after 20 SR steps)
- Cost: 96.6 s per SR step (4096 samples, 576 projection images, PP alone, 20,736 active parameters); thermalisation
  (20 sweeps) 1489 s; XLA asked for 91.8 GB device memory (above the 66 GB it could get; it ran after rematerialisation).
- (i) sanity: energies finite, acceptance 0.23-0.25: pass; the "last 50 vs steps 25-75" comparison is not evaluable
  (only 20 steps). E/N fell from -0.0881 (step 1) to -0.1867 (step 20) from the random start (target region ~-0.497).
- (ii) projection by the registered rule: (3000 + 900) x 96.6 s + 1500 x 4 x 96.6 s = 9.6e5 s = **~266 H100-h**,
  a lower bound (12x12 will need more steps than 6x6). > 100 H100-h -> **option (b) as configured is not the cheap
  route; the plan must be revised** (registered consequence).
