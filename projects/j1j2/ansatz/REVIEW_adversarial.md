# Adversarial review of ANSATZ_MEMO.md (J1-J2, J2/J1 = 0.5)
Reviewer: independent worker, 2026-10-24. Read-only: no jobs run. One CPU check (4x4 exact diagonalisation,
seconds). Every number below was re-read from the raw JSON or log named next to it (runs root
/scratch/a/A.Otaifi/ansatz_runs/). Verdicts: SOUND = the kill holds as registered; WEAK = the kill is plausible but
rests on something untested, with the cheapest test that would settle it; FLAWED = the evidence does not support the
kill as stated.

## Summary
| candidate | verdict | one-line reason |
|---|---|---|
| C1 global sign-rule interference | SOUND | R/R_null = 0.990 re-derived; the test could have passed |
| C2 add-on layer (C2-T1) | SOUND (procedural note) | 4.45x / 5.55x / 1.03x margins; registered at 6x6, run at 4x4 only |
| C2-T2 trained J2 layer | SOUND | R2 runs have plateaued; the J_H pole only handicaps the arm that wins |
| C5-T0 coupled output (parked) | WEAK | coupled arms still falling 3-6% per 500 steps, standard arms flat; seeds missing |
| C3 vison-dressed Z2 SL | WEAK | the control can repair an unconverged base; the base is not Hu's ansatz |
| C4 plaquette frame | WEAK | 4x4 only (rule says both sizes); local optimiser, 2 starts |
| C6 unfrustrated parent | FLAWED | min-variance search returns parents whose ground state is not Phi; failed sanity check not reported |
| C6b gate similarity | SOUND for the registered family | g(H) = 0.38321 re-derived independently; scope is one 1-parameter family at 4x4 |
| C2-L stored pre-image | SOUND kill, WEAK mechanism | 34x above the kill bar; "~30x amplification" mixes in a 2-2.4x worse bulk fit |

## Per candidate

### C1 (c1_t0/c1_t0.json): SOUND
Re-derived: U = viol_w(K1) = 4.842e-3, W_M = 1.9558e-2, R = 0.2476; R_null = 0.5 (4.842e-3 + 4.938e-3) / 1.9558e-2 =
0.2500; R/R_null = 0.990 (kill bar 0.6). The forced-class argument holds: with e_X = e_Y (symmetric), exactly one
of K1, K2 has all three terms the same sign, and K2..K4 can take any sign, so the forced class's violation weight is
the ansatz's only sign cost. The test could have passed: it needed violations concentrated in one class. The
per-decade table (e.g. -11: K1 1.020e-3, K2 1.005e-3, K3 1.023e-3, K4 1.046e-3) shows no structure. Checks are real
(broken rule 7.93M identity violations). Scope: only the three product rules; the memo says so.

### C2 as add-on (small/c2t1_4x4.json): SOUND, with a procedural note
Re-derived E_FN(H_J2 layer)/E_FN(H layer): |psi0|-Marshall 6.944e-3/1.560e-3 = 4.45; Marshall-FN 3.654e-3/6.58e-4 =
5.55; no-J2-hop GS 1.601e-2/1.556e-2 = 1.03. Kill rule (>= on 2 of 3) is met. Note: C2-T1 was registered for exact
6x6 with ViT, PP and Marshall-FN bases. The kill was declared on a 4x4 preview with other bases, and the plan itself
calls 4x4 anomalous. I still sign it, because the margins are 4-5x and C2-T2 at 4x6 points the same way. Cost to
make it as written: C2-T1 at 6x6, no training, <= 2 H100-h.

### C2-T2 (exact_vmc_4x6/verdict.json, c2t2/runs/*.json): SOUND
Re-derived: J 1.6131e-2 / 1.0228e-2 = 1.577; R2 3.556e-3 / 2.827e-3 = 1.258 (best of 3 seeds); the seed means give
3.948e-3 / 3.148e-3 = 1.25.
- "Unconverged at 3000 steps": the flag means L-BFGS tolerance was not met. The curves are flat: drop over steps
  2500-3000 is 0.0-0.4% of dE for all nine R2 runs. The arms differ by 25%, which is larger than the drift. The seed
  spread (local minima, up to 30%) is comparable to the gap, so "worse" is marginal for R2. But a pass needed a ratio
  <= 0.5, which no seed comes near.
- J_H a = -5.5e6 / -5.2e7 / -1.37e8 (3 seeds), with dE equal to 1e-10. The optimiser is at the pole b = 1/a -> 0- of
  the (1 + aH) parameterisation: psi -> H phi. "conv True" is spurious there, because dE/da = -(dE/db)/a^2 vanishes
  for any slope in b. The true H-layer optimum may lie at b > 0, which the optimiser cannot reach, so it would be
  lower still. This handicaps the H arm, the arm that wins, so the kill is conservative. Use the (b + H) form in any
  re-run.

### C5-T0 (parked, exact_vmc_4x6/c5t0_R*/runs): WEAK (not a kill; the inconclusive leans against the idea)
Relative drop over the last 500 steps: coupled R2 6.4%, R4 s0 3.1%, R1 s0 1.6%; standard R2 0.15%, R1 0.0% (L-BFGS
converged at 1991/2702 steps). The coupled arm was still descending when the standard arm had stopped, so the ratios
1.30 (R2, one seed) and 0.97 (R4) understate it. Seeds are incomplete (R1 two, R2 one; three were registered). The
memo does not count C5 as a kill, which is right, but "no consistent advantage" is not established. Cheapest test:
continue all C5 arms from the saved params to L-BFGS convergence, 3 seeds per alpha, about 1-2 A40-h.

### C3 (c3_t0_4x6/c3_t0_4x6.json, c3_fn_4x6/c3_fn_4x6.json): WEAK
Re-derived at rank 20: V 7.358e-4; R' 4.060 / 5.685 / 4.320e-4; V/R' 1.81 / 1.29 / 1.70 (kill bar 0.7 on 2 of 3).
E_FN 4.56e-4 vs 2.72 / 3.89 / 2.86e-4. The rebuilt dE_var in c3_fn matches the stored value to 3e-12.
- Greedy rank curves: the greedy stops when the lowest-energy candidate adds no rank (code line 292). It stopped at
  k = 20 / 23 / 28 for R' although the full R' spans reach rank 69-72 at dE 2.4-3.3e-4. V stopped at 37 (full rank
  39, 6.57e-4). So the "largest common rank" of 20 is set by the stopping rule, not by the spans. This does not save
  C3: V's full span (6.57e-4) is above every R' greedy value at rank 20, and the curves only fall, so V/R' >= 1.15 at
  any common rank >= 20. At ranks 6 and 10-13, V beats 2 of 3 R' seeds, so the verdict depends on where the
  comparison is made. The registered point (largest common rank) gives KILL.
- Unconverged base: the AA sector stopped at 60 Nelder-Mead evaluations (overlap^2 0.889, "Maximum number of function
  evaluations"), and the other BC sectors were not optimised. The R' control multiplies t and D on string bonds by
  U[0.5, 1.5]. That spans the tangent space of the mean-field parameters, so R' can partly repair the unconverged
  optimisation, while V (discrete sign flips) cannot. This is an unfair control for "flux fluctuations beyond a good
  Z2 state". The 4x4 result (V 1.1e-5 vs R 5e-4) did not reproduce, which is consistent with either reading.
- d_xy on (1,1) bonds: as registered ("NNN d_xy"). Hu et al.'s (2,2) d_xy cannot be posed cleanly on Lx = 4: x+2
  and x-2 are the same site, so the d_xy pair cancels in x-periodic sectors and doubles in x-antiperiodic ones. The
  tested base is therefore not Hu's ansatz, and the memo's "Z2 partons dressed by visons" kill is for this base only.
- Cheapest test: rerun c3_t0_4x6.py with (a) Nelder-Mead to convergence in all four BC sectors (maxfev ~300) and (b)
  the three parameter-derivative states d psi / d(D1, D2, mu) added to the base span before both arms. Then R' can no
  longer win by repairing the base. The original job took 1421 s on one node, so this is under 1 CPU-node-hour. The
  real test (Hu's (2,2) form) needs 6x6, which the plan already says.

### C4 (small/pre_4x4.json): WEAK
4x4: from the Marshall start, W_neg 0.012725 vs 0.012731 (ratio 0.9995, stopped after 13 steps); from the
plaquette-eigenbasis start, 0.0183 (shared) and 0.0153 (free) after 300 steps, worse than Marshall. The landscape is
non-convex and two starts were tried. The registered rule says pass/kill "at both sizes"; the 4x6 run was stopped
(ws1 load), so the kill was declared on half the rule, at the size the plan calls anomalous. Cheapest test: 20-50
random-start runs at 4x4 (CPU minutes), then the registered 4x6 run (about 1 CPU-h on a cluster node, not ws1).

### C6 (c6_t0/r4x4, r4x6 logs and JSON): FLAWED
- The criterion cannot certify absence. "Parent = argmin c^T C c" picks an H_p for which Phi is an approximate
  eigenvector, not its ground state. At 4x6 the stoquastic parents have ground states with S^2 = 11.8-12 (S = 3,
  magnetic; cond 5/10/30 s arms, r4x6/c6_t0_4x6.log), ferromagnetic J1 ZZ (-1.14 at cond 5) and rel_var 0.0088-0.019.
  "Fidelity 0" means "Phi is an excited state of the min-variance H_p". It does not mean "no stoquastic parent".
- The pre-registered sanity check failed at both sizes ("cond100_negJ2=False" in both VERDICT lines). It was not
  reported in PLAN or memo. With the J1 XY coefficient fixed to +1, the expected parent (about -H_J2, J1 small) is
  outside the search space, so the check could not pass either. The tool is unvalidated on the case it was meant to
  find.
- Counter-evidence in the same runs: at 4x4 the stoquastic arm reaches fidelity 0.949 / 0.973 / 0.988 at cond 5 / 10
  / 30. The rule's own baseline, (1 + aH_J2) on the Marshall-FN ground state, is itself a member of the C6 family: the
  FN Hamiltonian is local and stoquastic, and it gives E_var 4.06e-3 at 4x6. "No stoquastic parent" is false as
  stated. The honest finding is "no parent better than FN was found by min-variance".
- Formally the registered kill line (dressed s-parent E_var 0.818 >= baseline 4.06e-3 at 4x6; 1.36e-2 >= 4.17e-3 at
  4x4) is met. But it measures the search, not the hypothesis.
- Cheapest test: at 4x4 (D = 12870, Lanczos about 0.1 s), optimise the ~16 stoquastic couplings directly for maximum
  |<GS(H_p)|Phi>|^2 or minimum dressed E_var (Nelder-Mead or CMA, about 1000-2000 Lanczos calls, CPU minutes),
  starting from the FN Hamiltonian. Pass/kill as registered. 4x6 only if 4x4 gets fidelity > 0.99.

### C6b (c6b/c6b_4x4.json): SOUND for the registered family
Re-derived independently (scratchpad g44.py, 4x4 ED): E0/N -0.528620209, stoquastised Marshall-frame E_s/N -0.911832,
g(H) = 0.3832122 (JSON 0.3832122). The metric is right for a non-Hermitian K^{-1} H K. M_s keeps the diagonal and
sets the off-diagonals to -|.|, its Perron root is real, the code takes it with eigs(cI - M_s, 'LR'), and it is
invariant under positive diagonal similarity as it should be. The rate E0 - lambda(M_s) is the walker-sign decay of
any projector that samples |Ht|. The scan is fine-grained around a = 0 (+-0.08) and g has a strict minimum at a = 0.
The buggy first run is kept and labelled. Scope: one uniform-a gate family, one bond order, 4x4. The memo's
"stoquastic non-unitary similarities" (plural, as a class) is broader than the test.

### C2-L (c2l_6x6/run_17468620, run_17468621 cx_gtax.json): SOUND kill, WEAK mechanism claim
Re-derived: RBM arm tax_B2 = E_FN(K_2 psi_fit) 5.688e-5 - E_FN(K_2 phi_0) 3.182e-7 = 5.656e-5, 34x the kill bar
1.657e-6. RBM+PP 3.414e-3. Exact store tax_B2 = 2.50e-7.
- One seed, 400 fixed steps, tau 0.05 -> 0.01 over 20 steps: identical to the G-tax design (design block in both
  JSONs), so the comparison is matched. Held-out L_ho slope over steps 350-400: RBM 0.73% per 10 steps, RBM+PP 0.11%
  (both below the registered 1% escape). Passing needs tax / 70; with tax roughly proportional to rms^2, that is about
  8x lower rms. 0.7% per 10 steps will not get there in any affordable continuation. The kill holds.
- Unequal start: G-tax refits from A, which is near its target. The positive RBM starts far from Phi (control1:
  sign agreement 0.012, L_ho 9.03 at step 0, 0.076 at 400). This handicaps C2-L, but not by 34x.
- The mechanism claim is overstated. The memo says the network "fits the pre-image's tail 4.5x better" and that "the
  layer magnifies storage errors ~30x". Per-decade fit rms (a1.loss_dev_by_decade) vs G-tax: -8 0.026 vs 0.011
  (2.4x worse), -10 0.060 vs 0.031 (2x worse), -11 0.064 vs 0.061, -12 0.057 vs 0.129, -13 0.060 vs 0.273. The fit is
  better only in decades -12 and -13 (weight ~3e-3) and worse in the bulk. So part of the 34x is fit quality (2-6x
  if tax ~ rms^2); the amplification is not isolated. Cheapest test: add log-noise with the RBM's per-decade rms to
  the exact Phi, apply the cond-30 layer, compute tax_B2, and compare with the same noise added to phi_0 directly.
  That is about 4 FN solves of ~30 s on the A40, under 0.3 GPU-h.

## Does the bottom line follow?
Partly. The tested instances are dead or unpromising: the global rules (C1), the J2 layer as add-on or trained
layer (C2, C2-T2) and storage of the pre-image at the G-tax budget (C2-L) are killed soundly. "No static sign
structure captures frustration" does not follow, for four reasons.
(i) Three kills rest on tests that could not fairly succeed. C6 is FLAWED: the min-variance parent is not a
ground-state parent, and a failed sanity check went unreported. C3 is WEAK: a control that repairs an unconverged
base, on a base that is not Hu's state, at 4x6. C4 is WEAK: 4x4 only, two starts. C5 is not finished and leans
against the coupled arm.
(ii) Almost all evidence is at 4x4 / 4x6 with 13-205-parameter bases, while the claim is about the thermodynamic
problem. The plan itself calls 4x4 anomalous.
(iii) The memo's own landscape contradicts the slogan "computed, not stored". Stored, learned signs reach rel 8.9e-5
(RBM+PP) and 2.3e-6 per site (CNN-MPS) at 6x6. What the evidence supports is narrower: no compact, interpretable,
static rule tested here beats a learned or hop-computed sign.
(iv) The "robust positive fact" is weaker than written. At cond >= 30 the pre-image is dominated by H_J2's extreme
eigenvector (lmax = 9, the ferromagnet on each sublattice; 4x6 fid(Phi, psi0) 0.41 at cond 30, 0.15 at cond 100), and
cond 100 is called trivial in the plan. 6x6 cond 30 is 7.4e-8, not "exactly" 0. V_J1 also reaches W_M = 0 at 4x4
(cond 19). The J2 specificity at 6x6 is 2-2.5x at cond 3 and ~26x at cond 10 (H_J2 0.0011 vs V_J1 0.029).
Minor: the bottom line says "seven" kills and the table lists eight.
Recommended wording: "Eight structural candidates fail their registered tests at 4x4-6x6. C6, C3 and C4 fail on
tests that do not exclude the idea, and the fixes cost under 2 CPU-h in total. Keep the loop's computed cancellation
and improve the stored part."
