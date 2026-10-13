# Design: the size-consistent FN/Krylov loop (local sign step + FN amplitude), 2026-10-13

Opus design partner, after the DECISION 2026-10-13 (RESEARCH_MAP 3p). Nothing in this file has been run. The
pass/fail numbers in section 3 are proposals: they become binding once the PI approves them and they are copied into the
experiment READMEs before the runs.
Inputs read: RESEARCH_MAP 3a-3p; compress_6x6 round 4 and step size (S2-1, S2-2); loop_8x8 FU1-FU3; signs_8x8;
fn_calibration_6x6; STATUS.md 2026-09-28..10-01 (earlier walker density-ratio and walker-MLE routes).
Codex consult: `ws1:~/chatty_admin/consult_2026-10-13_codex_answer.md` holds only a CLI error ("Not inside a trusted
directory"), so there is no consultant answer to weigh.

Notation:
- psi(x) = s(x) a(x). Bonds b = (i, j) have J_b = 1 (NN) or 0.5 (NNN). x^b is x with the spins on b swapped
  (valid only if they are antiparallel).
- E_L(x) = (H psi)(x)/psi(x). Global energy-optimal Lanczos step: alpha_g = -0.157 at 6x6, -0.161 at 8x8.

## 1. Local Krylov sign step

**1.1 Bond energies.** For each bond b,

    e_b(x) = (H_b psi)(x)/psi(x) = J_b z_i z_j / 4 + [x_i != x_j] (J_b/2) s(x) s(x^b) a(x^b)/a(x),

so that sum_b e_b(x) = E_L(x) exactly. It needs the same hop as E_L.
- Kept (FN-allowed) bond: s(x)s(x^b) = -1, which gives a negative contribution.
- Violating bond: s(x)s(x^b) = +1. These sum to the wall term V.
- Reference: e_bar_b = <e_b> under a^2. Only two values exist (NN, NNN) by symmetry; they are estimated once per guide.
- Region excess: dE_R(x) = sum_{b in R} (e_b(x) - e_bar_b), where "b in R" means both ends of b lie in R.

**1.2 Derivation on decoupled blocks (the definition of "correct").**
Take H = sum_m H_m and psi = prod_m psi_m. The size-consistent Krylov step is the tensor product of the block steps:

    psi'(x) = prod_m [(1 + alpha (H_m - E_m)) psi_m](x_m) = psi(x) prod_m f_m(x),   f_m = 1 + alpha dE_m(x_m).

- Its sign is s'(x) = s(x) (-1)^{#marked blocks}, where a block is marked if f_m < 0, i.e. if dE_m > theta = 1/|alpha|.
- So **"flip if an odd number of regions is marked" is exactly the sign of the product of local Lanczos factors on a
  partition.** The PI's two candidate combiners are the same object on a partition.
- The energy per site after the step equals the single-block value for every M. The energy-optimal alpha does not depend
  on M, because E(alpha) = sum_m E_m(alpha).
- The global rule s' = s sgn(T - E_L), with T = E + 1/|alpha_g|, decides block m's flip from
  dE_m + sum_{m' != m} dE_m'. The second term has spread sqrt(M) sigma_block.
  - The global rule therefore keeps only defects whose excess exceeds about sqrt(M) sigma_block.
  - Its per-site gain goes to zero with M.
  - This matches the 8x8 data: energy-optimal T - E = +11.9, no a^2 sample flips, null gain +2.1(2.6)e-6/site.
  - The same mechanism makes p fixed Lanczos steps non-size-consistent (Becca, Hu, Iqbal, Parola, Poilblanc, Sorella,
    arXiv:1412.2656).
- The closest prior art for a size-consistent step that changes signs is Sorella's generalized Lanczos with r > 0
  (PRB 64, 024512 (2001)). It modifies the reference Green function in a fixed-node-like way, and it is global and per
  configuration. Our step is local (region-factorised); we must cite it and position against it.

**1.3 Combining regions on a coupled lattice.** There are three options.

- **Fixed tiling + parity.** Exact on blocks.
  - It is not translation-invariant, so it cannot be used on the 6x6 orbit tables.
  - A defect on a tile boundary splits its excess over two tiles. That gives 0 or 2 marks (no flip), or 1 mark if only
    one half passes theta.
- **Overlapping windows (all N translates of a w x w window) + parity, or the product over all windows.** Wrong.
  - One local defect marks every window that contains it, about w^2 windows, so the parity becomes arbitrary.
  - A partition-of-unity weighting fixes the double counting but dilutes each defect by the overlap factor.
- **Recommended: connected components of marked windows (C_w).**
  - Compute dE_W for all N translates of a w x w window. Mark W if dE_W > theta_w. Two marked windows are adjacent if
    they share a bond.
  - Flip x iff the number of components is odd. One defect gives one component, whatever its position.
  - The rule is invariant under translations, D4 and spin flip, so it is a function on orbit representatives (needed
    for the exact 6x6 lab).
  - On decoupled blocks with windows inside blocks it reduces to 1.2 as long as defects are isolated.
  - It is the same "syndrome counting" as decoding local errors. Linked-cluster expansions (e.g. Lanczos-boosted NLCE,
    arXiv:1810.06202) get size consistency the same way.
- **Amplitude (optional).** prod_{partition} |f_R|, clipped as in Lc_0.3, is the size-consistent version of the
  clipped-Lanczos target. It is one hop, explicit, and fits the existing write-back machinery. Keep it as the DMC-free
  fallback arm (section 3a).

**1.4 Threshold.** Choose one theta_w per window size (square windows give a single region type).
- A sign-only step leaves a^2 unchanged. So E(theta) is evaluated on fixed samples with no reweighting: exactly at 6x6,
  and at 8x8 by cross-fitting (select on some batches, evaluate on the others).
- The family contains theta -> infinity (no flip), so the selected step can only lower the in-sample energy.
- Amplitudes come from the FN step that follows, which shrinks the amplitude near new nodes. That is the "damped flip"
  that cx_l1bump asked for: a hard flip with the old amplitude makes the Lanczos polish worse
  (3.6e-5 vs 2.55e-5).

**1.5 Where the rule can fail on a correlated lattice.** Each failure mode has the quantity that monitors it.

- (i) **The exact ground state is not a fixed point.** dE_R fluctuates in the exact GS (var H_R > 0, boundary law), so
  rare configurations with dE_R > theta get flipped. Monitor: flipped a^2 weight and energy rise on (|psi0|, s0).
- (ii) **Merging.** Two defects closer than w form one component, so the parity is wrong. Monitor: distribution of the
  component count per configuration, and agreement of the flip decision between w and w+1.
- (iii) **Extended or non-local sign errors** (strings longer than w, wrong symmetry sector). A long string still gives
  one component, which is fine. A sector error gives no local syndrome; the global rule also traps there (bad_start).
  Monitor: the spatial extent of the excess in high-E_L samples (8x8 data).
- (iv) **Amplitude errors create false excess.** This is the same for global and local. Monitor: false flips from
  (|psi_P|, s0).
- (v) **Threshold noise at 8x8.** The energy curve is flat: the global T varied from -3.9 to -66.7 across half-sample
  splits. Report cross-fitted gains only.
- **w_s is not a size-consistent figure of merit.** With any fixed per-region error rate, the configuration-level
  wrong-sign weight goes to 1/2 at large N, while the energy per site stays fine. Report the intensive edge version
  w_e = a^2-weighted fraction of edges with a wrong relative sign, alongside w_s.

## 2. FN-walker amplitude

**2.1 What the walkers carry.** Lattice FN-DMC with guide (a_k, s) samples f = a_k phi, with phi = phi_FN[a_k, s].
There is no time-step bias in the distribution: 1 - tau(H_FN - E) has the eigenvectors of H_FN. To turn this into a
network there are three routes.
- **Density ratio (classifier).** The logit is log(phi/a_k), which is extensive.
  - The two distributions separate as N grows, so the logits are overconfident.
  - Tested 2026-09-30 at 8x8: it worsened E_FN by +4.5 total and the replicas disagreed. Not size-consistent as a
    learning signal.
- **Walker MLE.** It fits a_new^2 to f, so its optimum is sqrt(a_k phi), a geometric half-step that is 2x slower.
  - Tested 2026-10-01 at 8x8: iteration 1 was positive, iteration 2 worsened E_FN by +0.073 total.
  - Cause pinned: only 0.78% of the one-hop edge targets used by FN/K1 are in the sampled support. The unseen
    neighbour ratios moved 7-20x more than the seen ones.
- **Conditional NCE** (Ceylan & Gutmann, ICML 2018; noise = one-hop neighbour). It is consistent and local, and it
  constrains every edge incident to a walker.
  - But the labels are Bernoulli: an edge log-ratio to delta = 0.003 needs about 4/delta^2 = 4.4e5 pairs per effective
    degree of freedom.
  - Affordable at 6x6, not at 8x8.

**Proposal: the walkers supply the sampling distribution and E_FN. The labels come from the FN eigen-equation itself.**
Rao-Blackwellising the walker balance over the known propagator gives back exactly that equation. So it is the
zero-noise limit of "fitting to walkers". For a_{k+1} = a_k e^u define

    D(x) = d_FN(x) - E_FN     (d_FN = H_xx + V(x), the FN diagonal including the wall term of guide a_k)
    S_u(x) = sum_{kept y} |H_xy| (a_k(y)/a_k(x)) e^{u(y) - u(x)}
    r_u(x) = D(x) - S_u(x)    (additive form; corrected per the consultant, section 6: the normalised forms
                               1 - S_u/D and log D - log S_u do not add over independent blocks)

and minimise the sum of r_u(x)^2 over walker snapshots x (with all their kept edges), using robust_fit.

Properties:
- **Exact target.** r = 0 everywhere iff a_k e^u = phi, by Perron-Frobenius on the connected allowed graph
  (fncal_pockets: connected at 6x6). D > 0 holds on the support, because D(x) = S_phi(x) > 0.
- **Size-consistent.** Residuals add over decoupled blocks, so the centred loss decouples.
- **Zero label noise.** Only generalisation error remains.
- **Constrains every edge the next FN/K1 step uses**, which removes the walker-MLE failure mechanism.
- **The SI target is one Jacobi sweep of this same equation** (fixed neighbours, E_g instead of E_FN, the 1+
  regularisers). Its relative step is O(1/N), which is the non-size-consistent 1/(1+W). Solving the equation fully is
  size-consistent.
- **Monotone if exact** (ten Haaf 1995 plus the AM-GM identity). On every violating pair,
  <psi|H_FN - H|psi> = |H_xy| (sqrt(t) psi_x - psi_y/sqrt(t))^2 >= 0 with t = a(y)/a(x). Hence
  VMC(phi_FN[a_k]) <= E_FN[a_k] <= VMC(a_k), and so E_FN[a_{k+1}] <= VMC(a_{k+1}) <= E_FN[a_k].
  Any rise along the loop is therefore fit error, and an acceptance gate catches it.

**2.2 Noise, bias, sizes.**
- **Population control.** The energy bias is <= 0.2e-6/site at M = 512 (b/M fit). A distortion of the snapshot
  distribution enters only the sample weights, not the labels.
- **Snapshots must be weighted by beta-duration.** The step average biased energies by -11..+25e-6.
- **Decorrelation.** Thin by Delta beta >= the autocorrelation time, measured at 6x6.
- **E_FN error.** It is about 3e-5/site from the harvest run at 8x8. The sensitivity is measured in check (b). If it
  matters, treat E as a free parameter (centred residual = fixed-sign variance minimisation).
- **Size.** About 3e4 decorrelated snapshots, i.e. 3e4 summed equations (one per snapshot, each a sum over ~145
  edges; not 4e6 independent constraints, errors on edges can compensate). The SI fits used 1e4-6e4 samples with
  their neighbours.

**2.3 Cost, and how this composes with the re-base.** The residual at x needs a_k on the one hop of x. Every DMC step
computes that already, so it is free if stored. It also needs u at the neighbours.

Feature sets for u:
- **F2 = round-4/S2 features** (one exact hop of psi_P, sign-split). u(y) then costs 2 psi_P hops, and the DMC guide
  costs 2 hops per walker-step: about 1e4 projected-ViT evaluations, or 2 s at 5e3/s.
  - With 3e4 snapshots x 10-50 steps that is 170-800 A40-h per iteration. **Infeasible at 8x8.**
- **F1 = hop-free features:**
  - spins and log a_P(x) (free);
  - a bond-output surrogate net that gives all ~145 log a_P ratios in one pass, from which V-hat, W-hat and
    E_L-hat follow;
  - the relative signs per bond from the sign net (cheap CNN).
  - DMC then costs 1 ViT hop per walker-step (~0.03 s), i.e. 2.5-12 A40-h per iteration.
  - The surrogate's error enters only as an input feature, not as the target. Its precision failed in P2 when it was
    used as the target.

Re-base:
- **u is fitted directly in the stacked parametrisation**, G_{k+1} = G_k + h_k, against the residual of the stacked
  state. There is no separate write-back.
- This removes the S2-2 failure (re-base retention 0.69 / 0.33 / 0.09, caused by write-back/re-base feature mismatch),
  because there is only one fit.
- Reused from round 4: clipping, re-seed on collapse, held-out acceptance, blow-up guard.
- Each h_j uses its own sign split s_{j+1}. The small-net cost is linear in k; the hop cost is constant.

**Literature check** (quick web search, to be extended before any claim):
- Neural-network trial functions inside FN-DMC are known (Ren et al., Nat. Commun. 2023).
- So are NQS learned from samples by MLE (Torlai et al. 2018; data-enhanced VMC) and supervised fits of (1 - tau H)psi
  (Kochkov & Clark 2018).
- I found no work that fits an NQS to the lattice FN eigen-residual on DMC walkers, and none with a local
  (region-parity) sign step.

## 3. Minimal decisive checks (exact where possible; numbers to be pre-registered)

**(a0) Decoupled 4x4 copies (CPU, minutes).** Use exact single-copy tables with the guide (|psi0^4x4|, Marshall).
- For M = 1..64 copies (N up to 1024), compute the global rule's exact per-site gain, scanning T.
- It needs only the CDF of the sum of the other M-1 copies' E_L (FFT convolution): a pair (x_m, y_m) keeps its relative
  sign with probability 1 - 2[F(T - min E) - F(T - max E)].
- Unit test: the local rule (regions = blocks) must reproduce the M = 1 gain to 1e-12 at every M.
- Output: M_1/2, the M at which the global gain halves. This is a calibration, not a pass/fail.
  - If N_1/2 = 16 M_1/2 > 64, a null at 8x8 in (a8) is expected. The local advantage would then first show at
    12x12 and beyond.

**(a) 6x6 exact, sign-only, amplitude fixed (cx_lib Lab, orbit tables).**
- Starts:
  - S1 = (|psi_P|, s_P);
  - S2 = (|psi_P|, Marshall);
  - S3 = (|psi0|, Marshall);
  - S0 = (|psi0|, s0), the false-flip test.
- Rules: G (global, T scan) and C_w for w = 2, 3, 4 (theta scan). Each is applied once, and also iterated 3 times
  (S2, S3).
- Report: <H>, w_s, w_e, flipped weight, component-count histogram.
- **Pass (no harm at the size where G still works):**
  - the best C_w reaches >= 0.80 of G's one-step energy gain and >= 0.80 of its w_e reduction, on both S2 and S3;
  - after 3 steps, error <= 1.25 x G's;
  - on S0, at the theta chosen on S3: flipped a^2 weight <= 1e-4 and energy rise <= 1e-6/site.
- Fallback arm (secondary): C_w sign + clipped |prod f| amplitude vs global clipped Lanczos from S1 (2.62e-5). Bar:
  <= 4.7e-5, which is 0.8 of that gain.

**(a8) 8x8, existing two-hop referee batches (loop_8x8 / s8_thresh data, CPU only).** These hold the guide on
{x} u N1 u N2, so s' at x and at all neighbours and the exact new E_L are available.
- Starts: s_V and Marshall. Rules G vs C_3 and C_4. Thresholds by leave-one-batch-out. SE from paired per-chain means.
- **Pass (local advantage visible at 8x8):**
  - from Marshall, C_w gain >= 1.5 x G gain, with the paired difference at > 2 SE;
  - from s_V, the cross-fitted C_w gain is < 0 at > 2 SE (G: +2.1(2.6)e-6).
- (Withdrawn after the consultant's critique, section 6: a product-block N_1/2 does not predict the coupled crossover,
  so an 8x8 null is not 'expected'.)
- Also report what fraction of the high-E_L excess lies in one 4x4 window. This tests the "local defect" premise.

**(b) 6x6 amplitude from FN walkers vs the exact phi (sec.fn_solve).**
- Guides: S1 (exact: E_FN 1.02e-4, <H>(phi, s_P) 8.41e-5 from 1.319e-4) and (|psi_P|, s from C_w), i.e. realistic
  iteration 1.
- Walkers: fncal engine, M = 512, beta-time-weighted snapshots after beta = 0.8. Two independent walker sets.
- Arms:
  - RES-W (residual on walkers);
  - RES-V (same loss on a^2 VMC samples; tells whether the walkers matter);
  - CNCE (noise reference);
  - ORACLE (edge regression on the exact log(phi/a); representation ceiling);
  - SI (current target; realistic reference 0.734).
- Feature sets F2 and F1. n = 1e4, 3e4 and 1e5 snapshots. E_FN fixed vs free, and with an injected +-1e-3 total
  error.
- Metric: exact frozen-FN capture C = (E_f[a] - E_f[a e^u]) / (E_f[a] - E_FN), plus <H>.
- **Pass:**
  - RES-W/F2 at n = 3e4: C >= 0.85;
  - RES-W/F1: C >= 0.75 (then 8x8 uses F1);
  - two walker sets agree to held-out edge rms <= 0.003.
- **Readings:**
  - RES-V within 0.03 of RES-W: drop the walker harvest and keep DMC as the E_FN referee only (a cost win).
  - F1 < 0.75: 8x8 DMC must use a cheaper guide (the unprojected ViT is 16x cheaper), or the fallback arm.

**(c) 8x8 loop (after (a), (a8), (b) and a throughput gate).**
- **Gate.** Measured walker-step cost with the F1 guide must project to <= 10 H100-h per iteration; otherwise the run
  is "inconclusive (cause: cost)" before any start.
- **Iterations k = 0..3, starting from (|psi_P|, s_V):**
  1. C_w labels from one hop of (a_k, s_k) on tempered samples and their neighbours, trained into a sign net
     (signs_8x8 machinery). This gives s_{k+1}.
  2. FN-DMC with (a_k, s_{k+1}), harvesting 3e4 snapshots and their one-hop guide values.
  3. Fit h_k by RES-W/F1 with acceptance. This gives a_{k+1}.
  4. Paired VMC referee (1 hop, ~1.5e4 samples, SE ~5e-6).
- **Final:** + 1 Lanczos step, and calibrated FN-DMC (~50-100 populations, SE ~1e-5).
- **Budget:** <= 60 H100-h.
- **Pass:**
  - (i) VMC <H> is non-increasing within 2 sigma at every k;
  - (ii) loop guide (no hop) < RBM+PP -0.498886(1) at 2 sigma;
  - (iii) loop + 1 Lanczos < projected ViT + 1 Lanczos (-0.498953(16) combined; conservative -0.498948(23)), paired on
    the same batches, at 2 sigma;
  - (iv) it beats the same F1 net trained by fixed-sign minSR VMC with the same signs at equal GPU-h, by >= 1.5x in gain
    or at 2 sigma (rule 5).
  - If (iii) fails while (i) and (ii) pass, the loop is a guide improver but not yet better than the polish: report
    that, and do not drop it.

## 4. Risks

| # | What breaks | Size (measured or estimated) | Monitor | Test that settles it |
|---|---|---|---|---|
| R1 | DMC cost with hop-feature guides | F2: ~2 s per walker-step, 170-800 A40-h per iteration; F1: 2.5-12 | evaluations/s, steps per decorrelated snapshot | throughput gate before (c); (b) F1 vs F2 |
| R2 | Unseen neighbour ratios drift (walker-MLE failure, 0.78% support) | iteration 2: +0.073 total E_FN | held-out edge residual; edge-ratio change seen vs unseen | (b) RES-W vs CNCE; (c) acceptance gate |
| R3 | Stiff residual fit / slow long-wavelength modes | lambda_max(F) 1.6e6 (itfit) | capture vs steps; gap to ORACLE | (b) ORACLE vs RES-W; log-form residual; Adam vs Gauss-Newton |
| R4 | Exact GS not a fixed point (false flips) | unknown; bar 1e-4 weight, 1e-6/site | flipped weight per iteration | (a) S0 |
| R5 | Component merging/splitting | unknown | component-count histogram; w vs w+1 agreement | (a), (a8) |
| R6 | 8x8 threshold noise | global T varied -3.9 to -66.7 over splits | cross-fitted vs in-sample gain | (a8) leave-one-batch-out |
| R7 | Walker distribution bias / beta weighting | energy <= 0.2e-6 at M = 512; step average -11..+25e-6 | two walker sets; beta-weighted harvest | (b) replica agreement |
| R8 | Fit error breaks FN monotonicity | iteration 2 of walker-MLE | VMC <H> per iteration | (c)(i); exact chain is monotone (2.1) |
| R9 | C_w labels not learnable by the sign net at 8x8 | signs_8x8 pending | held-out label error, disagreement on neighbours | signs_8x8 arm, then C_w labels |
| R10 | E_FN error in the residual | ~3e-5/site per harvest at 8x8 | fixed- vs free-E fit | (b) injected +-1e-3 |
| R11 | Global rule not yet diluted at 8x8, so no visible local gain | alpha_g flat 36 -> 64 | M_1/2 | (a0) |

**Recommended first check:** (a0) together with (a8). Both are CPU-only on existing data, about one day of work.
- (a0) says at which N the global rule stops working.
- (a8) says whether the local rule already beats it at 8x8.
- (a) and (b) on 6x6 follow on ws1 (<= 6 GPU-h). (c) waits for all of them.

## 5. The fold-back (residual re-base) failure: diagnosis and redesign (added 2026-10-13, after D1/D2)

**Terms.**
- *Re-base*: storing each iteration's increment h_k as a small net whose inputs are psi_P's one-hop scalars, so that
  the guide costs one hop of psi_P at every k.
- *Retention*: the share of the iteration's exact gain that survives this step.
- *Light cone*: the set of configurations whose psi_P values the exact k-th iterate depends on.

**5.1 The lead's reading, and what it misses.** The dependence chain is right:
- d_k = F(E_L of G_k);
- E_L of G_k needs G_k at all neighbours x^b, hence h_j(x^b), hence psi_P's one-hop features at x^b, i.e. a second
  hop.

Two refinements:
- **A fixed-radius representation never closes.** If the stored guide uses psi_P within R hops, its local energy, and
  so the next target, needs R + 1. Any fixed-R re-base must predict the outer shell from the inner ones. The light
  cone can only make that shell cheap if it is predictable (local); it cannot remove it.
- **The exact light cone is not area-law.** k iterations need psi_P on connected clusters of k swapped bonds, about
  N_b c^(k-1) configurations, with c ~ 7-15 nearby valid bonds. That is constant in N but exponential in k. It becomes
  area-like only if multi-bond log-ratios are pair-additive beyond a radius, which D2a tests.

**5.2 Measured (results/compress_6x6, "Fold-back diagnostics"; 1.2 slice-h).**
- **Information limit confirmed.**
  - Re-base of the exact k = 3 target on psi_P's one hop: retention 0.02. With 3x the steps: 0.02.
  - With the target normalised to unit rms: 0.33. This is a real scale effect (Adam does not move on a target of
    rms 0.0065); normalising should be the default from now on.
  - Same net plus G_3's own local energy as one extra input: **0.99**.
- **Light cone not viable at 6x6.**
  - Every partial light-cone level gives a worse target than ignoring the neighbours: d error 0.027-0.11 against
    0.017.
- **The cause is physical, not the network.**
  - The exact 6x6 ground state has pair interactions rms I = 1.69 / 1.02 / 0.80 at bond distance 1 / 2 / 2.8, with
    5-19% sign non-additivity.
  - psi_P agrees to 2%.
- So at 6x6 the outer shell is neither cheap nor predictable. Whether I(b, b') decays at 8x8+ near J2 = 0.5 is open.
  Measuring it needs about 0.5 GPU-h of new 8x8 two-hop amplitudes; the stored referee data hold local energies, not
  x^{bb'} amplitudes.

**5.3 Redesign: rolling re-anchor into the base network.**
- A network that reads the configuration directly has no hop radius. A fixed-radius increment stack always has the
  one-shell gap.
- So the fold-back should end in the base network's weights. Each iteration fine-tunes the base NQS, warm-started,
  on the normalised increment only:
  - edge loss in the target metric;
  - sampling by the increment's Dirichlet weight (the round-3 lesson);
  - minSR/natural gradient with a trust region, Adam as the control (rule 2);
  - held-out acceptance and the blow-up guard.
- The next target is then computed from one hop of the fine-tuned network itself. There is no stack and no
  recursion.
- Cost per iteration at 8x8 is constant in k: the guide value costs 0 hops, E_L 1 hop, and an edge-loss training
  sample 2 hops (as in loop_8x8).
- **Known risk (P2, 2026-10-09):** re-distilling the whole accumulated correction into the warm ViT failed on
  precision (<= 10% kept; tolerance 0.003 bond-rms). That attempt used Adam / damped Gauss-Newton on the whole
  correction, with plain sampling.
  - What is untested is the increment-only fine-tune with increment-weighted sampling, a normalised target and
    natural gradient. These are the three fixes that rescued the small-net re-base.
- **Decisive test, to be pre-registered (6x6 exact tables, symmetrised ViT, k = 1..3, 2 seeds):**
  - **Pass:** iteration share >= 0.70 at each k (the stage-2 bar). Reference: the stage-2 re-base gives
    0.84 / 0.66 / 0.31, and the k = 1 target is exact (2.62e-5).
  - **Fail with a diagnosed cause:** share < 0.3 at k = 1 under both optimisers, after normalisation and increment
    sampling, means the ViT cannot hold 0.003-precision updates. Then cap the depth (guide <= 2 hops + Lanczos
    polish; at 6x6 this ties psi_P + 2 Lanczos steps), and test a local base architecture where the 8x8 I(b, b')
    allows it.
- Budget: <= 4 slice-h.

**5.4 How this fits sections 1-2.**
- The FN-walker residual fit (2.1) has the same one-shell gap if it is stored as increments on psi_P features: the FN
  solution depends on the guide beyond any fixed radius.
- Re-anchoring removes the gap. The residual is fit directly by fine-tuning the base network (fixed-sign, on walkers
  and their edges). The F1/F2 feature question (2.3) disappears.
- The FN-DMC guide becomes the network itself, at 1 hop per walker-step. That also removes risk R1 (170-800 A40-h
  per iteration for hop-feature guides).
- The local Krylov sign step is unchanged: a stored sign net + one hop. Its hop is a final-evaluation cost only.
- **Ordering:**
  1. re-anchor test (5.3);
  2. (a0)/(a8) sign checks (CPU), in parallel;
  3. check (b) with re-anchored amplitudes instead of F1/F2 increments;
  4. 8x8.

**5.5 Re-anchor test result (2026-10-13, results/compress_6x6 "Rolling re-anchor test"; 2.55 slice-h).**
- FAIL at k = 1. All arms worsen the energy: minSR -0.14 / -0.12, Adam -0.10.
  - The fits halve the held-out edge loss and match the increment to 0.003-0.006 where a^2 >= 1e-8.
  - The errors grow to 0.1-0.9 below a^2 ~ 1e-10.
  - Splicing the fit with the exact target below a^2 = 1e-9 (7.7% of the weight) recovers 0.91 of the gain.
- So the gain of the step lives in the tail around the new nodes, and a spin-only network cannot follow it there.
  The hop-feature nets could (they compute E_L), but they carry the one-shell gap of 5.1.
- Both representations fail on opposite sides:
  - spin-only networks lack tail precision (here, writeback_tail, P2);
  - one-hop-feature nets lack the outer shell (D1/D2).
- **Pre-registered fallback, now in force:** cap the loop depth.
  - The guide stays at <= 2 hops + Lanczos polish.
  - At 6x6 the stage-2 k = 4 guide gives 2.0e-5 at one hop (vs psi_P + 1 Lanczos 2.55e-5). At two hops it gives
    0.92e-5 (vs psi_P + 2 Lanczos 0.86e-5).
- **The one design not yet tested, combining both sides:** re-anchor the ViT on the bulk every iteration, plus ONE
  small tail head on one-hop features of the CURRENT ViT. The head is refit each iteration, not stacked, so the hop
  cost stays constant.
  - It needs only the tail part of the increment from one-hop information. Its one-shell gap is limited to 7.7% of
    the weight.
  - Exact 6x6 test: share >= 0.7 at k = 1..3 with the same machinery. About 3 slice-h.
  - This needs a PI decision; it was not part of the approved test.


## 6. Consultant critique and response (Codex, results/consult/codex_2026-10-13_answer.md)

**(1) Lanczos dilution is M^(-1/2), not 1/M.** For M independent blocks, the optimised one-step Lanczos gain per
site falls as M^(-1/2); a fixed alpha gives M^(-1).
- Accepted. Section 1.2 only claims that the gain goes to zero. The "~1/M" in the consult post is corrected to
  M^(-1/2) for optimised Lanczos.
- The slow M^(-1/2) fits the measured fixed fraction from 16 to 64 sites. It makes Lanczos a stronger control at
  accessible sizes.
- The scaling of the sign-only threshold is not known; (a0) measures it.

**(2) Component parity is not derived.** Two negative factors whose marked windows merge give one component, which
reverses the parity. Also, e_b depends on distant spins through the amplitude ratios.
- Accepted. Only partition parity is derived (1.2). C_w is a heuristic.
- D2a/FB-1 support the second point: at 6x6 the exact ground state's pair interactions are O(1) out to the largest
  distance, so geometric windows do not localise the information.
- Added tests:
  - deliberately inserted one- and two-defect sign patterns (on |psi0|, in 4x4 copies and in 6x6), with the
    separation swept through the merging distance, scored by energy-weighted relative-sign errors;
  - fixed-tiling partition parity kept as the controlled baseline on unreduced configurations (4x4 full space and
    copies), not dropped for orbit-table convenience;
  - agreement between w and w+1 removed as evidence.

**(3) FN residual loss.**
- (a) D - S_u is additive over blocks; the normalised forms are not, and they hide a local defect at O(1/M).
  Accepted and corrected in 2.1: use D - S_u.
- (b) One snapshot gives one summed equation, not one per edge. Accepted and corrected in 2.2. Mitigations: also
  impose the equation at sampled neighbours (two hops), and keep a per-edge anchor (CNCE) as a diagnostic.
- (c) Deterministic residuals still carry sampling, E_FN, approximation and optimisation errors. Accepted.
- (d) The walker-weighted centred residual is not the quantum variance. Accepted; it is renamed "residual least
  squares under the walker measure".

**(4) Decoupled copies are an algebraic test, not a crossover predictor.** Accepted.
- The "8x8 null expected" reading is withdrawn.
- Added: couple two 4x4 blocks continuously into a 4x8 with inter-block coupling lambda in [0, 1]. The exact
  symmetric sector is about 1e7 orbits, the same scale as 6x6.
- Re-optimise every method at each lambda and M. Report global sign-only, full Lanczos and compressed updates
  separately.

**(5) Other points, all accepted.**
- "Non-increasing within 2 sigma" allows cumulative drift: use an untouched final evaluation set, and select
  thresholds only on separate data.
- Calibrate injected E_FN errors to each harvest's actual SE.
- Validate the sign net separately from its labels.
- Report structure factors and dimer correlations alongside energies.
- Run sign-only / amplitude-only / full-loop ablations.

**Where the evidence now stands (6x6, this session).**
- Compression is the central problem; the consultant agrees.
- Fixed-radius hop-feature increments are information-limited (D1: 0.02 vs 0.99).
- Spin-only re-anchoring is tail-limited (share -0.10 to -0.14; 0.91 recoverable with the tail exact).
- A guide's own lattice E_FN is amplitude-dominated at the nodes (sign-channel check).
- The approved next test is the hybrid: re-anchored ViT plus one non-stacked one-hop correction net.

**5.6 Hybrid test result (2026-10-13; results/compress_6x6 "Hybrid re-anchor test"; 1.0 slice-h).**
- k = 1 passes (share 0.96; own E_FN 2.17e-5 vs stage-2 2.94e-5). k = 2 fails (0.23). Pre-registered verdict: FAIL.
- The cause is pinned (HY-1): with the guide's own local energy as one extra input (a second hop), the same
  correction reaches 0.92 at k = 2.
- So every constant-hop variant tested at 6x6 meets the same one-shell gap from iteration 2 on:
  - psi_P one-hop re-base (D1: 0.02 -> 0.99 with the information);
  - spin-only ViT (RA: tail-limited);
  - ViT + one-hop correction (HY-1: 0.23 -> 0.92).
- The k-th update needs E_L of the k-th guide, which needs one more hop than the guide uses. At 6x6 that outer shell
  is neither local (D2) nor predictable from the inner ones.
- **Consequence.** A loop iteration costs one extra hop, as a Lanczos step does. The constant-hop claim does not
  hold at 6x6.
- **What survives. Correction:** the hybrid k = 1 guide is NOT the best one-hop guide.
  - psi_P + 1 Lanczos is better at one hop (<H> 2.55e-5, E_FN 1.89e-5 vs 3.01e-5 / 2.17e-5).
  - So is the stage-2 k = 4 guide (<H> 2.0e-5).
  - At two hops: hybrid + 1 Lanczos 1.40e-5 vs psi_P + 2 Lanczos 8.6e-6.
- So the fallback (cap the depth, Lanczos polish) remains in force.

**5.7 Lanczos-wrapped network (lead proposal), pre-test W0: NOT SOUND, no training run.**
- Proposal: G = (1 + beta (H - E)) N with N a spin-only network, evaluated at one hop.
- Exactly on the k = 1 tables: with N exact on common configurations and the ViT fit in the tail (a^2 < 1e-9), the
  wrapped state keeps 4-7% of the gain. The reverse splice keeps 91%.
- In the wrapped exact state the inherited term beta sum_y H_xy N(y) dominates < 1e-4 of the weight down to 1e-12.
  It only matters where a step creates new nodes; afterwards those configurations belong to N's own tail.
- Prior art for optimising through Lanczos factors (by energy, not by regression):
  - Sorella's generalized Lanczos (PRB 64, 024512, 2001);
  - Hu, Becca, Parola, Sorella 2013 and Chen et al. 2022 apply Lanczos steps to an optimised state without training
    through them.
- The wrapper's linear inverse is well conditioned in the spectrum (1 + beta (eps - E) > 0 for beta = 0.108). The
  ill-conditioning is only in log space, at the new nodes.

**5.8 Route B (spin-only tail representation) and locality vs J2 (2026-10-13).**
- **Tail net: inconclusive, cause located.**
  - Spin-only tail nets on top of the re-anchored ViT reduce their held-out loss by 4-5% (one-hop inputs: 88%) and
    worsen the guide (share -0.23 / -0.25 at C = 64 / 32).
  - The tail residual has an inherited, one-hop structure (log-sum over the neighbours' amplitudes) that spins alone
    do not express.
  - Input-gradient features are useless: a single exchange move is non-perturbative for the ViT (linearised
    log-ratio error 3.2 rms, correlation with the exact E_L about 0).
  - The only representation that works is one that includes the hop.
- **Locality vs J2 (4x6 exact).** The double-swap interaction meets the locality bar at J2 = 0, is close at 0.3, and
  fails at 0.4-0.6 (worst at 0.5).
  - At 8x8, J2 = 0.5 (results/locality_8x8) it is NOT local either.
- **Consequence for the loop.** At J2 ~ 0.5 every constant-cost fold-back tested fails for one physical reason: the
  ground state's log-amplitude is non-additive at all accessible distances. So each loop iteration costs one extra
  hop, as a Lanczos step does.
- A light-cone loop could be viable in the weakly frustrated Neel regime (J2 <~ 0.3). That regime is of limited
  interest (weak sign problem), but it is a clean scientific statement: the fold-back locality tracks the
  frustration.
