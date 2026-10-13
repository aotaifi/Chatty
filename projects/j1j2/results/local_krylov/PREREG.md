# Pre-registration: checks (a0) and (a8) of the size-consistent loop (local Krylov sign step)

Written 2026-10-13, BEFORE any gain in this file was computed. Source: `projects/j1j2/DESIGN_scalable_loop_2026-10-13.md`
section 3 (a0), (a8) and 1.2-1.5. The lead (main session) approved the design's pass/fail numbers for (a0) and (a8) as
written; they are copied below. Clarifications and amendments made before the run are marked **[A1]**, **[A2]**, ...
with the reason. Code: `projects/j1j2/experiments/local_krylov/`. Results: this directory.

## Definitions (used in both checks)
- Guide psi = s a (sign s = +-1, amplitude a >= 0). Local energy E_L(x) = (H psi)(x)/psi(x).
- Bond energy e_b(x) = J_b z_i z_j/4 + [x_i != x_j] (J_b/2) s(x) s(x^b) a(x^b)/a(x); sum_b e_b(x) = E_L(x).
- **Global Krylov sign step G:** s'(x) = s(x) sgn(T - E_L(x)), one threshold T on the total local energy.
- **Local rule C_w:** for every one of the N translates W of a w x w window, dE_W(x) = sum_{b in W} (e_b(x) - e_bar_b)
  (b in W: both ends in W; e_bar_b = <e_b> under a^2, one value for NN and one for NNN bonds). W is marked if
  dE_W > theta_w. Two marked windows are adjacent if they share a bond. s'(x) = s(x) (-1)^{#connected components}.
  [A1] For square windows the reference sum_{b in W} e_bar_b is the same constant for all W, so it only shifts theta_w;
  the code thresholds the raw window sum and reports theta_w relative to e_bar (e_bar from the selection data).
- **Sign-only step:** a is unchanged, so <H>(a, s') is evaluated on the same a^2 samples without reweighting:
  E1(x) = D(x) + sum_y c_xy q(x) q(y), q = s'/s, c_xy = (J_b/2) s(x)s(y) a(y)/a(x).
- **Gain** g = (<H>(a, s') - <H>(a, s)) / N per site. Negative = energy lowered. [A2] "C gain >= 1.5 x G gain" in the
  design is read in this sign convention as g_C <= 1.5 min(g_G, 0) (C lowers the energy by at least 1.5x what G
  does; if G does not lower it, any lowering counts), and C's own gain must be resolved (g_C < -2 SE). Reason: without
  this, the criterion could pass with zero or positive energy change of the local rule when G harms.

## (a0) Decoupled 4x4 copies (calibration, not pass/fail)
- One copy: periodic 4x4 J1-J2, J2/J1 = 0.5, S^z = 0 sector (12870 states). psi0 by exact diagonalisation; check
  E0 = -8.45792335139486 (repo value) to 1e-10. Guide per copy: (|psi0|, Marshall sign). Configurations with
  |psi0| < 1e-12 have zero weight and are excluded.
- M copies, N = 16 M, M = 1 ... 64. Product state, H = sum of copy Hamiltonians.
- **Exact global-rule gain via the per-copy structure.** An edge (x, y) of copy m keeps its relative sign unless
  exactly one of e(x) + R, e(y) + R exceeds T, where R = sum of the other M-1 copies' local energies. So
  g_G(M, T) = (1/16) sum_{edges of one copy} (-2) a(x)a(y)H_xy s(x)s(y) P_{M-1}(T - max(e_x,e_y) < R <= T - min(e_x,e_y)),
  with P_{M-1} the distribution of the sum of M-1 independent copies of e under a^2.
  - M = 1: exact (R = 0). M = 2: exact CDF of e. M >= 3: CDF on a uniform grid by FFT convolution of the binned
    single-copy distribution (bin width chosen so that the M = 2 and M = 3 grid results agree with the exact ones to
    1e-3 relative in g).
  - Energy-optimal T per M: scan of T (grid + local refinement); report T* - E.
- **Local rule (regions = copies, theta = the M = 1 optimum):** g_L(M) = g_G(1) for all M (tensor product of block
  steps, design 1.2).
- **Checks (must hold before any number is used):**
  - M = 1 brute force (explicit psi', sparse H) vs formula at >= 5 values of T: |difference| <= 1e-12 per site.
  - M = 2 brute force (full 12870^2 state, H = H1 x 1 + 1 x H2) vs formula at >= 3 values of T including T*(2):
    <= 1e-10 per site (rounding over 1.7e8 terms). The local rule on the same M = 2 state (flip iff an odd number of
    copies has e(x_m) > T*(1)) must give g_G(1) to 1e-10.
  - Monte Carlo of the global rule at M = 16 and M = 64 (independent product samples, exact E1 per sample) agrees with
    the grid formula within 3 SE.
- **Output:** curve g_G(M) vs N; M_1/2 = smallest M with |g_G(M)| <= |g_G(1)|/2; N_1/2 = 16 M_1/2.
  Secondary: sd of e per copy and the per-site E_L sd of the 8x8 guides, to translate N_1/2 by matched fluctuation
  of the rest of the lattice.
- **Pre-registered reading (design):** if N_1/2 > 64, a null of G vs C at 8x8 in (a8) is expected (the local advantage
  first shows at 12x12+); if N_1/2 <= 64, 8x8 is already in the regime where the global rule is diluted.

## (a8) 8x8, existing two-hop referee data
- **Data [A3].** The loop_8x8 referee files (`ref_l8ref*.npz`) store only per-state summaries of x and its neighbours
  (E_L of the neighbours, not their bond energies), so C_w at the neighbours cannot be computed from them. The signs_8x8
  referee chunks store the guide (log a, ViT sign) on the full two-hop set {x} u N1(x) u N2(x), which gives every
  bond energy at x and at all its neighbours. Used: ws1 `/project/theorie/a/A.Otaifi/chatty_signs8/runs/`
  s8RA_17160014 (1536 samples), s8RB1_17160055 (3072), s8RB2_17163769 (3072); x ~ a^2 (beta = 1), a = |psi_P|
  (D4 x flip projected 8x8 ViT), 3 x 1024 independent chains, 7680 samples. All three are used as one referee
  (no thresholds were chosen on them for G or C before this file).
- **Starts (sign bases):** s_V (ViT's binarised sign) and Marshall.
- **Rules:** G, C_3, C_4 (primary, design); C_2 secondary (reported, not in the criteria).
  [A4] Threshold sets: G on T, C_w on theta_w, each on a fixed grid of 240 levels (60 quantile levels 0.01-0.99 and
  180 levels 1 - 10^{-u}, u in [2, 6.5], of the rule's statistic over all one-hop states: E_L for G, max_W dE_W for
  C_w) plus "no flip" (+inf). The grid is fixed from the pooled statistic before any energy is computed; it is not a
  selection. Same grid treatment for G and C.
  [A5] "C_w gain" in the criteria = C_sel: w in {3, 4} and theta chosen together on the selection fold (one
  cross-fitted rule, no best-of-two on the evaluation data). C_3 and C_4 are also reported separately.
- **Cross-fitting (primary, task instruction):** 2-fold by chain; thresholds (and w for C_sel) minimise the in-sample
  energy on one half and are evaluated on the other; 16 random half-splits (seed 0); each sample's out-of-fold
  E1(x) - E_L(x) is averaged over the splits. Secondary: leave-one-batch-out over RA/RB1/RB2 (design).
- **SE:** the larger of (i) the SE of per-chain means of the split-averaged out-of-fold per-sample change and (ii) a
  chain bootstrap (200 resamples of chains, whole procedure redone incl. threshold and w selection). Paired: C - G per
  sample.
- **Pass (local advantage visible at 8x8), design numbers with [A2]:**
  - **Marshall start:** g_C <= 1.5 min(g_G, 0), paired g_C - g_G < -2 SE, and g_C < -2 SE.
  - **s_V start:** cross-fitted g_C < 0 at > 2 SE (for reference, G on the loop_8x8 data: +2.1(2.6)e-6).
  - If (a0) gives N_1/2 > 64, a null is reported as "expected at this N", not as a failure.
  - Otherwise the result is "inconclusive (cause: X)" with the cause diagnosed (referee SE, merging, threshold noise);
    "failed" only with a conceptual limit (CLAUDE.md rule 1).
- **Diagnostics (reported, not criteria):** flipped a^2 fraction of x and of neighbours; component-count histogram at
  the selected theta; agreement of the flip decision C_3 vs C_4; G and C in-sample vs cross-fitted gains; check
  sum_b e_b = E_L (<= 1e-9). **Locality premise:** for the samples in the top 1% (and 0.1%) of E_L, the fraction of
  the total excess sum_b (e_b - e_bar_b) that lies in the best single 4x4 window (uniform spread: 42/256 = 0.16).

## Budget
CPU only (Mac + ws1 file transfer), about one day of work.
