# Storing the fixed-node state in a network: the three issues (PI + ansatz lead, 2026-10-27)

Equations: equation board https://claude.ai/artifact/LwUGJoTxGJku3k6mTVgMVP (block numbers below).
Derivation and 4x4 check: ../../handoffs/2026-10-27-fn-error-weights.md. Tried ideas: TRIED_IDEAS.md (row ids below).

| # | issue | cause (equation) | what was tried | status |
|---|---|---|---|---|
| 1 | log problem | output ln|psi_T| and loss in ln weigh errors by psi^2 (13); FN energy weighs pairs across the node by the big side, |H| psi(y)^2 (r - r*)^2 / r (5); a smooth log net cannot dip ln(1/r*) in one swap (7) | edge/pair losses with log-output nets: B13 (4x4 66% vs 46%), B30 (6x6, does not generalise), B36 (weak) | fix on paper: e^J Pf output + pair loss L_fix (15). L_fix needs phi pointwise (exact tables at 6x6 only). E_FN training (16) is C15: tried, failed sampled (issue 3). Linear-zero output + L_fix at 6x6: untested |
| 2 | no walkers at the node | walkers ~ g phi, rare near nodes (AVSL-08: 0/198 profitable orbits visited) | flatter importance g = |psi_T|^beta: F1 parked (ESS 5% at beta 1/2, 6x6); eigen-equation labels from visited neighbours (10): B18 itfit semi-implicit step 87-97%, inconclusive | open |
| 3 | gradient denominator | E_FN force F(x) = sum_viol K_xy a(y)/a(x) [O(y) - O(x)] (16): a(x) small near nodes | C15/E1: exact 4x4 force works (1.74x descent with SR); SAMPLED 8x8 FAILED: walkers for force cosine 0.7 grow 8e4 -> 2.5e8 near the fixed point, physical E_FN worse by +0.046 total (3549023/24). Factorised residual a_new = a_k e^{r} gives finite-variance gradients (Tim, seq72-74); B10 exact 4x4 99.7%, B11 8x8 inconclusive (OOM). Clipped Lanczos factor (B27/D7): works as one-step target, retention 0.84 -> 0.09 | open: the noise problem is the measured blocker of E_FN training |

Reading (lead): issue 3 is not new; the factorised residual is the known handle. Issue 1's fix and issue 3's fix
are compatible: e^{r} J-type correction on a guide whose node (PP) is kept, trained on E_FN with SR.
Next decision for the PI: which of 1-3 to test first, on paper before any run.

## CURRENT 2026-11-17 (theorist mode; the account for the storage question, newest first)
Notation (read from code, not from record names): PP = Gutzwiller-projected singlet pair product, psi_PP = eps(sigma) det f[U, D] (Nomura-Imada
2005.14142; ansatz/code/c2l_rbmpp_lib.py l.3-7); the special case of a Pfaffian with only up-down pairing. 'Pfaffian' in older records means this.
8x8 base (ansatz/code/simple/pp2_lemma.py l.1-12, rbm 'inner', projection 'simple' = 256 images): B(x) = sum_g phi_RBM(z_g) [a_1(g) + lam a_2(g)],
a_j(g) = par(z_g) det f_j[Up(z_g), Dn(z_g)], f_j, b, W complex. B is complex; the loop stores only |B| (sign from S), so arg B plays no role.
Rule: before deriving about an object, copy its definition from the code into this line.
Loop notation (fixed 11-17 with the PI; 'phi' in files = φ in chat): psi_k = S_k A_k (loop state, round k); S_k = its sign (+-1 per configuration,
stored in the sign network); A_k = stored magnitude = |B_0| e^{r_0 + ... + r_{k-1}}; B_0 = base network, frozen; r_k = 'head' of round k (a small
network giving one number per configuration); phi = fixed-node state of the round; R = phi/A_k (ratio); f_J = one-step estimate of ln R;
c = over-relaxation (c ~ 4, clipped |c f_J| <= 4.88); 'dip' = configuration where A_k(x) << phi(x).
STEP BACK 2026-11-17 (PI: 'our amplitude gets its correction from the fixed node under a given sign'):
1. What makes it simple: store the RATIO, not the amplitude. Lattice FN with guide psi_T: (H_xx - E + V_T(x)) |phi(x)| = K_phi(x),
   V_T = Vs_T/|psi_T(x)|. As |psi_T(x)| -> 0, |phi(x)| -> K_phi |psi_T(x)| / Vs_T: phi inherits the guide's wall, and
   ln(phi/psi_T) = ln K_phi - ln(D|psi_T| + Vs_T) stays bounded [exact asymptotic, own]. The round's new information is this bounded ratio.
2. Who solved this form: lattice FN, ten Haaf, van Bemmel, van Leeuwen, van Saarloos, Ceperley, cond-mat/9412037 (abstract read; V ~ 1/psi_T);
   continuum DMC [std]: phi_FN and psi_T share the node, phi/psi_T is smooth there, which is why importance sampling uses f = psi_T phi.
3. Continue, reframed: the frozen-base form is FN's own coordinate, not a shortcut: B_k carries the wall, e^{r} the bounded FN ratio.
   Records already said 'store the smooth part, compute the near-node part' (PLAN_ansatz l.83, 10-24; DISCUSSION_sign 11-08 P6) and the per-row
   relaxation c_eff = c min(1, f0/|f_J|) (PLAN step back 11-03); R1' broke that by refitting the base. Open: can r hold the ratio at 8x8 (B11 OOM).
LOOP FIXED POINTS (11-17, PI: 'with the sign updated, the loop can only go to the ground state?'):
- [std, ten Haaf et al. cond-mat/9412037 construction] H_FN psi_T = H psi_T exactly: removing a violating hop and adding V(x) psi_T(x) =
  sum_viol H_xy psi_T(y) to the diagonal leaves (H psi_T)(x) unchanged.
- [exact, own] Perfect copy, A_{k+1} = |phi[A_k]|: a fixed point has H_FN psi = E psi, hence H psi = E psi. Every fixed point of a perfectly
  copied loop is an exact eigenstate of H (in the sector); there it also satisfies the sign angle's no-flip condition (E < E').
- [own] Not shown: (a) that a round lowers E_FN (lattice E_FN depends on the guide amplitude via V and on the Lanczos sign of an approximate
  state; no monotonicity theorem); (b) that the ground state is the attractor rather than an excited eigenstate in the sector; (c) imperfect
  copy (finite heads, clip, sampling) creates biased fixed points, as Caffarel-Pinar-Scemama arXiv:2609.01301 found for SHDMC with a finite basis.
- [exact, own 11-17, after the PI's push: 'FN energy is better than the guide, ten Haaf'] MONOTONE DESCENT. (1) [std] H_FN - H = sum over violating
  bonds of H_xy [[r, -1], [-1, 1/r]], r = psi_T(y)/psi_T(x) > 0: PSD, and zero on psi_T. Hence E_0 <= E_FN[psi_T] <= E_var[psi_T], and
  E_var[chi] <= <chi|H_FN|chi> for any chi with psi_T's sign; for chi = phi: E_var[phi] <= E_FN[psi_T].
  (2) Flipping one configuration x of S A changes <H> by -4 A(x)(Vs - K) (norm unchanged): it lowers E_var iff Vs > K, the C-free criterion.
  For a flipped set F only bonds leaving F count: Delta = -4 sum_{x in F, y not in F} ... ; per-site C-free labels are safe when no two
  flipped configurations are neighbours.
  (3) Loop guide g_k = S_{k+1} A_k (sign from the step, amplitude stored). With a perfect copy A_{k+1} = |phi_k| and C-free flips:
  E_FN[g_{k+1}] <= E_var[S_{k+2} A_{k+1}] <= E_var[S_{k+1} A_{k+1}] = E_var[phi_k] <= E_FN[g_k]. The FN energy falls every round.
  The full Lanczos rule's extra flips (K > Vs rows) RAISE E_var by 4 A (K - Vs): not covered. Imperfect copy breaks the step E_var[S A_{k+1}] = E_var[phi_k].
  So pushback 1 ('no round is guaranteed downhill') is withdrawn for C-free labels with a perfect copy; pushback 3 (imperfect copy) stands.
- [exact, own 11-17, PI's P.S.: use the Lanczos amplitude too] Design B: FN guide = the full Lanczos state psi'_k = (1 + alpha(H - E)) psi_k
  (sign AND amplitude; the 'p=1 FN' of the headline runs), store psi_{k+1} = phi_k. Chain with a perfect copy and alpha Lanczos-optimal for psi_k:
  E_FN[psi'_{k+1}] <= E_var[psi'_{k+1}] <= E_var[psi_{k+1}] = E_var[phi_k] <= E_FN[psi'_k]. Monotone with the FULL Lanczos sign: the extra flips
  are safe because their amplitude comes with them. (Design A, guide S_{k+1} A_k, needs C-free flips.) Cost of B: guide ratios need neighbours
  of neighbours (PLAN l.512: 5.4x for the PP at 6x6). Note: the f_J storage target so far was built on S_{k+1} A_k (design A), the headline
  FN on psi' (design B): the two halves of the present loop use different guides.
- [exact] Storing |psi'| instead of |phi| (PI's P.S. read literally) is never better: E_var[phi_k] <= E_FN[psi'_k] <= E_var[psi'_k].
- [exact, own] Copy criterion: with copy error raising E_var by eps_k, descent survives iff eps_k <= Delta_k = E_FN[guide_k] - E_var[phi_k] (the
  round's slack). Delta_k -> 0 as the loop converges, so the loop stalls where the copy error equals the per-round gain: the biased fixed point,
  quantified.
- [open] Which eigenstate: resting points are eigenstates; the ground state is guaranteed only if E_FN[start] < E_1 (lowest excited level in
  the sector); otherwise an instability argument for excited eigenstates is missing.
REVIEW 11-17 #2 (Astra cj-76fc73751b2d, /scratch/a/A.Otaifi/astra/ratio_review/ASTRA_ratio.md, check_ratio.py -> checks.json). Adjudicated by the lead:
- accepted: the FN potential is right (cond-mat/9412037 Eqs. 1-4), but 'bounded ratio' is conditional: needs D t_x / Vs_T -> 0 and K_p bounded away
  from 0; approaches to Vs = 0 or K_p = 0 can make ln(p/t) arbitrarily large (triangle t = (eps, 1, sqrt eps): ratio ~ eps^-1/2). Finite on a
  finite lattice, but no uniform bound across rounds or sizes.
- accepted, CORRECTS the lead: f_J is not ln R. Exact: R_x = e^{f_J[A](x)} sum_kept (|H_xy| A_y / K_A(x)) R_y; f_J = ln R only if the neighbours
  are already right (R_y = 1). 'f_J measures the remaining gap at x' was wrong.
- accepted, CORRECTS the lead: the clipped c = 4 iteration is UNSTABLE. Two kept rows: d+ = d - 2 clip(c d), multiplier 1 - 2c = -7, exact period-2
  cycle at the clip; c = 1/4 halves d each round. Balanced row: multiplier 1 - c(1 - Vs/K), stable iff 0 < c(1 - Vs/K) < 2. 'Overshoot self-corrects'
  was wrong; the clip bounds the excursion, it does not stabilise. c ~ 4 was a one-round energy line search, not a stable iteration (cf. 11-03
  'nonlinear SOR instability'). Implementation: clip = c f0 only in mode 'fix' (pp2_lemma.py l.788); nonfinite -> -69.
- consequence, own: the monotone-descent chain (LOOP FIXED POINTS above) assumes the stored amplitude is the exact FN state phi; the c-relaxed
  Jacobi target is not phi, so the chain does not cover the present target.
- accepted: finite heads with |r_j| <= L bound log changes by m L; B_0 not a floor in principle, but finite rounds and capacity limit the reach.
- literature: the frozen-base FN ratio e^{J~} = Psi_FN / Phi_T with fixed antisymmetric Phi_T is published: Reboredo-Hood-Kent 0808.3417 Sec. III.2
  Eqs. (25)-(26). Close: Sorella cond-mat/0009149 (r_x psi^f(x)); Pilati-Inack-Pieri 1907.00907 (A+ = sqrt(A phi_0), log contraction 1/2);
  Kochkov-Clark 1811.12423 (supervised loop); 2303.08902 (self-learning power method). No exact match for Lanczos-sign + stacked heads found.
DESIGN OPEN (i), JACOBI LABELS (11-17, with the PI; on paper):
- [exact, Astra] R_x = e^{f_J[A](x)} sum_kept w_xy R_y, w_xy = |H_xy| A_y / K_A(x) (sums to 1). Write u = ln R and P u (x) = ln sum_kept w_xy e^{u_y}.
- [own, linearised in u - mean] f_J(x) ~ u_x - (P u)_x = ((I - P) u)_x: the one-step label sees the correction only through its deviation
  from the weighted neighbour average. A uniform u gives f_J = 0.
- [own] For an eigenmode of P with eigenvalue mu: u = f_J / (1 - mu), so the right c for that mode is 1/(1 - mu): smooth modes (mu -> 1) need
  c -> inf, the checkerboard mode (mu = -1) needs c = 1/2. A single scalar c fits one band only (c ~ 4 <-> mu ~ 0.75: the 6x6 correction is
  mostly smooth). Iterated, the error per mode is multiplied by 1 - c(1 - mu): stable iff 0 < c(1 - mu) < 2, i.e. c < 1 if rough modes are
  present (Astra's two-row case is mu = -1). This is damped Jacobi on a linear system [std: Jacobi/SOR theory]: a smoother for rough modes,
  too slow for smooth ones.
- [conj, ~55%] Consequence: storing r = c f_J cannot be right for all scales. The repair is to store the solution of (I - P) r = f_J (a linear
  solve on the configuration graph; the pool loss's pair terms (e_x - e_y)^2 are of this form) rather than a rescaled label. Killed if the
  6x6 exact correction ln(phi/A) is matched by a scalar c f_J to within its bulk scatter at every scale.
- Head constraint [exact]: the head must be a function of the configuration x alone; if it read neighbour amplitudes, the guide ratio at
  each neighbour would need two more hops and the one-level cost is lost.
STEP-4 REVIEWS 11-17 (Astra cj-3e0bd21cd52c -> /scratch/a/A.Otaifi/astra/step4_review/ASTRA_step4.md; Opus reviewer -> OPUS_step4.md, opus_checks/;
lead re-ran head_test.py seeds 1-2 and recomputed the 40-seed table from sweep40.txt: matches, a2-unweighted beats no-head 24/40 vs 23 quoted):
- accepted (both, exact): the (a2) loss as written is zero at u = ln(phi/G), G = |psi'|, not at the stored r = ln(phi/A_k); it drops the Lanczos
  factor ell = ln(G/A_k) that PLAN step 4 includes. Corrected residual F_x = ln[a_x A_k(x) e^{r_x}] - ln sum_kept |H_xy| A_k(y) e^{r_y},
  a_x = D_x + Vs_G(x)/G_x - E. Literal loss: next-round E_FN worse than no head in a 6-site toy, 3/3 seeds.
- accepted (both): 'blind directions match' is WRONG. The residual is -ln[1 - (E_L^FN - E)/(D - E + V)], a fixed-sample variance of the
  frozen H_FN: on smooth modes loss ~ nu^2, energy ~ nu (toy slope 0.50-0.52); Astra's 4-state case: loss O(eps^2), energy cost O(eps).
  F1 concerns violating edges and is an upper bound, not the FN energy change.
- accepted: E must equal E_FN for consistency (minor in the toy: dE 0.02 -> <= 3e-4); no spurious nonzero-loss stationary points in
  unrestricted log amplitudes (Collatz-Wielandt), but none of this survives a finite head, sparse rows (toy: 15 near-node rows, three starts,
  equal loss, rms error 10/164/91) or a disconnected kept graph.
- both recommend (d) = TRIED_IDEAS B10/B11: minimise the Rayleigh quotient of the FROZEN H_FN[psi'_k] over the head, chi = S' A_k e^{r}:
  exact minimum phi_k, energy-matched conditioning, no prescribed E, no 1/B. Astra 70%; Opus 55% (with the corrected residual, Huber-clipped,
  on capped near-node rows; forward-walking bulk anchor later). Round acceptance certificate (both): E_var[chi] below the round's E_FN.
- toy table (6-site disordered ring, linear head, 40 seeds; median next-round E_FN deficit vs perfect copy / beats no head): no head 0.014/-;
  direct phi^2 0.011/25; a2 unweighted 0.033/24; a2 phi^2-weighted 0.012/24; a2 pi-weighted 0.014/26; B10 Rayleigh 0.017/31; VMC head 0.010/34.
  Robust only: literal loss worse than no head; unweighted a2 worst median. Rest within seed spread.
- lead's adjudication: withdraw (a2) as main loss; (d) B10 frozen-H_FN Rayleigh becomes the leading candidate for step 4, pending the PI.
  Open for (d) at 8x8: sampling noise of its gradient near nodes (C15 failed with a moving guide; frozen guide gives finite variance per Tim
  seq72-74) and B11's unfinished 8x8 run (OOM).
STEP-4 CONSENSUS 11-17 (PI asked >= 70%; Astra cj-817d8e13fd7d -> step4_review/ASTRA_consensus.md, check_consensus.py; Opus -> OPUS_consensus.md,
opus_checks/w_checks.py; lead recomputed the 31/40 sign test, p = 6.8e-4): try (d) = frozen-H_FN[psi'_k] Rayleigh over the head as the next 6x6
check: Astra 75%, Opus 70%, lead 70%. Success at scale: Astra ~45%, Opus ~50%.
- W1: paired toy evidence favours (d) over no head (31/40; median paired difference -0.0068); medians of separate deficits mislead.
- W2 [Astra, derived + exact toys]: covariance-gradient estimator (Umrigar-Filippi cond-mat/0412634 Eqs. 5-6), single-draw covariance
  Sigma_ij = (4/Z) sum_x b_x^2 Obar_i Obar_j - g_i g_j, b = (H_F - E_F) t. Zero at t = phi (toy: 0 vs 0.107 for C15's estimator). Not C15.
  Caveat: design B freezes H_FN[S'G] but multiplies A_k: where A_k lacks a new guide wall (t_x = O(1), G_x = eps) the row's second moment is
  O(eps^-2) (toy 2.07e7 at eps 1e-4, 2.07e11 at 1e-6; matched wall 3.5). The Lanczos factor again. Corrections ~ eps at p_x ~ eps still
  need eps^-2 samples.
- PROVENANCE CORRECTION (Astra): the 8e4 -> 2.5e8 walker counts in the issue-3 row are Tim's 4x4 study (STATUS.md:785), not 8x8.
- W3: Rayleigh error (3) = sum_kept h p_x p_y (e^{e_x} - e^{e_y})^2 / sum p^2 e^{2e}: filling a frozen wall costs O(1/eps), so (d) protects
  walls; it cannot make the Lanczos factor representable. Opus: clipping the guide's Lanczos factor to the head's range helped the toy
  (34-36/40), descent for the clipped guide unproven.
- W4: B11's OOM was JIT constants (STATUS.md:943-953), chunked retry passed a CG solve; design B differs from B10 (wall of G, not of A_k),
  so B10's 4x4 99.7% does not certify it.
- Required control (Astra): an equal-cost head trained on the physical H (VMC). If it matches or beats (d), prefer it.
- Agreed kill line (merged; no run before the PI says 'now test it'): 6x6 exact tables used for grading only, realistic sampled rows,
  x-only head, early and late design-B states, three rounds. Stop if: retained fraction of the perfect copy's next-round FN gain < 0.5
  (round 1 or 3) or falls > 0.3 over the rounds; or the certificate E_H(chi) < E_FN[g] fails; or the sampled arm is > 0.15 below the
  exact-gradient arm, or M_req = 16 d^T Sigma d/(d^T g)^2 exceeds the declared budget. Void unless the literal STEP4 loss fails (rho_1 <= 0.2).
RANGE-LIMIT BOUND FOR (d) (rung 1, own 11-17, to be reviewed). Frozen guide g = S'G, FN ground state p > 0 (sum p^2 = 1), a_x = D_x - E + V_G(x)
so that K_p(x) = a_x p_x. Ideal head r* = ln(p/A_k). Any head with |r| <= L can reach r = clip(r*, -L, L); the best range-limited head is no
worse [exact: min <= any feasible value]. Then t = p e^{e}, e = 0 on good rows (|r*| <= L), e < 0 on under-filled rows (r* > L: A_k too small),
e > 0 on over-filled rows (r* < -L: A_k lacks a wall of G). From (3), using (u - v)^2 <= max(u, v)^2 for u, v >= 0 and summing each bad row
over all its kept edges (bad-bad edges counted twice):
   E_F[t] - E_FN  <=  sum_{x bad} a_x max(p_x^2, t_x^2)  /  sum_{x good} p_x^2,      t_x = A_k(x) e^{-L} on over-filled rows.  [exact inequality]
Reading: under-filled rows cost at most a_x times their own p^2 weight; over-filled rows cost a_x A_k(x)^2 e^{-2L}, with a_x ~ Vs/G_x large at a new
guide wall: the dominant term. Covers only the range limit, not other capacity limits of the head. Rung 2 quantity: this bound per round on the
6x6 exact data, to be compared with half the round's gain B/2 (kill line); if bound << B/2 the range limit cannot fail the test.
RUNG 2 PLAN for the range-limit question (lead 11-17, written before computing; PI: 'judge whether paper + existing data settle it'):
Data on disk, CPU only, no training, no sampling: 6x6 copy table A_0 (pfaffian_c/runs/c1_cplx_s1/round_1/la_table.npy), oracle sign s2
(amplitude-only check, exact signs as Astra advised), exact FN solver (delta0.efn_vec, LOBPCG, ~100 s per solve), Lanczos guide
G = A_0 |1 + alpha*(E_L - E)| with alpha* = -0.10 (P8; cstar_lanczos.py). Compute p = FN state of guide s2 G; r* = ln(p/A_0); the bound above
for L in {1, 2, 3, 4.88}; and the RETENTION CEILING of a range-limited head, rho_L = (N(chi_0) - N(chi_L)) / (N(chi_0) - N(phi)),
chi_L = s2 A_0 e^{clip(r* - m, -L, L)} (m = p^2-weighted mean of r*), N(chi) = E_FN of chi's own optimal Lanczos guide.
Decision rule (written first): rho_4.88 < 0.5 -> B10 with a head of that range cannot pass its kill line: stop before any training run.
rho_4.88 >= 0.7 and bound << B/2 -> the range limit is not the risk; what remains (learnability of r* by an x-only head) needs a run.
Lead's prediction: rho_4.88 >= 0.7 at ~55% (161,028 orbits have |1 + alpha(E_L - E)| < 0.05, i.e. new guide walls A_0 lacks).
Does not settle: whether a finite x-only network can learn the clipped r* (gap 1), nor 8x8.
RUNG 2b (PI yes 11-17; written before computing): TANGENT SHARE of the clipped target for a candidate head, no training.
Head (lead's choice for this check): symmetric RBM factor r_theta(x) = mean over the 144 projection images g of sum_k ln cosh(b_k + W_k . g x),
M in {8, 16, 32}, 3 random inits (scale 0.01). J = d r_theta / d theta on a p^2-weighted sample of 1e5 orbits (stratified by decade).
Target u = clip(r* - m, -L, L) at L = 4.88 from rung 2. Share in two norms: (i) p^2-weighted L2; (ii) the energy norm of (3) linearised,
<u, v>_Q = sum_kept h_xy p_x p_y (u_x - u_y)(v_x - v_y). Share = |proj_J u|^2 / |u|^2 (ridge-scanned).
Decision rule: energy-norm share >= 0.5 at M = 16 -> a first-order head can hold most of the correction, the run is justified;
< 0.1 -> this head cannot store it, stop and rethink the head before any run; between -> report, the PI decides (PI 11-17 18:4x: overruled, the lead decides 2b).
Lead's prediction: energy-norm share 0.1-0.5 (40% that it is >= 0.5); Test Theta found 0-4% for the PP itself, but an RBM factor is
built for smooth positive corrections. Not covered: nonlinear training beyond the tangent space; other head classes.
8x8 RUNG-2 QUANTITY (goal item 3; written before computing, lead 11-17). Exact p is not available at 8x8, so use the rung-1 bound's
dominant (over-filled) term with p replaced by its wall asymptotic (p ~ G at a wall up to a bounded factor) and G_y ~ B_0(y) at the big
neighbours: with f(x) = |1 + alpha(E_L(x) - E)| (Lanczos factor of psi_0 = S|B_0|), over-filled rows = {f < e^{-L}}, and
   Q(L) = E_{|B_0|^2}[ e^{-2L} (sum_viol |H_xy| B_0(y)/B_0(x)) / f(x) * 1{f(x) < e^{-L}} ] / E_{|B_0|^2}[ f(x)^2 ] / N_sites.
One-hop only (E_L(x) and violating-neighbour ratios: pp2_lemma one_swap_ratios on CPU), on the beta = 1 rows of the s306 pool
(pool_comb/pool/pool.npz, sampled ~ |B_0|^2), alpha = the moment-optimal alpha of B_0 (fall back to the declared -0.170 if unavailable),
E = E_var[B_0] from the same rows. Calibration first: the same proxy on 6x6 (A_0, s2, alpha -0.10; rung-2 files) must reproduce the exact
6x6 over-filled bound term within a factor 3, else the proxy is invalid and the 8x8 round needs the PI.
Threshold (written first): Q(4.88) <= 2.5e-7 per site (10x below 6x6's B/2 = 2.5e-6; the 8x8 gain B_8 is unknown) AND the weight
E_{|B0|^2}[f^2 1{f < e^{-4.88}}]/E[f^2] <= 10x its 6x6 value. Pass both -> 8x8 rung 2 passes. Lead's prediction: pass (60%).
RUNG 2 RESULT (11-17, worker; lead re-derived from rangelimit_B.json / rangelimit_A.json): PASS. B = N(chi_0) - N(phi) = 8.6826e-6 - 3.6670e-6
= 5.016e-6 /site. rho_L = 1.0051, 1.0014, 1.0002, 1.000005 at L = 1, 2, 3, 4.88; break test L = 0 gives rho 0 (bit-identical N). Bound at 4.88 =
6.5e-8 << B/2 = 2.5e-6. Certificate E_var[chi_L] ~ 7.2e-6 < E_FN 8.68e-6 at every L. Reading: at 6x6 the ideal correction ln(p/A_0) lies almost
entirely within +-1 (weight beyond 1: 1e-6), so the head's RANGE is not a risk; slightly clipped heads even beat the perfect copy (rho > 1).
FN solves reproduce cstar_lanczos.json to 2e-11; K_p = a p to 2e-8 (weighted). Line search on a 0.1 grid (alpha* = -0.10 for all).
RUNG 2b RESULT (first attempt): INVALID. The random-vector break test failed: energy-norm share of a random vector (0.022-0.048) equals the
target's (0.021-0.049). Diagnosis: (i) init scale 0.01 makes the RBM tangent space nearly degenerate (rank 11-13 / 48-69 of 296-1184);
(ii) the in-sample share has a chance level ~ rank/ESS (ESS 4,920), so in-sample shares measure the sample, not the target.
RUNG 2b RETRY (rule written before computing): init scales 0.3 and 1.0, M in {16, 32}, 3 seeds; larger sample (target Kish ESS >= 20 P);
PRIMARY metric = HELD-OUT share: projection fitted on one half of the sample, share evaluated on the other half (chance ~ 0), both norms,
ridge chosen on the fit half. Break tests: held-out share of a random vector ~ 0; of r_theta at another init ~ 1.
Decision (lead decides, PI 11-17): held-out energy-norm share of u >= 0.1 at M = 16 (median of seeds, either scale) -> pass;
< 0.05 at both M -> fail (rethink the head class on paper; next candidate a small translation-invariant CNN, rung 2 again);
between -> pass only if M = 32 reaches >= 0.1. Lead's prediction: pass at 50%.
8x8 RUNG-2 RESULT (11-17, worker; lead re-read q8_100000_s1.json and q8_rows_100000_s1.npz, calib6.json):
- Calibration PASS: proxy / exact 6x6 over-filled term = 1.010 at L = 3 and 4.88 (gate: within 3x); exact term reproduces rangelimit_A.json.
- 8x8 (s306 B_0, simple projection, sign sgn Re[w B_0] w = -1, 1e5 unfiltered beta = 1 draws ~ |B_0|^2, lnB reproduced exactly;
  alpha = -0.170 declared, moment alpha needs two hops; E = -0.4988301/site): f_min = 0.0337; 1 row with f < e^-3, 0 rows with f < e^-4.88.
  Q(3) = 3.6e-7 +- 3.5e-7 (one row); Q(4.88) = 0 (sampled).
- Criterion 1 (Q(4.88) <= 2.5e-7/site): PASS (point value 0).
- Criterion 2 (f^2-weighted wall weight <= 10x 6x6's 1.85e-12 -> 1.85e-11): UNRESOLVED. Sampling bounds P(f < e^-4.88) < 3e-5 (95%), six
  orders above the bar. Lead's reading: the criterion was ill-posed (the f^2 weight suppresses exactly the rows that cost; Q is the right
  quantity); not redefined after seeing the data. Gate for the 8x8 round therefore stays CLOSED on criterion 2 unless the PI accepts
  criterion 1 alone; raised only if the 6x6 check passes.
- Observation: the 8x8 Lanczos factor is tame under |B_0|^2 (30 of 1e5 rows with f < 0.5) vs 6x6's 161,028 orbits with f < 0.05 (unweighted).
RUNG 2b RETRY RESULT (11-17; lead read the logs in rangelimit_6x6/tangent_retry/, 10 of 12 configurations; M32 s1.0 seeds 1-2 still running):
| M | scale | held-out L2 share u / random xi | held-out energy-norm (Q) share u / random xi |
| 16 | 0.3 | 0.119-0.121 / 0.0005-0.0010 | 0.019-0.020 / 0.014-0.018 |
| 16 | 1.0 | 0.101-0.105 / 0.0016-0.0032 | 0.016 / 0.021-0.022 |
| 32 | 0.3 | 0.132-0.143 / 0.0010-0.0018 | 0.023-0.026 / 0.026-0.028 |
| 32 | 1.0 | 0.121 (seed 0) / 0.0031 | 0.019 / 0.035 |
(r_theta break test ~1 in both norms; tangent rank now full, 592/1184.)
Verdict: L2 metric VALID (random ~0.001-0.003): the RBM head's first-order span holds 10-14% of the target, ~100x a random vector.
Energy-norm metric INVALID again: random xi's held-out Q share (0.014-0.035) equals u's. Diagnosis: the Q norm couples each sampled row to its
neighbours, and fit and held-out halves share neighbours, so the held-out half is not independent (leakage); the chance floor ~0.02-0.03
equals everything measured. So the rule's primary metric cannot decide; NOT a pass. Repair (same rung, existing data, no sampling):
EXACT full-table shares, J^T W J and J^T W u accumulated over all 15.8M orbits and their kept edges (both norms), no split; chance level for a
random vector ~ P / N_eff ~ 0, checked by the xi break test. Decision rule unchanged: Q share >= 0.1 at M = 16 -> pass; < 0.05 at M 16 and 32
-> fail (rethink the head class on paper); between -> pass only if M = 32 >= 0.1.
RUNG 2b FINAL (11-17, exact full table, 15.8M orbits, no sampling; worker; lead read tangent_full/combine.txt): FAIL by the written rule.
Energy-norm share of u: 0.021 (M 16), 0.028 (M 32) < 0.05 at both, and BELOW a random vector's share (0.034, 0.063). L2: 0.132 / 0.163 vs a
random vector's 0.056 / 0.098 (expected 0.068 / 0.110). Break test r_theta(other init) 0.98-1.00. Ridge scan flat.
DIAGNOSIS ON PAPER (lead): (1) the records already said so and I did not check them before choosing the head: M-R KILL 10-28 (PLAN l.289):
an RBM residual stored ~6% of R* = ln(phi/A); Test L: 2-body features leave eps^2 = 0.973 of R*, while the copy's own Jacobi factor f_J alone
leaves 0.079. (2) The law tax/gain = eps^2 (M-R, verified to 1%) turns a share into a retention: rho ~ 1 - eps^2 ~ energy-norm share, so the
RBM head predicts rho ~ 0.02-0.03 at round 1, far below the kill line 0.5. The 6x6 B10 check with this head is not recommended by the data;
not submitted. (3) Mechanism: near init an RBM factor spans symmetrised one- and two-spin functions (Jastrow class); the FN correction is a
functional of the base's LOCAL AMPLITUDE RATIOS (f_J is built from them), not a low-order spin correlation, so a head that sees only spins
must rebuild the base's ratios to express it. B10's 4x4 success used a 155k-parameter ViT (results/frozen_hfn_..._2026-10-03.md).
NEXT-METHOD RUNG 2 (goal item 5; written before computing, lead 11-17). Norm fixed by the records: the M-R law tax/gain = eps^2 holds
(to 1%) with eps^2 = D(R - R*, R - R*)/D(R*, R*), D(a,b) = sum over BLOCKED pairs w (a_x - a_y)(b_x - b_y), w = n_r 0.5 HW_k phi_r phi_y
(mr_residual.py). (My kept-edge Q norm in rung 2b was the wrong norm for retention; the RBM verdict stands on M-R's own 6%.)
Candidate head class: a FEATURE head, r(x) = g(phi(x)) with local features of the stored state at x:
  l(x) = ln|1 + alpha(E_L(x) - E)| (Lanczos part, one hop), f_J[A](x) (one hop), f_J[G](x) (two hops), ln A(x), K/Vs at x.
Question for existing data (6x6, design B round 1: u = ln(p/A_0) from rangelimit_6x6): eps^2 of the best least-squares fit of u in the
D-norm by (i) c * f_J[G] alone, (ii) l + c f_J[G], (iii) a cubic polynomial in all features (in-sample, P << N, and 5-fold by orbit blocks).
Predicted retention rho = 1 - eps^2. Decision: rho_pred >= 0.7 for (ii) or (iii) -> the feature head is the next method (paper proposal +
Astra); < 0.5 for all -> features of the stored state cannot carry the round's correction either; the remaining route is consolidation.
Lead's prediction: (ii) reaches eps^2 <= 0.3 at 65% (Test L: f_J alone left 0.079 in design A).
Cost note for the proposal (paper): features of A_k at x need A_k at one or two hops, and A_k contains earlier feature heads, so features
of the current state nest one level per round; bounded only with periodic consolidation (merge into a refitted base).
NEXT-METHOD RUNG 2 RESULT (11-17, worker; lead re-read features/features.json): PASS, strongly. In the M-R D-norm, design B round 1 at 6x6:
  (i) c f_J[G] alone: eps^2 0.852 (CV 0.861), c = 3.22;  (ii) a l + c f_J[G]: eps^2 0.0033 (CV 0.0034), a = 1.015, c = 1.307;
  (iii) cubic in five features: 0.0019 (CV 0.0020). Break tests: R = u -> 0, R = 0 -> 1, random column -> 0.99999.
  Direct FN check of (ii): E_FN[s2 A_0 e^R] = 6.511e-6/site vs 6.285e-6 for s2 p, gain 1.464e-5 -> retention 0.985 (tax/gain 0.0155;
  the M-R law eps^2 = tax/gain is off by 4.65x here, so use direct checks). Feature inputs reproduce pfaffian_l f_J exactly.
READING [own]: the round's FN correction relative to the stored state is, to 99.7% (D-norm) and 98.5% of the energy gain, an EXPLICIT local
formula: u ~ l + 1.31 f_J[G], i.e. keep the Lanczos amplitude factor (coefficient 1.01) and add 1.31 Jacobi steps of the FN equation on the
Lanczos guide. Nothing needs to be LEARNED inside a round; the storage problem is a COST problem (the formula needs the stored state at two
hops, and rounds nest), to be solved by periodic consolidation into a refitted base with a loss whose 1/B is a fixed weight (relative square
or frozen-H_FN Rayleigh with a full network). Proposal: docs/theory/2026-11-17_next_method_proposal.md.
ROUNDS 2-3 RUNG 2 (written before computing): apply the formula exactly on the table (A_{k+1} = A_k |1 + alpha_k(E_L - E)| e^{1.307 f_J[G_k]},
sign s2), recompute G_k, p_k, u_k and refit (ii). Rule: eps^2(ii) <= 0.05 and refitted c within +-30% of 1.307 in rounds 2 and 3 -> the
explicit formula is a stable round map; else not (then the formula is round-1 specific). Also report each round's FN energy of the
explicit state (exact, no storage error) as the ceiling the consolidated loop can reach. Prediction: pass at 60%.
ROUNDS 2-3 RESULT (11-17, worker; re-read by lead from rangelimit_6x6/rounds23/rounds.json; script ansatz/code/rangelimit_rounds.py sha 146b8c3d):
PASS. Fit (ii) eps^2 in/CV: round 2 0.00145/0.00152 (a 1.007, c 1.381, +5.7%); round 3 0.00389/0.00403 (a 0.998, c 1.575, +20.5%).
FN check of (ii): tax/gain 0.026 and 0.029. Explicit states (dE/site): E_FN[s2 A_1] 6.568e-6 (ceiling p_0 6.285e-6, from A_0 2.092e-5:
98.1% of the round's gain kept), E_FN[s2 A_2] 3.320e-6 (ceiling p_1 3.194e-6: 96.3%), ceiling p_2 2.074e-6. Optimal alpha -0.10, -0.10, -0.075.
Breaks: R=u 0, R=0 1, random column 1.0000. Caveat [own]: c rises (1.31, 1.38, 1.58); at that pace round 4 leaves the +-30% band, and f_J alone
needs c 2.0 then 4.4: the Jacobi term's optimal weight grows as the state nears the FN fixed point. Not a kill under the written rule.
B10 6x6 HARNESS (11-17, staging worker; built and CPU-tested, NOT submitted; runs/j1j2/b10_6x6/, code ansatz/code/b10_6x6.py): tests (a)-(d) pass
(clipped-r* head reproduces rung 2's rho = 1.000005 exactly; zero head bit-identical; exact frozen-H_FN gradient vs finite differences 1.2e-8,
broken gradient 237; arm X loss 8.6e-16 at ln(p/G) vs 3.1e-5 at ln(p/A_0)). LATE state proposal: pfaffian_r6k3/k3/tab_th/tab_t6/round_1
(E_var 8.927e-6/site). One CPU pilot step (EARLY, round 1), lead re-read cpu_pilot/log.txt:
  arm S (sampled frozen-H_FN Rayleigh): population M_req = 5.24e9 vs the kill bar 2e5; the 4096-draw estimate of M_req was 1.45e4 (the heavy
  tail is invisible in the sample); exact directional derivative +0.01 (wrong sign). arm C (physical-H, same head and draws): M_req 8.8e3.
  Reading: Astra's unmatched-wall term (W2) is real at 6x6 design B: B10 sampled on the Lanczos guide is not trainable at an affordable
  sample size, and the sample cannot tell. The RBM head's failure (rung 2b) and this sampling failure are independent; either blocks B10.
Organising question: which storage loss moves B_k toward the Jacobi-corrected target without a 1/B term at nodes the network itself moves?
- [exact] Loop: psi_k = S_k|B_k| -> psi' = (1 + alpha(H - E)) psi_k -> FN phi with sgn phi = sgn psi' -> S_{k+1} <- sgn psi', |B_{k+1}| <- |phi|.
- [exact] Target T(x) = B_k(x) e^{c f_J(x)}, f_J(x) = ln[ sum_z |H_xz| |B_k(z)| / ((H_xx + V - E') |B_k(x)|) ]; |f_J| <= 0.14 at q.9999 (8x8).
- [exact] Why R1' failed: a loss on ln|B_theta| has d ln|B| = dB/B, so the rows near nodes of B_theta own the gradient (tangent norms 26-73x typical,
  PLAN_pfaffian 2577); the moving base B_0' + lam PP moved its nodes. Stored fraction f 0.145 (6x6: 0.49). Five step-size patches did not change this term.
- [exact, PI 11-17] Reformulation: with B_k frozen, B_{k+1} = B_k e^{r_theta} and storing is plain regression r_theta(x) ~ c f_J(x): no 1/B in the
  gradient, target bounded on sampled rows (not near spurious zeros, below). Kept only for cost (recomputing B_k e^{r} nested k times costs 256^k neighbour sums otherwise). Price: zeros of B_k persist.
- [exact, Astra cj-5ca32ee74fb4, /scratch/a/A.Otaifi/astra/lfix_review/ASTRA_lfix.md] L_fix = sum 2h t_x t_y (cosh delta - 1) is NOT log-free: its gradient
  ~ sinh(delta)(O_x - O_y) scales 1/eps^2 at a spurious node. Astra's ranking: frozen base x bounded head (~70%) > amplitude L2/Hellinger > L_fix.
  Precedents: TRIED_IDEAS B10 (exact 4x4, 99.7%), B11 (8x8, OOM, outcome never recorded).
- [cond] Auditor (docs/reviews/2026-11-17_drift_audit.md row 15): asinh(B/s) has a bounded gradient at the node, dB/sqrt(s^2 + B^2); it was closed on a
  freezing counterfactual that cannot create sign-crossing moves (biased). Not re-opened yet: to be weighed on paper.
- [exact, own 11-17] No node moves: B_{k+1} = B_k e^{r}, e^{r} in (0, inf) for finite r, so {B_{k+1} = 0} = {B_k = 0} and arg B_{k+1} = arg B_k.
- [exact, own 11-17] Gradient of L = sum_x w(x) (r_theta(x) - c f_J(x))^2: dL/dtheta = 2 sum_x w (r_theta - c f_J) dr_theta/dtheta. Near a zero of B_k,
  |B_k(x)| = eps with kept neighbours O(1): f_J = ln(1/eps) + O(1), so one row contributes ~ w c ln(1/eps). R1' log loss: ~ w / eps. The pole becomes a log.
- [cond: zero of B_k is spurious, i.e. A'(x) = sum_kept |H_xz||B_k(z)| / (H_xx + V - E') stays O(1); own 11-17] The target is unbounded there:
  c f_J ~ c ln(1/eps), and T = B_k e^{c f_J} = B_k^{1-c} A'^c ~ eps^{1-c}, which DIVERGES for the measured bulk c = 2.4-3.9 (PLAN 282-317). The true
  correction ln(phi/B_k) ~ 1 x ln(1/eps) (phi finite where B_k has a spurious zero). So near spurious zeros c f_J overshoots by a factor c, and a
  bounded r cannot carry it at all. At a true node (V large, A'(x) ~ eps too) f_J stays O(1): no problem.
- [conj, ~50%; KILLED 11-17, see 'where B is small' below: raw f_J in [-0.78, 0.55] on the 2.1M SAMPLED pool rows (decades -9..-2), 0 rows > 3 (Astra; the [-3.13, 2.2] quoted first was c f_J, c = 4); deeper or unsampled dips not excluded] Spurious near-zeros of B_k exist at 8x8 and matter: the 6x6 bond split put 75-84% of the copy's tax on "small side far too small"
  (e_x ~ -6 to -9 log units), i.e. exactly such dips. Killed if B_k at 8x8 has no rows with A'(x)/|B_k(x)| > e^{3}.
- [exact, own 11-17; corrected the same day after an outsider's remark and reading the code] Per image term: an exchange hop changes one row AND
  one column of F_g = f[Up(z_g), Dn(z_g)], so a_j(g; y) / a_j(g; x) = +- det(I_2 + K_g), K_g built from F_g^{-1} (rank-2 lemma, pp2_lemma.py l.13-15).
  For ||K|| < 1, |det(I_2 + K)| >= (1 - ||K||)^2 > 0, so a drop of one image term to near zero across a hop needs ||K_g|| >~ 1, i.e. F_g^{-1} large
  along the hop (sigma_min(F_g) <~ ||v||). For real F this is the statement 'a sign flip needs near-singular F'. (Superseded first version: a
  one-particle row update of a general Pfaffian, ratio 1 - e_a^T A^{-1} b, sign fixed by the PI; wrong move for a spin exchange.)
- [cond, not covered] |B| can also be small by cancellation among the 2 x 256 complex terms with none of them small; the per-term argument does
  not cover that. Exact zeros of complex B need Re = Im = 0 (codimension 2 in theta at fixed x): a training path passes near them, not through;
  d ln|B| = Re(dB/B) is still ~ 1/|B| there. Which mechanism makes the 8x8 dips (near-singular F_g or image cancellation): open, on paper.
- [measured, existing data; runs/j1j2/r1prime_fit/diag_ho/jac_rows.npz, step-0 theta of R1'] At the 5 worst rows (tangent 26-73x the decade -9
  median): (i) the copy did NOT move toward 0 in the fit; ln|P_1| ROSE by 1.5-3.4 (step 0 -> 150; already in PLAN_pfaffian 2572). (ii) No P_1 vs lam P_2
  cancellation: ln|B| - max(ln|P_1|, ln|lam P_2|) = -0.45..+0.30 (typical rows +0.05, lam P_2 ~ e^-3 P_1). (iii) The stiffness is in P_1's parameters
  (nP1 ~ norm). (iv) Near-zero signature: |B(x)| is 3.2-7.3 e-folds below its bond partners while |grad B(x)| is within -2.3..+0.6 e-folds of theirs,
  i.e. B small but its slope not small: B ~ g (theta - theta_0), not smooth suppression. Reading: the rows sit close to a zero of B in parameter space.
- [conj, 55%, stated 11-17 before any computation] Mechanism (b), cancellation among the 256 images of P_1, not (a) one near-singular F_g. Prior: 6x6 N0
  (delta 10-23) cancellation 0.056 vs 0.583 on the top-0.1% bonds. Killed if kappa = |sum_g t_g| / sum_g |t_g| at the 5 rows is >= 0.5 x the r9 median.
- [measured 11-17, worker CPU, verified by the lead; runs/j1j2/r1prime_fit/diag_ho/image_cancel.json, .py; reproduces lnB to 1.4e-14, lnP1 to 7e-14;
  break test sigma_min/med 0.0082 -> 1e-16] (b) CONFIRMED, (a) not supported. kappa = |sum_g t_g| / sum_g |t_g| over the 256 images of P_1:
  worst 5 rows 6.1e-4, 7.9e-3, 4.9e-3, 8.9e-3, 3.5e-3; r9 median 0.27 (q05 0.082, min of 200 rows 0.018), r5 0.63, r2 0.83, partners ~0.46.
  Kill bar (>= 0.5 x 0.27) not reached: the worst rows are below every one of 200 random decade -9 rows. Top-image sigma_min/median 0.008-0.014
  vs r9 median 0.0063: no special near-singular matrix. n_eff ~ 30-60 images contribute; none dominates (max share 0.07-0.14).
- [own, cond: image gradients do not cancel like the images] B = sum_g t_g, |B| = kappa sum|t_g|, grad B = sum_g grad t_g, so
  |grad ln B| ~ (1/kappa) x typical. Check: kappa_r9 / kappa_x = 443, 34, 56, 30, 76 vs measured tangent ratio 46, 30, 26, 34, 73: same order in 4 of 5,
  the first off by ~10 (its image gradients partly cancel too). Also: kappa falls with depth (0.83, 0.63, 0.27 at decades -2, -5, -9): deep amplitudes
  are made by interference of the symmetry images, and the log loss pays 1/kappa there.
- [own + existing data, jac_rows.npz dlnB_actual/linear, steps 0 -> 150] Linear region: ln B(theta + delta) is linear while |grad B . delta| << |B| =
  kappa sum|t_g|; with sum|grad t_g| not cancelling, |delta| <~ kappa x (bulk scale). Worst rows: linear prediction -8.7..+7.7, actual +1.6..+2.7
  (wrong sign in 2 of 5; corrected by Astra): the fit left their linear region. Typical r9 rows: 0.15 vs 0.10. So the steps are too large FOR these rows (PI 11-17), but a
  step inside their region is kappa_x / kappa_bulk ~ 1/30 - 1/450 of what the bulk needs: one step size cannot serve both. The five step-size patches
  (clip, backtracking, hinge) met exactly this. The term to change is the coordinate: ln of a sum that cancels. The frozen-base form keeps B_k's
  cancellations fixed and fits only e^r, which has none.
- [exact, own 11-17; PI's split: log loss for e^r, square loss for B] Model Psi = B_theta_B e^{r_theta_r}, target T = B_k e^{c f_J}.
  Log loss l = (ln|B| + r - ln|B_k| - c f_J)^2: dl/dtheta_r = 2 res dr/dtheta_r (no 1/B for any B); dl/dtheta_B = 2 res Re(dB/dtheta_B / B) (1/B returns).
  So B must not see the log loss. B frozen (= B_k): res = r - c f_J, plain regression. B trainable under L_B = sum w |B e^r - T|^2:
  dL_B/dtheta_B = 2 w Re[(Psi - T)* e^r dB/dtheta_B], no 1/B; weight ~ T^2, so B learns the bulk and barely sees deep rows. Price: nothing holds
  B's cancellations in place; a cancellation moved onto a row with finite T leaves the log loss on r a residual ln(1/eps), and r chases it.
  Open: the two losses share a minimum only where Psi = T is reachable; otherwise alternating updates may not have a joint fixed point.
- [measured 11-17, CPU on existing files: pool_comb/pool/pool.npz (s306 base, 2.1M rows, betas 1, 0.5, 0.25), image_cancel.json, jac_rows.npz]
  WHERE B IS SMALL. The worst rows sit on the small side of a sign-violating (blocked) bond: their blocked partners are 3.1-7.5 e-folds larger.
  The Jacobi target agrees they are small: ln T - ln B = -0.18, -1.5, -0.12, -1.4, -0.14 (target even smaller). [corrected by Astra: this does not certify them as true; the target is built from the dip itself.] Over 200 random decade -9 rows: corr(ln kappa, largest blocked-partner excess) = -0.53 (decade -5: -0.43): the network makes the
  small side of the FN wall by cancelling its symmetry images.
- [exact, own 11-17] Why 'the target agrees' is not circular although T is built from B_k: with K = sum_kept |H_xz| |B(z)|, Vs = sum_viol |H_xy| |B(y)|,
  f_J = ln[K / ((H_xx - E')|B(x)| + Vs)]. At a wall row Vs >> (H_xx - E')|B(x)|, so f_J ~ ln(K / Vs), independent of |B(x)|. The local FN fixed point
  needs |B(x)| = (K - Vs)/(H_xx - E'), which exists only if K > Vs. f_J < 0 there means K < Vs: the FN equation pushes x smaller whatever its present
  size. So x small is demanded by its neighbours. (V-dominance at the 5 rows not yet checked: pool stores WB = Vs/|B(x)| but not K.)
- [measured] 1/kappa law: corr(ln kappa, ln |grad ln B|) = -0.93 (r9), -0.88 (r5) over 200 random rows each. The stiffness IS the cancellation.
- [measured] R1' trigger: the warm start added lam P_2 from another optimum (s307), whose images do not cancel where P_1's do: ln|lam P_2| - ln|P_1|
  = +1.55, -4.66, -1.04, -1.01, -2.02 at the worst rows vs -3.0 typical; at row 1481899 the warm B was 1.75 above the base, 1.93 above the target,
  at the stiffest row (1/kappa = 1600). Large residual x 1/kappa.
- [WITHDRAWN 11-17 by Astra's review: an update moves ln B by O_x . delta, not |O_x||delta|; the norm is an upper bound only, so freezing is sufficient, not shown necessary] Cancellation is fragile under ANY loss: an update delta chosen for the bulk moves
  ln B at a cancelling row by ~ (1/kappa) |delta| x bulk scale, whether or not the loss looks at that row. So a trainable B (option 2, square loss)
  scrambles the wall rows while fitting the bulk, and r is then left a target ln T - ln B_theta with jumps of order ln(1/kappa'). Freezing B_k is
  therefore needed, not just convenient. Option 1's r-target: the implemented target is clipped, |c f| <= 4.88 (pp2_lemma._apply_mode); raw c f_J is unbounded as K -> 0 or D b + Vs -> 0.
- Not checked: rows the pool under-samples (drawn ~ |B_k|^(2 beta), beta >= 0.25; decades -2..-9 only); the pair weights rho at the wall rows in
  option 1's loss (bounded regression, but possibly dominant weights); whether a smooth r can follow c f_J across the wall.
- [measured 11-17, pool.npz, CPU] Option 1 at the wall rows. With Psi = B_k e^r the pool_fit loss becomes e = r - f - c, pair term (r_x - r_y - (f_x - f_y))^2:
  no 1/B. Target jump across a blocked bond f_y - f_x: q01 -0.73, q99 +0.85, |.| > 1 on 0.97% of bonds (wall rows, partner > e^3: q99 +1.16).
  Pair-weight concentration (w_in x sum rho): decade -9 ESS 74k of 349k rows, top 1% of rows 14% of the weight, wall rows (10%) 31%; decade -2: ESS
  59k of 87k. So the regression is well spread; the wall rows weigh more but do not take it over. Smoothness of r across a bond: needs jumps of
  c x (<= 1.2) at the 1% tail; not shown that a network r can follow them (Test Theta-type tangent share would decide).
- [exact, own 11-17] Nesting cost of option 1 with FITTED heads: B_k = B_0 e^{r_0 + ... + r_{k-1}}; evaluation = one B_0 (256 images) + k small heads,
  linear in k; round k's target c f_J[B_k] needs one level of neighbour sums of B_k. The 256^k cost holds only if r_i := c f_J[B_i] is computed,
  not fitted. Heads can be merged by a bounded regression (r_sum ~ sum r_i) when k grows.
- [conj, was ~60%; LARGELY REFUTED at the 5 rows by the K, Vs values below] The real limit of option 1 is the moving sign. B_0's cancellations sit at the walls of S_0 (corr -0.53 above). A sign round
  flips 0.8-2.7% of rows (PLAN_sign l.806-810, other pool), i.e. moves walls. At an old wall row the new state need not be small: r must reach
  + ln(1/kappa) (up to ln 1600 ~ 7.4) in one swap; at a new wall row B_0 is not small and r must dip by several units in one swap (board 7: a smooth
  log net cannot). Killed if f_J[B_k] at bonds that become blocked after a sign round stays inside the present pool range (|f| < 3).
- [exact, but its premise 'a dominates' FAILS at the 5 rows: K/Vs = 0.71-0.98] One bond changing class. Let a = |H_xy||B(y)|, K0, V0 the other kept / blocked sums, D = H_xx - E'.
  kept:    f = ln[(K0 + a) / (D|B(x)| + V0)]        blocked: f' = ln[K0 / (D|B(x)| + V0 + a)]
  If a dominates (y much larger than x's other terms): f ~ + ln(a / (D|B(x)| + V0)), f' ~ - ln(a / K0). The target at x swings by ~ 2 ln(a / typical):
  with the measured partner excess e^3-e^7.5 that is ~ 6-15 in f, x c. Within a round (S fixed) the pool shows |f| < 3.2, so option 1 is safe; a sign
  round that flips a bond next to a large neighbour asks r for a dip or rise of that size across one swap. So the open question moves from the loss
  to the sign round: how many flipped bonds sit next to large neighbours (to read on paper from the flip rule, then the sign angle's labels).
- [exact, own 11-17; alpha < 0, H_xy = J_b/2 > 0, psi = S|B| real] Which rows a C-free sign round flips. Q(x) = alpha sum_y H_xy psi(y); a blocked
  (violating) y has S(y) = S(x), a kept one S(y) = -S(x), so Q(x) = alpha S(x) (Vs - K) and sgn Q = S(x) sgn(K - Vs). C-free flips x  <=>  Vs > K.
  (CORRECTED 11-17 by the sign angle, PLAN_sign 09f3f9b, checked by the lead: psi'(x) = alpha S [(D - E')A - (K - Vs)], E' = E + 1/|alpha|; the full
  label flips x iff (D - E')A(x) > K - Vs, so it DOES flip every Vs > K row (C-free flips are a strict subset: 38,959 of 181,330 on table_r3, 0
  exceptions). My first version assumed the C term dominates in the bulk; there the K - Vs term does. Also: two mutually violating isolated rows
  flip together and stay violating (period-2 cycle under simultaneous update; frequency not counted).)
- [cond met: Vs > K at all 5 rows, Vs/(D b) = 28-503 (Astra); conclusion 'target RISES e^3-7.5' WRONG, see review block] Our worst rows have f < 0 with partners e^3-7.5 larger: Vs > K, so C-free flips
  them. After the flip the big partners become kept: f'' ~ ln(Vs_old / (D|B| + K_old)) ~ + (3..7.5). The target then wants x to RISE by that much,
  where B_0 made x small by cancelling images. So the rows the sign round moves are exactly the rows where the base is most fragile: option 1 is
  clean inside a round; across rounds the base must change where the wall moved, and changing it with ln|B| meets 1/kappa again.
- [conj, ~35%, idea only; later list] Tie the wall to S instead of to image cancellation: |B| = |B_smooth| x w(x), w an explicit factor computed from
  the current sign and smooth amplitude (e.g. a function of K/Vs). Deciding equation: whether the FN fixed point at a wall row is a function of
  (K, Vs, D) alone (from the local equation above, |B(x)| = (K - Vs)/D when K > Vs). Killed if the exact 6x6 FN state at wall rows departs from that
  local formula by more than its bulk scatter.
- [exact ONLY at the loop's fixed point, guide = its own FN amplitude (Astra: FN's V uses the guide's ratios, V_k = Vs[B_k]/b_k, so phi itself need not obey K[phi] > Vs[phi])] Local FN equation (stoquastic frame of S, H_xy > 0): (H_xx + Vs/|phi(x)|)|phi(x)| - K = E|phi(x)|, so
  (H_xx - E)|phi(x)| = K(x) - Vs(x), with K, Vs built from phi's own neighbours. Since H_xx - E > 0 (PLAN_sign: D - E_REF >= 17.9), the exact FN
  state satisfies K(x) > Vs(x) at EVERY configuration. Rows with Vs > K are rows where the amplitude is inconsistent with its sign; no change of
  |B(x)| alone repairs them (K, Vs do not contain B(x)), so the Jacobi target there only drives x toward 0. These are exactly the rows C-free flips.
  Statement: amplitude-sign consistency  <=>  K(x) > Vs(x) for all x; the near-wall storage problem and the sign problem meet at this inequality.
- REVIEW 11-17 (Astra cj-7018a90b8916, /scratch/a/A.Otaifi/astra/cancel_review/ASTRA_cancel.md, scripts audit_data.py, audit_direct.py;
  image log-amplitudes reproduced to 3e-13, f_J to 2e-13, break test passed). Lead's adjudication, by equation:
  (1) accepted: the per-row gradient is 2 w e Re(grad B / B) plus pair terms; the residual and direction matter, the norm is a bound.
  (2) accepted: f_J -> ln(1/eps) needs D b + Vs = Theta(b), i.e. no large blocked neighbour; small b alone does not give it.
  (3) accepted with numbers: 1/kappa law holds statistically, log-log slope -0.99 (r9), -1.23 (r5); for B use kappa over all 512 terms
      (row 1481899: 0.0034, not P_1's 0.0006, because lam P_2 un-cancels there).
  (4) accepted: 'freezing necessary' withdrawn; 'true wall' withdrawn; K > Vs holds only at the loop's fixed point (guide = own FN amplitude).
  (5) measured at the 5 rows: K/Vs = 0.958, 0.711, 0.985, 0.729, 0.978, Vs/(D b) = 503, 29, 67, 28, 81. The rows are BALANCED: kept and blocked
      neighbour sums nearly equal, both much larger than D b. Local FN formula b ~ (K - Vs)/D: small because two big sums nearly cancel.
  (6) consequence, own [exact given (5)]: flipping S(x) swaps K and Vs at x, so f goes from ~ ln(K/Vs) to ~ ln(Vs/K): swing 2|ln(K/Vs)| =
      0.03-0.68, x c = 0.1-2.7. My e^3-7.5 'rise after the flip' assumed one dominant neighbour; it is wrong here. At these rows option 1 survives a
      sign round. Neighbours' targets move by |H| b_x, negligible.
  (7) accepted: frozen pair weights still carry 1/b_x (rho = (Vs/b_x) e^{...}); max w_b 250 at the worst row, pool max rho 16131: the frozen
      regression can be stiffly weighted, though no 1/B moves with theta.
- [own 11-17, from (5)] Physical reading: the worst rows are configurations nearly ON the node of S, where the exact amplitude is small because
  kept and violating neighbour sums nearly cancel; the network reproduces that smallness by cancelling its symmetry images. Two cancellations,
  one in the physics (K - Vs) and one in the ansatz (sum_g t_g), at the same rows.
- [exact, own 11-17, PI asked: why not (T - copy)^2] Three losses near a cancelling row (b = |B_theta(x)| ~ T, kappa small):
  plain square  (|B| - T)^2        : grad 2 (|B| - T) d|B|, no 1/B, but weight T^2: relative error delta costs T^2 delta^2. The FN energy weighs it
                                     ~ |H| phi(x) phi(y) delta^2, so the near-node rows are under-weighted by phi(y)/phi(x) = e^3..e^7.5 (20-1800x).
  relative sq.  ((|B| - T)/T)^2    : grad 2 (|B| - T)/T^2 d|B|: divides by the FIXED T, not the moving B; residual LINEAR in |B|.
  log           (ln|B| - ln T)^2   : grad 2 res d|B|/|B|: divides by the moving B.
  Expansion in delta theta: log residual = O . d + [ (d^2 B)(d,d)/(2B) - (O . d)^2 / 2 ], the last term ~ (|grad B| |d| / B)^2 ~ (|d|/kappa)^2,
  so its linear region is |d| <~ kappa (bulk scale). Relative-square residual = (grad B . d + (d^2 B)(d,d)/2)/T: both orders carry the same 1/T,
  their ratio ~ |d^2 B| |d| / |grad B| is kappa-free: linear region on the bulk scale. 1/kappa remains only as a FIXED heavy weight.
  [cond: |B| smooth enough, i.e. B not crossing 0; for complex B, d|B| = Re(e^{-i arg B} grad B) is bounded by |grad B|, direction ill-defined at 0.]
  Reading: the trouble is the curvature of ln near a cancellation, not the weighting; relative-square keeps the weighting and drops the curvature.
  Precedent: Astra ranked amplitude L2 / Hellinger second (cj-5ca32ee74fb4); auditor's asinh(B/s) is the same idea with a fixed scale s.
- NEXT EQUATION: with the clipped bounded target and frozen base, the remaining stiffness is the fixed pair weight rho ~ Vs/b_x (max 16131):
  is it needed (it is the FN bond weight) or can it be bounded without biasing the stored energy? Write E_FN's dependence on the pair term first.
- (superseded next step) NEXT EQUATION: what r should be near a spurious zero: r* = ln(phi/B_k) vs c f_J; does a c(x) that tends to 1 as f_J grows (c f_J in the bulk, f_J
  near zeros) follow from the FN equation itself, or is it a patch?
- Assumed, not shown: that the frozen-base product stays cheap enough over rounds (one extra head per round), and that persisting zeros of B_k cost less
  than the 1/B gradient does.

## Test T-Lfix (pre-registered 2026-10-27, PI asked; before any run)
Does the pair loss L_fix (board 15) store the fixed-node state better than the log loss in our code (board 13), same network?
- Target phi = exact FN state of a fixed guide (4x4: the existing exact loop targets; 6x6: the G-tax target phi_0 or an O2b round).
- Arms, same network class (RBM+PP, has PP zeros), same parameters, same optimiser and step budget, 3 seeds:
  (A) log loss as per_sample in ansatz/code/c2le_gtax.py l.471-473; (B) L_fix. Optional (C): L_fix on a log-only net
  (positive RBM x Marshall sign) to separate loss from architecture.
- Readout: tax_k = E_FN(K_k fit) - E_FN(K_k phi), k = 0, 1, 2; plus fit error by |phi|^2 decade and by pair class.
- Pass: tax_2(B) <= 0.5 x tax_2(A) at both sizes. Kill: tax_2(B) >= 0.9 x tax_2(A) at both sizes. Between: inconclusive.
- 4x4 on workstation CPU first. 6x6 needs a GPU job: launcher read by the lead before submission.

## Test T-flat (pre-registered 2026-10-27, PI said yes; before any run)
Does flattening the walker distribution cut the walkers needed for a usable E_FN gradient (issue 3)?
- Exact 4x4, the guide(s) used in C15 / Tim seq48 (early Marshall-like guide and a near-fixed-point guide), exact phi_FN.
- Exact E_FN gradient (Hellmann-Feynman, board 16; C15 force F(x) = sum_viol K_xy a(y)/a(x) [O(y) - O(x)]) as reference.
- Sampled estimator: draw M configs i.i.d. from g phi / Z with g = |a|^beta, beta in {1, 0.75, 0.5, 0.25}; reweight to phi^2 exactly
  (w = phi/g, self-normalised); gradient estimate = weighted mean of F. Cosine with the exact gradient vs M; M_0.7 = walkers for
  mean cosine 0.7 (fit over M = 1e2..1e7, 20 repeats); also effective sample size per M.
- Pass: M_0.7(beta best) <= 0.1 x M_0.7(beta 1) at the near-fixed-point guide. Kill: >= 0.5 x. Between: inconclusive.
- Check: beta = 1 reproduces C15's order of magnitude (8e4 early, 2.5e8 near fixed point, on that guide) or the deviation is explained.

## Step back 2026-10-27 (PI: "you should excel at training networks on this")
1. Verdict: continue the two tests, but add the missing control: an ideal ceiling arm in T-Lfix, training the same
   network directly on the exact E_FN(K_2 fit) at 4x4 (full sum, no sampling, no loss proxy). Without it a T-Lfix
   miss cannot tell a wrong loss from a network class that cannot represent the near-node structure.
2. Thought that changed it: as an ML problem this is task-metric loss + coverage of a rare boundary + the right
   inductive bias, and we never measured the class's best possible tax (representability). G1/G14 gave 7.06e-6
   with an oracle sign at 6x6. That hints the class itself may be the floor.
3. Cheapest test: the ceiling arm, minutes of CPU at 4x4. Independent Opus review of the logic requested by the PI.
Literature: one arXiv query (neural guide + lattice fixed node + guiding) returned nothing; guess stands as unpublished.

## Independent review 2026-10-27 (Opus, ML/NQS; verified by the lead where marked) - CORRECTS issue 1
- VERIFIED (c2le_gtax.py l.426-464): our 6x6 copy loss is NOT psi^2-weighted. Configurations are drawn equally per decade
  (-5..-13), and bond terms are drawn proportional to |H| phi(y)/phi(x) on violating bonds: already FN-shaped (small-error
  form of board 5). So "the loss weighs errors by psi^2" is wrong for our code. Issue 1 as stated does not explain the G-tax.
- Derived by the reviewer: E_FN[psi_T] <= <phi|H_FN[phi]|phi> + L_fix (exact upper bound at frozen phi, same signs).
  Storing with L_fix is therefore majorise-minimise, and its fixed point is a stationary point of E_FN over the class,
  the same as direct E_FN training (C15). Issues 1 and 3 are one issue. No loss can beat the class's E_FN ceiling.
- 4x4 T-Lfix as built was limited by a sign floor and by bulk capacity (392 parameters for 441 orbits), not by the node.
  Amended: target sign in all arms, tax_0 primary, ceiling arm via the stage (c) code at cfc2681.
- T-flat crashed at import (Python 3.9) and measured in table space. Not rerun; redesign in network parameter space if needed.
- New deciding number: R_ceil = tax(ceiling) / tax(warm start). R_ceil >= 0.7 means the network class is the floor.
  Then storage is a representability (ansatz) question, not a loss question.
- Reviewer's ideas, not in TRIED_IDEAS: shell-only flattening, g = |psi_T| max(1, eps max_viol |psi_T(y)/psi_T(x)|)
  (lattice Attaccalite-Sorella / Pathak-Wagner); zero-variance control variates for the force (Assaraf-Caffarel 1999/2003).

## Bond split of the 6x6 refit tax (O2b round 1; worker, ws CPU; raw split read by the lead; /scratch/a/A.Otaifi/ansatz_runs/bond_split/)
- Target phi_1 and copy a_2 share the oracle sign s2, so the sign-mismatch share is 0 by construction.
- Measured zero-hop E_FN(copy) - E_FN(phi_1) = 1.999e-5 per site (own LOBPCG matches stored 2.98e-5 / 9.85e-6).
- Frozen-phi pair cost (board 5 summed) = 3.53e-4 per site, 17.7x the measured difference: at a real 6x6 fit the bound is very
  loose (the FN state relaxes), so its decomposition is indicative only.
- Share of that cost, by r* decade x r/r*: 75% from r < r*/3 (the copy's small side too SMALL, an over-deep wall), only 3.5% from
  r > 3 r* (copy too large). r* mostly 1e-2..1e-1 (52%) and 1e-1..1 (33%); under 0.3% below r* = 1e-3.
- By phi(y)^2 decade: peaks at 1e-9..1e-12 (75%). J1 bonds 65%, J2 35%.
- Reading: the tax is not the "near-node r >> r*" regime; it is moderate-ratio pairs in the tail where the copy makes the small side
  too small. Checks: copy = phi gives 0; phi e^{0.01 g} matches board formula to 1.00005; noise 0.3 gives 2.1e-3; 1% sign flips
  route 5% to mismatch classes. Round 3 running (unchecked).
- Follow-up pre-registered (2026-10-27, before results): hypothesis "FN raises the rare small sides relative to the guide and the copy
  does not follow". In the r < r*/3 class: SUPPORTED if rho_g < rho_phi for >= 70% of cost and median f < 0.5 (f = how far the copy
  moved from guide toward phi); REFUTED if rho_g >= rho_phi for >= 50% of cost or median f >= 0.8. Also which side (x or y) is wrong.
- Follow-up RESULT (worker, raw JSON read by the lead; /scratch/a/A.Otaifi/ansatz_runs/bond_split/guide_split_round_{1,3}.json):
  REFUTED by the pre-registered rule. In the r < r*/3 class (75% of cost in round 1, 84% in round 3) the guide does NOT sit below phi
  (rho_g < rho_phi for only 45% / 44% of cost), and FN does not lift the rare part (ln phi - ln g ~ 0, slightly negative). The
  error is the COPY's small side x being far below phi: e_x cost-weighted mean -6.2 (round 1), -8.9 (round 3), median -6.0 / -10.6;
  count median only -1.6. The big side y is right (e_y ~ +0.02). So the copy makes deep spurious dips (near-zeros) at configurations
  where phi has none. This fits the Pfaffian angle's hypothesis (board 19): the network's own sign sgn A disagrees with the imposed
  s2 there, so |A| sits near its own node. Round 3 split: cost/measured 28.4, J1 49% / J2 51%.

## T-Lfix 4x4 result (worker; summary read by the lead: /scratch/a/A.Otaifi/ansatz_runs/t_lfix/4x4/v2/summary.txt)
RBM+PP 392 params, target phi = FN(K_1 v_0) (C13 round 0), target sign imposed, exact full sums, LM/Gauss-Newton 400 steps, 3 seeds.
| | tax_0 | tax_2 |
|---|---|---|
| warm start | 7.77e-4 | 1.24e-4 |
| sign floor |phi| sign(v_0) | 2.01e-4 | 3.92e-5 |
| A log loss | 8.34e-5 [7.36, 8.98] | 5.84e-5 [5.11, 6.72] |
| B L_fix | 1.28e-5 [1.24, 1.35] | 4.51e-5 [3.32, 5.87] |
- Registered rule (on tax_2): B/A = 0.77, between 0.5 and 0.9: INCONCLUSIVE at 4x4. On tax_0 (amended primary readout) B cuts 6.5x.
- Reading: L_fix fixes the zero-hop FN energy strongly, but after two Lanczos steps most of the gain is gone (K_2 uses ratios
  L_fix never sees, as the review predicted). The 4x4 tax is bulk-dominated (tail-only tax_0 ~ 0 for A, 1.1e-6 for B), so it does
  not test the near-node story.
- Ceiling arm D INVALID: the E_FN-trained guides reach amplitude ranges of 1e15, the FN eigensolver fails (E_FN below E0). R_ceil
  not available; needs a convergence-checked solver or an amplitude clamp.
- 6x6 launcher prepared (t_lfix/6x6/t_lfix6.sbatch, ~5 H100-h for A and B x 3 seeds), NOT submitted.
