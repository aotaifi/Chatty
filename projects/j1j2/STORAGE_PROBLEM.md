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
NEXT-METHOD REVIEW (Astra gpt-6-astra xhigh, cj-b0203391dc9d -> /scratch/a/A.Otaifi/astra/next_method_review/REVIEW.md, table_audit.json) and
ADJUDICATION (lead, 11-17). First invalid implication (99%): 'eps^2 in the D-norm => tax/gain = eps^2'. ACCEPTED, my error: the M-R 'law'
(mr_residual.py l.226-228) is an empirical calibration of C(error)/C(no copy) on two poor copies (retention 4.6%, 5.9%), not an identity; D is
the quadratic part of a trial-vector upper bound, and the rebuilt FN energy moves at FIRST order (Astra eq. 4). Measured on our tables:
tax/gain / eps^2 = 4.7, 18.1, 7.4. The retention numbers stand as separate FN solves (98.5/97.4/97.1% for the fitted (a, c); literal a=1,
c=1.307: 98.07%, 96.27%). Also accepted: (a) rounds23 is a FIXED-oracle-sign (s2) amplitude surrogate; design B's moving Lanczos signs
(S_{k+1} = S_k sgn[1 + alpha(E_L - E)]) untested; (b) refitting c needs p labels = circular; c must come from an energy search like alpha;
(c) the 1/(1-rho_slow) reading is conditional (D uses blocked edges, Jacobi kept edges); scalar c f_J captures 86-88% of the post-Lanczos
correction, not 99.7%; (d) literal log-D loss brings back the moving 1/B; the frozen-relative pair loss sum_b h T_x T_y (t_x/T_x - t_y/T_y)^2
does not, needs a normalisation, and its sampled moment is O(delta^2) only if each blocked edge is drawn from its bulk end; (e) A4 (8x8
transfer) unsupported: a rare tail in Born weight does not bound the gradient's second moment. Prior art: SWO (Kochkov-Clark 1811.12423)
and Ledinauskas-Anisimovas 2307.15521 (J1-J2: frozen network -> one-hop explicit target -> full-network fit, accept on energy decrease);
self-healing DMC can converge to a wrong node (2609.01301). New is only the FN-Jacobi teacher.
CERTIFICATE ON EXISTING DATA (lead, from rounds.json + rangelimit_6x6, fixed sign s2; Astra eq. 10: accept a stored chi_k only if
E_H[chi_k] < E_FN[g_k]): the literal explicit teacher passes it by a margin E_FN[G_k] - E_var[A_{k+1}] = 8.683e-6 - 7.938e-6 = 7.45e-7 (round 1),
3.950e-6 - 3.780e-6 = 1.70e-7 (round 2) /site, i.e. 5.1% and 5.0% of the round's FN gain. So consolidation may cost at most ~5% of a round's
gain in variational energy for the conservative certificate (sufficient, not necessary). Learned heads/R1' kept ~6% of the gain: far outside.
CONSOLIDATION TEST 6x6 SPEC (PI 11-17 late: 'yes test'; written before any run). Question: can the base class (complex PP with 3x3 backflow, no RBM: cx_c1fit arm ppbf, theta 3888 = Re+Im of 1296 f + 648 u; corrected 11-17 late; the class of
A_0 = pfaffian_c/runs/c1_cplx_s1/round_1, code cx_c1fit.py) store the explicit teacher A_1 (rangelimit_6x6/rounds23/lA_1.npy, literal a=1,
c=1.307, sign s2) well enough that the next round still gains? Arms, warm start = A_0 theta, same steps/budget, 2 seeds each:
 P: frozen-relative loss: cx_c1fit's per-row residual e = ln t - ln T - c replaced by r = e^{e} - 1 in BOTH the value and bond terms
    (Jacobian rows scaled by (1 + r)); everything else unchanged.
 C: control: cx_c1fit's own loss (log value + log bond) on the same target.
Grade (exact 6x6 tables, CPU, rangelimit_rounds.py machinery, sign s2): N(chi) = min_alpha E_FN[s2 chi |1 + alpha(E_L - E)|];
 retention_N = (N(A_0) - N(chi)) / (N(A_0) - N(A_1)), N(A_0) = 8.683e-6, N(A_1) = 3.950e-6 /site. Also E_FN[s2 chi], E_var[s2 chi] and the
 certificate E_var[s2 chi] < 8.683e-6. Grader break test first: chi = A_1 table must give 1.000, chi = A_0 must give 0.000.
Verdict: PASS (consolidation works in this class): best arm's two-seed mean retention_N >= 0.5. KILL: both arms < 0.25 (then on paper:
 capacity vs optimiser). BETWEEN otherwise. Loss claim (P better than C) only if the arm difference exceeds both arms' seed scatter.
Prediction [own]: P >= 0.5 at 40%, C >= 0.5 at 30%, P beats C beyond scatter at 50%. Precedent: this class's log-loss copy of an earlier round-1
 target reached E_FN 2.09e-5 vs 1.22e-5 with the rows below 1e-9 spliced exact: the copy's loss sits in near-node rows.
Budget: 6 A40-hours (or 3 H100-hours) for the four fits; launcher read by the lead before submission.
HOLD REVERSED (PI 11-17 late; lead's correction). PI: 'the October failure probably suffered from the 1/B divergence' and 'the task was
fitting the e^r add-on to e^{c f}'. Both right against my hold: (1) L-SWO refit the whole network with the LOG loss, so its 37% may be
loss-limited, not capacity-limited; Theta v2 is a first-order tangent test and October's own records show nonlinear refits bank more than
the tangent (11-18% vs 0-4%), so 'capacity-limited' was not established and 'VERDICT ON COPYING' above is WITHDRAWN. Arms P vs C of the spec
test exactly the PI's hypothesis (same network, same target, relative vs log loss). (2) The trained e^r add-on on the frozen base fitted to
the formula was never run: rung 2b is first-order only; M-R (10-28) trained a small RBM add-on (alpha 4/16) on ln(phi/A): ~6%. Added arm:
 H: frozen A_0 x e^{r_theta}, r_theta a ViT log-amplitude head (cx_c1fit 'vit' architecture, last layer zero-initialised so r = 0 at step 0),
    target ln A_1 = ln A_0 + l + 1.307 f_J, cx_c1fit's own loss on ln t = ln|A_0| + r (base frozen -> no 1/B in the gradient).
Same grade and verdict rule as the spec, per arm. Extra reading: P > C beyond scatter supports the PI's 1/B hypothesis for October; H vs P
says add-on vs full refit. Predictions [own]: P - C > scatter at 50%; H >= 0.5 at 35%. Budget raised by the lead to 10 A40-h (5 H100-h) for
six fits (P, C, H x 2 seeds).
SCOPE (PI 11-18): the run is ARM H ONLY (2 seeds, <= 4 A40-h). Arms P vs C dropped: the 1/B blow-up of the log loss is derived on paper
(d ln B/d theta = (dB/d theta)/B, large wherever B is small), a run does not need to confirm it, and H's frozen base has no 1/B by construction.
ARM H DATA STEP (lead, 11-17 late, existing data, before the run; features/feat.npy cols l, fJG; la_table of A_0; norb weights):
add-on target r* = l + 1.307 f_J[G]. Under A_0^2 weights: mean 0, sd 0.0103; rows with |r* - mean| > 0.5 are 3.23M of 15.8M orbits (20%) but
carry 1.9e-5 of the weight and 32% of the variance (> 2: 0.4M orbits, 1.8e-6 weight, 18%). Under p^2: sd 0.0088, > 0.5 holds 6.8%.
Unweighted quantiles 1e-4..0.999: -7.39, -5.06, -2.80, -0.02, 1.20, 2.36. corr(r*, ln A_0) under p^2 = 0.056.
Reading: the add-on is ~0 on the bulk and must learn O(1)-O(7) excursions on a fifth of the configurations that hold ~2e-5 of the Born
weight. A Born-sampled fit cannot see them; cx_c1fit samples by amplitude decade (L-SWO record: band-uniform over decades -5..-13), so
they are in H's training set. Prediction for H: its error concentrates in the deep decades; if H fails, compare per-decade error first.
TEACHER CHECK A1 (rung 2, CPU, written before computing): rounds 1-3 from A_0 with MOVING signs S_{k+1} = S_k sgn[1 + alpha_k(E_L - E_k)] and
c_k from a 1-D E_FN search (grid 0.5..3), alpha_k from the same 1-D search as rounds23. Kill: retention < 90% or certificate margin
E_FN[g_k] - E_var[next] < 0 in any round. Prediction [own]: pass at 55% (moving signs add isolated rows the teacher does not fill).
CONSOLIDATION TEST HELD (lead, 11-17 late, before any staging; recall: 'relative loss, consolidation, SWO, moving sign' -> L-SWO 10-30,
Theta v2 11-01, label-free loop report 2026-10-31, R1' linear-region record above). The 6x6 spec above repeats L-SWO (refit the PP to its own
Jacobi-corrected state, exact tables): it stored 37% of a round's gain, and the loop saturated after ~3 rounds because the correction is not
in the PP's tangent space (Theta v2: 0-4% held-out). Prediction for the spec from these records: retention_N 0.3-0.4 = BETWEEN at ~70%; no
outcome likely enough to change a decision -> not run. The new loss does not fix 8x8 either [exact, from existing data]: at the five worst R1'
rows the displacement the fit made from step 0 to 150 changed ln|B| by +1.6..+2.7, so by the triangle inequality |dB|/|B| >= e^1.6 - 1 = 4.0
at every one of them; any loss that is a function of |B| (log or relative) is then outside its step-0 linear region there (relative form: linear prediction 1 + Re(dB/B) =
-7.7..+8.7 vs actual 5..15). The relative loss removes the moving-node pole, not the coordinate problem (ln or |.| of a cancelling sum).
Proposal v2 (b) is therefore not a fix for 8x8 storage in this class. TEACHER CHECK A1 deferred (its answer matters only if the explicit
layer is deployed). VERDICT ON COPYING (lead): storing a round in this class is capacity-limited at 6x6 (11-37%/round) and coordinate-limited
at 8x8; small log-space heads store 2-3%; the explicit formula keeps 96-98% but costs two hops per round. What remains open: a class whose
small amplitudes are not made by cancellation, with inputs that carry the node geometry (PLAN l.410 'node-bond inputs untried').
NODE-GEOMETRY CHECK (rung 2, CPU, no training, written before computing): on round 1 (rangelimit_6x6 features/gram, sign s2), regress the
post-Lanczos correction v = u - l in the D-norm on SIGN-ONLY features: blocked J1 and J2 bond counts at x, their products and squares, and the
same counts averaged over x's neighbours (one hop of the sign only, no amplitudes). Pass: residual D(v - fit)/D(v) <= 0.5 (a node-bond-input
network can carry a large part of the correction without amplitude hops -> design it on paper). Kill: >= 0.9. Between otherwise.
Prediction [own]: residual 0.6-0.85, pass at 25%. Break tests: R = v gives 0, a random feature column ~1.
NODE-GEOMETRY RESULT (11-18, worker, lead re-read node_geom/node_geom.json; script rangelimit_nodegeom.py 598ad6c): BETWEEN by the written rule
(0.5 < r < 0.9). Residual D(v - fit)/D(v) on v = u - l: x-counts 0.956, + neighbour means 0.940, all 44 with squares/products 0.849 (CV 0.858);
f_J alone 0.121; (iii) + f_J 0.088. Breaks: R = v 0, random 0.99995, half-flipped sign worse at every level (0.947 vs 0.849). Reading: the sign
pattern alone carries ~15% of the post-Lanczos correction; the rest needs the network's own amplitude ratios (f_J). For arm H: a spins-only
add-on must rebuild those ratios to learn the target. Side check, closed; does not change arm H.
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

8x8 TRAINING-ROW SAMPLING, PAPER CHECK (lead 2026-11-18 01:10; PI 'Go'; Astra risk (3) of 10-30, never resolved).
Question: at 8x8 nothing is enumerated; the add-on target r* lives on rows holding ~2e-5 of |B|^2. Can the fit draw its rows from |B|^2?
Derivation (sign fixed, so the set of violating bonds is fixed; H_FN[psi] = H_kept + diag V_sf[psi], V_sf(x) = sum_{y viol x} |H_xy| psi_y/psi_x):
 E_FN[psi'] <= <phi|H_FN[psi']|phi> = E_FN[t] + sum_x phi_x^2 dV_sf(x)   (variational; phi = normalised FN state of the target guide t = A_0 e^{r*}),
 so with Delta = r - r*:  dE_FN <= sum_x phi_x^2 sum_{y viol x} |H_xy| (t_y/t_x) (e^{Delta_y - Delta_x} - 1)   [exact upper bound].
 With phi ~ t: to second order dE_FN <= sum_{violating bonds} |H_xy| t_x t_y (Delta_y - Delta_x)^2 (first-order term vanishes iff phi_y/phi_x = t_y/t_x).
Reading: (1) only differences of the copy error ACROSS VIOLATING BONDS cost FN energy, weighted by the geometric mean t_x t_y of the two
 endpoints; a rare row y bonded to a common row x enters with weight t_x t_y, not t_y^2. (2) Drawn as x ~ t^2, the per-row loss is
 sum_y |H_xy| (t_y/t_x)(Delta_y - Delta_x)^2 = cx_c1fit's bond term with x drawn from t^2 instead of decade-stratified; its variance is
 sum_y |H_xy|^2 t_y^2 Delta^4 (no 1/t_x divergence), so the rare rows need not be DRAWN, they enter as bond partners of drawn rows.
 (3) x ~ t^2 at 8x8 = x ~ A_0^2 (plain MCMC of the frozen guide) with importance weight e^{2 r*(x)}.
 (4) Labels are fixed (frozen base) -> one dataset computed once; each row needs r* at x and at its violating partners y, i.e. f_J at y =
 neighbours of neighbours (two levels; measured 137x plain sampling at 8x8 for the two-hop neighbourhood).
Not covered: the Lanczos factor in the grade N uses E_L over ALL bonds (not only violating), so the bound covers the FN part only; phi ~ t;
 the 6x6 run uses decade stratification (PI's spec, unchanged), which weights rare x 1/9 per decade, far above t_x t_y.
EXISTING-DATA CHECK (written before running; 6x6 exact tables, CPU minutes): (a) Kish n_eff/n of the weight e^{2 r*} under A_0^2;
 (b) share of the bond-energy measure M = sum_viol |H| t_x t_y captured by bonds whose higher endpoint has t^2 above 1e-9, 1e-11, 1e-13
 (normalised t); (c) relative variance per row of the x ~ t^2 bond-loss estimator at Delta = the 6x6 ViT fit's error once it exists, else
 at Delta = r* (the zero add-on). PASS (row drawing from |B|^2 is feasible at 8x8): (a) n_eff/n >= 0.5 AND (b) >= 0.99 above 1e-11
 AND (c) rows needed for 10% error on the loss <= 1e5. KILL: (a) < 0.1 or (b) < 0.9 above 1e-11. Prediction [own]: pass at 65%
 (r* sd 0.0103 under A_0^2 says (a) is near 1; (b) is the uncertain one).
 CHECKED (lead 01:45, paper-check skill step 1): exact bound = Rayleigh-Ritz on H_FN[psi'] with the trial vector phi, plus ten Haaf et al.
 PRB 51, 13039 (1995): the kept off-diagonal part depends only on the fixed sign, so H_FN[psi'] - H_FN[psi] = diag(dV_sf) exactly (conditions
 met: sign fixed, violating set fixed). Second route: E_0(H + lam V) is concave in lam (minimum of linear functions), so the first-order
 (Hellmann-Feynman) term is an upper bound - same result. Limits: Delta = const -> 0 (ratios unchanged) and Delta = 0 -> 0, both hold.
 CORRECTION to my second-order form: the linear term per bond is |H|(phi_x^2 t_y/t_x - phi_y^2 t_x/t_y)(Delta_y - Delta_x); it vanishes only
 if phi = t. Here phi != t (E_var[A_1] 7.94e-6 vs E_FN[A_1] 6.57e-6 /site), so the error cost is first order in Delta in general and
 't_x t_y' is the weight only at phi ~ t. The SAMPLING claim does not need phi ~ t: drawing x ~ phi^2 and enumerating its violating
 partners covers every directional bond term of the exact bound; per-draw variance sum_y |H|^2 (phi_x/t_x)^2 t_y^2 (e^d - 1)^2 stays bounded
 when phi/t is bounded. Using A_0^2 draws needs the weight phi^2/A_0^2 (~ e^{2 r*} if phi ~ t): data check (a).
 Confidence: exact bound 97%; sampling claim 85% (open: phi/t bounded on the rare rows; e^{2r*} weights). Step 1 convinced -> steps 2-4 proceed.
 DATA CHECK RESULT (lead, verified from sampling8.json sha 31e6788a; /scratch/a/A.Otaifi/ansatz_runs/rangelimit_6x6/sampling8, script b8177513):
 (a) Kish n_eff/n 0.9996 (pass); (b) bond-measure share with higher endpoint t^2 > 1e-11: 0.9943 (pass; > 1e-9 0.889, > 1e-13 0.9999;
 lower endpoint below 1e-9 carries 0.255 -> rare rows do enter as partners); (c) at Delta = -r* (zero add-on) Var/E^2 7.6e5 -> 7.6e7 rows
 for 10% (FAILS the 1e5 line). Sanity: M two routes agree to 1e-16. Break test (shuffle r* within A_0 decades) bites (c) (8.3e4/8.9e4) but
 not (b) (3rd decimal): my design fault, (b) depends on t ~ A_0, not on r*; (b) is untested by a break version. VERDICT: BETWEEN, not pass.
 Diagnosis (equation): Var/E^2 ~ E[L^2]/E[L]^2 with L_x ~ sum_y |H| (t_y/t_x) Delta^2; at the zero add-on Delta = r* up to 15 on rare
 partners, so L^2 ~ Delta^4 concentrates on few rows. The written rule evaluates (c) at the trained fit's residual once it exists:
 recompute (c) at arm H's fitted Delta (CPU, after the 6x6 verdict). The 8x8 sampling claim stays open on (c).
 KNOWN-CASE CHECK (verified from /scratch/a/A.Otaifi/ansatz_runs/bound_toy/results.json sha e380e166, bound_toy.py 9a950b11): exact bound
 0 violations in all tests (dense random H n 60/200, 4x4 J1-J2 J2 0.5 with three bases; Delta scales 0.01-3 + rare-row); never saturated
 (max excess -4.4e-6 |E|). Broken versions fail: sign of the correction flipped (87-200 of 200 per family), sum over all bonds (32-112 of 200
 at small Delta). 4x4 base s = sign(GS), t = |GS|: E_FN = E_0 to 1e-14 (Marshall is not the GS sign at 0.5: E_FN -8.3415 vs E_0 -8.4579;
 my brief's premise was wrong). Second-order 't_x t_y' form: actual/predicted ~0.6 (J1-J2, phi = t), sign-indefinite on random H (phi != t),
 confirming the 01:45 correction: use the exact first-order bound, not the quadratic form.
 STATUS: paper check DONE for the bound and the sampling claim's math; the 8x8 sampling claim is BETWEEN on data ((c) open, above).

ARM H STEP REJECTION, PAPER CHECK (lead 2026-11-18, on the staging worker's finding: tau 0.05 rejects every step at B 256 CPU).
Derivation: consol_fit's arm H forms r(x) = ln|(1/16) sum_g exp(z_g(x))|, z_g = complex ViT output (sum_k w_k ln cosh(amp_k + i sign_k)) on the 16
 symmetry images (logA_X, cx_reanchor.sym_logA, minus LOGC = ln 16). Exact: ln mean_g e^{z_g} = mean z + (1/2) Var_g(z) + O(z^3) (complex variance),
 so r = mean_g Re z_g + (1/2)[Var_g Re z - Var_g Im z] + O(w^3): NONLINEAR in w_out even at step 1, when only w_out moves (body Jacobian 0).
 Gauss-Newton uses only the linear part, so its step misses the curvature (1/2)(Var Re - Var Im) of the image sum (phase interference).
Checked on the worker's own scan (step 1, trust scale 1, only w_out moving; dev/out_f progress): loss 5.0977e-3, 4.9749e-3, 4.7891e-3, 6.2418e-3 at
 s = 0, 0.1, 0.3, 1. A parabola through s = 0, 0.1, 0.3 predicts 4.765e-3 at s = 1; observed 6.242e-3: the loss is not quadratic in w_out,
 as the derivation says. Solver residual equals the loss at s = 0 (5.097729e-3 both), so no solver/loss mismatch.
Reading: the spec asks for an add-on e^r with r real; the image sum of complex amplitudes is the full-amplitude ViT's symmetrisation
 (cx_c1fit arm 'vit' modelled a whole amplitude), not an add-on's. For a positive add-on the symmetric form is r(x) = mean_g Re z_g(x):
 exactly invariant, r = 0 at w_out = 0, linear in w_out (GN exact at step 1), same ViT body and head (fixed ansatz unchanged).
 Lowering tau to 0.005 (worker's fix) only shrinks steps against an artefact; at 400 steps it would test the budget again.
Prior results: recall (sym_logA, image sum, interference, logsumexp): nothing. Literature: not needed (exact Taylor identity).
Known case + broken version (CPU, worker): step-1 scan with r = mean_g Re z_g must be a parabola (s = 1 value within 2% of the
 3-point prediction) and lower than s = 0; the current image-sum head is the broken version (observed 31% off). Plus: r(gx) = r(x) to
 1e-12 for all 16 g; r = 0 and ln t = ln A_0 at step 0; 5 CPU steps at tau 0.05, B 256: >= 3 accepted.
Next rung: if all pass, restage H with this head, tau 0.05 / tau_min 0.01 unchanged, submit. If the parabola fails: stop, re-derive.
Confidence: derivation 95%; that the new head removes the step rejections at tau 0.05: 75%.
RESULT (lead 08:15, worker CPU B 256, commit 0d8fb71 sha 5020444f, sym='mean64' = mean over 16 images x 4 patch offsets of Re z; numbers in
 runs/j1j2/consol_6x6/STAGING_STATUS.md): parabola s = 0, 0.1, 0.3, 1: 5.0977e-3, 4.9725e-3, 4.7593e-3, 4.4032e-3; 3-point prediction at s = 1
 4.4037e-3 (lead, from the rounded values; worker full precision -5.6e-8 rel) PASS, below s = 0; broken version (image-sum head) 31.0% high, FAILS.
 Symmetry 5.6e-17 (float64); r = 0 exactly at w_out = 0; 5 steps tau 0.05: 4 accepted. Memory fix (chunk_rows, same maths): float64
 equivalence 1.6e-13 / 5.9e-14, dropped-chunk break 0.99 rel; a40-24gb probe 18041302: peak 12.8 of 23.5 GB, 62.4 s/step. Status: DONE.

ARM H (ADD-ON ON FROZEN A_0), PAPER CHECK (lead 2026-11-18 05:45; collects the pieces written 11-17/18 into one block before the run starts)
Derivation: copy chi = A_1 e^{Delta}, Delta = r - r*. Exact bound (ten Haaf FN variational, proved and toy-tested above):
 N-tax <= sum_x phi_x^2 sum_{y viol x} |H_xy| (A1_y/A1_x)(e^{Delta_y - Delta_x} - 1), phi = FN ground state of A_1.
 Pairing (x,y),(y,x): first-order term = sum_pairs |H_xy| d_yx [phi_x^2 A1_y/A1_x - phi_y^2 A1_x/A1_y]; it vanishes only if phi = A_1 [exact].
 Here phi != A_1 (E_var[A_1] 7.94e-6 vs E_FN 6.57e-6, grade_break/grade.log), so the tax has a first-order part in the bond differences d,
 which the cx_c1fit loss (squared d) controls only through Cauchy-Schwarz: |first order| <= sqrt(sum w a^2) sqrt(sum w d^2).
 Deciding term: d on violating bonds that join a deep row x to a bulk row y (weight phi_x^2 A1_y/A1_x ~ phi_x phi_y, not small).
Checked: limits Delta = 0 -> tax 0; constant Delta -> tax 0 (d = 0); phi = A_1 -> first order 0 (toy: ratio 0.6 to the quadratic form,
 sign-indefinite for phi != t, bound_toy results.json e380e166). Theorem: ten Haaf et al. PRB 51, 13039 (1995), FN energy is a variational
 upper bound for any guide with the stated sign; our case meets it (real Hamiltonian, fixed sign s2). Confidence 85%.
Prior results: M-R 10-28 RBM add-on alpha 4/16 on ln(phi/A): ~6%; L-SWO 10-30 full refit, log loss: 37%/round; rung 2b first order only;
 add-on target r*: 20% of orbits carry |r*| > 0.5 with 1.9e-5 of the weight (ARM H DATA STEP). Nothing run for a trained add-on on this target.
Known case: grader break test A_1 -> 0.9999, A_0 -> 0.0001 (grade_break/grade.log); step 0 reproduces A_0 to 9.9e-14, broken base fails.
Not covered: the loss weights deep rows by decade stratification, the tax by phi_x phi_y; a mismatch shows as falling loss with flat retention.
 The sum head is nonlinear in w_out (block above): steps may be rejected and lam_rel inflated; read acceptance in progress.txt.
Next rung (arm H, 2 seeds): PASS mean retention_N >= 0.5; KILL both < 0.25 with the convergence stop fired; BETWEEN otherwise; NOT CONVERGED
 = step cap with held-out loss still falling -> resume. Prediction (written 11-17, unchanged): H >= 0.5 at 35%. If it fails: per-decade
 error first, then loss vs retention (Not covered).
Status: DONE.
 DATA CHECK (lead 2026-11-18 ~06:15, exact 6x6 tables, job 18041074, /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/first_order/results.json
 sha 26e39876; E_FN[A_1] reproduced to -4.5e-8 rel). Per site: kappa = sum c1^2/c2 = 3.22e-6, sum c2 = 4.25e-2. Copy error Delta = s (ln A_0 - ln A_1):
   s     first     second    exact bound   first/second   (c1 flipped)
   0.03  1.71e-7   4.49e-8   2.16e-7       3.81           -1.26e-7
   0.1   5.70e-7   4.98e-7   1.07e-6       1.14           -7.08e-8
   0.3   1.71e-6   4.49e-6   6.23e-6       0.38            2.82e-6
   1.0   5.70e-6   4.98e-5   6.16e-5       0.11            5.04e-5
 Sanity: s = 1 bound 6.16e-5 >= actual E_FN[A_0] - E_FN[A_1] = 1.435e-5 (bound 4.3x loose); flipping c1 changes it (check can fail).
 Two independent evaluations of the exact form agree to 1e-14. kappa and first-order shares: 90% in A_1 decades -13..-8 (peak -10/-11),
 inside the decade-stratified training rows.
 Reading: retention 0.5 needs a tax of about half the gap; the s at which the bound reaches that (s ~ 0.15-0.3) is where the second-order
 term (what the squared loss controls) is >= the first-order one. First order dominates only for copies already better than s ~ 0.1,
 where the bound is <= 7% of the E_FN gap. DECISION (lead): keep cx_c1fit's squared loss; bound-as-loss NOT needed for the pass line.
 Not covered: a trained copy's error shape is not s*Dr; worst case |first| <= sqrt(kappa Q) could match Q at the pass line. Signature to
 watch in arm H: held-out loss falls while retention stays flat -> bound-as-loss is the next step (paper check first).
 Data check: DONE (loss choice).

ARM H HEAD CAPACITY: CAN A GEOMETRIC-MEAN ADD-ON REACH THE TARGET'S DIPS? PAPER CHECK (lead 2026-11-18 ~08:40, PI question via j1j2-discussion)
Derivation: mean64 gives e^r = prod_{g,o} (e^{Re z_{g,o}})^{1/64}, a geometric mean of positive factors: no cancellation, r is a smooth bounded
 function of the weights, so r cannot follow a log singularity ln|1 + alpha(E_L - E)| at a zero of the target factor; r ~ -c needs every image
 at ~ -c [exact]. A sum head e^r = |sum e^{z}|/64 can reach a near-zero by cancellation of O(1) terms, at the cost of the step-model
 nonlinearity (31%, block above). Deciding question: how much of the A_1 gain sits in orbits with r* < -c. Best case for a head whose
 floor is -c: target clipped below, ln t_c = ln A_0 + max(r*, -c), copy otherwise exact.
Checked: c -> inf gives A_1 (retention 0.9999, grade_break); c = 0 keeps only upward corrections. Clipping is monotone in c. Confidence 90%
 that the clipped retention upper-bounds a mean64 head limited to depth c; that a ViT reaches |r| ~ 7 on all images is NOT derived.
Prior results: ARM H DATA STEP (20% of orbits |r*| > 0.5, weight 1.9e-5); nothing on clipped targets.
Known case: c = inf must give 0.9999 (grader break test); c = 0 must differ from it (else the test cannot fail).
Next rung (data, CPU): retention_N of ln t_c for c = 0, 0.5, 1, 2, 4, inf. retention(c=1) >= 0.9 -> dips do not matter, mean64 stands.
 retention(c=2) < 0.5 -> mean64 cannot pass by construction unless it learns depth > 2 everywhere: cancel Hm (PI), paper check the sum
 head + relative-square loss (l.224-233). Between: report the curve, arm H continues. Cost ~1.5 h CPU. Prediction: retention(c=1) >= 0.9 at 60%.
Status: DONE (paper); data rung running.
 Fallback candidate if the fail branch fires (j1j2-discussion 08:50, not paper-checked, ~50%): linear-amplitude mean head A_0 |h|, h = 1 + mean_g w_out.feature_g,
 relative-square loss ((h - T)/T)^2, T = e^{r*}: linear in w_out (GN exact) and can cross zero. To check on paper: false nodes where T != 0;
 weight e^{2|r*|} on deep rows vs the tax's phi_x phi_y; a signed h moves the node (outside the fixed-sign FN spec unless the PI widens it).
 Compare against the sum head + relative-square loss before choosing.
 RESULT (lead 09:52, job 18041586, /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/clip_target/grade.log + clip_stats.json; consol_grade sha 777268a9 =
 grade_break; exact tables, no MC error). retention_N of ln A_0 + max(r*, -c):
   c        0       0.5     1       2       4       inf
   ret_N    0.532   0.997   1.006   1.002   1.000   0.9999
   A_1^2 weight clipped: 0.446 (c=0), 3.5e-6 (0.5), 3.4e-7 (1), 1.1e-8 (2). Orbits clipped: 57%, 15%, 7.7%, 2.3%.
 Known case: c = inf reproduces 0.9999 (grade_break); c = 0 differs (0.53), so the test can fail.
 Verdict (rule above): retention(c=1) = 1.006 >= 0.9 -> the dips below r* = -0.5 carry none of the gain; a head that reaches depth 0.5 suffices.
 The geometric-mean head's lack of cancellation does not limit arm H; mean64 stands, arm H continues. Downward corrections to depth 0.5
 carry about half the gain (c = 0 -> 0.53), so the head must make them. Not covered: whether the trained head reaches depth 0.5 where needed.

ARM H STEPS DO NOT TRANSFER: DOES A STRONGER RIDGE FIX IT? PAPER CHECK (lead 2026-11-18 11:20)
Derivation: minSR step on batch X: dtheta = G^T (G G^T + lam)^-1 e, so on fresh rows dr(X') = K(X',X)(K + lam)^-1 e, K = G G^T [exact, linearised].
 Fresh loss change = -2<e', dr'> + |dr'|^2. lam -> inf: dtheta -> G^T e / lam = the batch gradient direction (plain gradient descent);
 then E[change] = -(2/lam)|grad L_pop|^2 + O(1/lam^2) (descent lemma; batch gradient unbiased for the population gradient) [exact to first order].
 So a large enough lam MUST give steps that are better or noise on fresh rows, unless the population gradient is ~0 against its batch noise.
 Observed (Hm_s1 ckpt_last_hist.json): lam_rel at its cap 1.0 from step 6; 148/200 worse, 13 better: the cap keeps the step in the
 interpolating (batch-fitting) regime. Deciding term: lam relative to the spectrum of K.
Checked: lam -> 0 gives exact batch interpolation (observed worse); lam -> inf gives gradient descent; sign of the 1/lam term fixed (negative).
 Theorem: descent lemma for L-smooth loss (Nesterov, Intro. Lectures, Thm 2.1.5 / Bottou-Curtis-Nocedal 2018 SIAM Rev. Lemma 4.2, stochastic
 version: expected decrease needs step < |grad|^2 / (L * (|grad|^2 + var/B))). Our case meets it locally (smooth net, log loss on fixed rows).
 Confidence 85%.
Prior results: recall (minSR ridge, lam_cap, worse fresh): nothing for arm H; t1s2_vit_17418858 (full-amplitude ViT, same minSR) learned.
Known case: lam_rel 1 from the checkpoint must reproduce mostly 'worse' (control; it is the broken version). Large lam must give
 mean dmean <= 0 if the population gradient is resolvable.
Next rung (run, a40-24gb, from Hm_s1 ckpt, 5 steps per arm): arms lam_rel 1 (control), 10, 100, 1000, and w_out-only at 1.
 PASS: an arm with >= 3/5 better or noise and mean dmean < 0 -> resume arm H with that lam cap. KILL: lam 1000 still mostly worse ->
 population gradient unresolvable at B 8208 with these features -> option 2 (local-energy input / exact Lanczos factor), paper check next.
 Cost ~25 min of one slice. Prediction: PASS at 40%.
Status: DONE.
 RESULT (lead 14:30, worker report verified against the rule; jobs 18042407/08, branch j1j2-lamfix 62f1c0e, consol_fit sha 22ba0a8b;
 /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ridge_scan/<arm>/cx_t1s.json; 5 steps 301-305 from the Hm_s1 step-300 checkpoint, same batches):
   arm        better/noise/worse   mean dmean    held-out 300 -> 305
   lam 1      1/3/1                +6.3e-4       0.011425 -> 0.011427
   lam 10     1/3/1                +1.0e-5       -> 0.011430
   lam 100    1/4/0                +5.2e-8       -> 0.011431
   lam 1000   0/5/0                -6.0e-7       -> 0.011430
   w_out only 1/3/1                +2.4e-5       -> 0.011431
 Letter of the rule: PASS (lam 1000). Lead's reading: NOT a fix. The lam 1000 gain is -6e-7 per step on a loss of 1.14e-2 (5e-5 relative,
 ~1.5 sigma); 400 such steps move the loss <= 2%, and held-out rose in every arm. The rule lacked a size threshold (lead's error).
 With a large ridge the steps are safe but empty: the spins-only network finds no direction that lowers fresh-row loss by a useful amount.
 Consistent with ARM H OPTION 2: the target is a function of A_0's one-hop ratios, which this add-on has to rebuild from spins.
 Control (lam 1) gave 1/5 worse at step 300 vs 148/200 in the run's history (late, tau 0.01): known case only partly reproduced.
 Decision (lead): no resume of arm H with this add-on; the next step is the ratio-input add-on, gated on the one-hop FN grade.

ARM H OPTION 2: ADD-ON WITH THE FROZEN BASE'S ONE-HOP RATIOS AS INPUTS. PAPER CHECK (lead 2026-11-18 11:40; parallel to the ridge scan)
Derivation: E_L[A_0](x) = V_x + sum_y H_xy s_x s_y rho_y, rho_y = A_0(y)/A_0(x) over the one-hop neighbours y [exact]. So l(x) = ln|1 + alpha(E_L - E)|
 is an explicit function of the one-hop ratios [exact], and so is f_J[A_0](x) (w_xy = |H_xy| A_0(y)/K_A(x), block above) [exact].
 f_J[G] needs G's ratios = rho_y |1 + alpha(E_L(y) - E)| / |1 + alpha(E_L(x) - E)|, i.e. E_L at y: two hops [exact]. A head fed the one-hop
 ratios of the FROZEN A_0 (plus spins) represents l and f_J[A_0] exactly and f_J[G] only through what one hop carries.
 Cost: for a pair-product base all one-swap ratios come from one inverse by rank-1 updates (pp2_lemma one_swap_ratios): ~ one A_0 evaluation.
 No nesting inside a round: inputs come from frozen A_0 only. Records it rests on: (ii) l + 1.307 f_J[G] leaves eps^2 0.0033 (D-norm), retention
 0.985 direct; the spins-only head must rebuild the ratios (NODE-GEOMETRY RESULT: sign-only features leave 0.849 of v = u - l).
Checked: l coefficient 1.015 in (ii) (the Lanczos factor enters as is); limits: alpha = 0 -> l = 0; constant rho -> E_L constant. Confidence 80%
 that the derivation is right; that a trained head reaches the explicit fit's quality: not derived.
Prior results: NEXT-METHOD RUNG 2 (two-hop f_J[G]); no number recorded for one-hop-only features.
Known case (data rung, CPU minutes, features/feat.npy of rangelimit_6x6): u fitted by l alone; by l + c f_J[A_0]; by a cubic in one-hop
 features (l, f_J[A_0], ln A_0, K/V at x); break: random column, and the two-hop (ii) as the reference (must give 0.0033).
Not covered: rounds after the first (round 2's l needs A_1 ratios = rho e^{r(y) - r(x)}: r at neighbours, one network evaluation per
 neighbour, no A_0 nesting but n_bonds network calls); the 8x8 cost.
Next rung: one-hop eps^2 <= 0.1 -> design the ratio-input head (paper), then a run; >= 0.5 -> one hop is not enough, two-hop inputs or
 consolidation. Prediction: l + c f_J[A_0] reaches eps^2 <= 0.1 at 55%.
Status: DONE (paper); data rung next.
 RESULT (lead 13:40, verified onehop.json; script ansatz/code/rangelimit_onehop.py sha 6dca28cc; /scratch/a/A.Otaifi/ansatz_runs/rangelimit_6x6/onehop/):
 D-norm eps^2 of u (design B round 1), in-sample / 5-fold CV: (a) 1.055 l: 0.0263 / 0.0265; (b) 0.941 l + 0.720 f_J[A_0]: 0.0133 / 0.0137;
 (c) cubic in one-hop (l, f_J[A_0], ln A_0, Vs/K of A_0), 35 par: 0.0105 / 0.0106; (d) two-hop reference reproduced 0.00332, a 1.0145, c 1.3073;
 random column 0.99999. Rule: one-hop eps^2 <= 0.1 -> PASS. Prediction 55% held.
 Reading: the round's correction relative to the frozen A_0 is, to 98.7% in the D-norm, an explicit function of A_0's one-hop ratios.
 eps^2 is not retention (rung 2: off by 4.65x): direct FN grade of the (a), (b) tables next (CPU, ~40 min), before any head design.
 FN GRADE (lead 15:10, job 18042890, /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/onehop_grade/grade.log, consol_grade sha 777268a9; exact, no MC error):
   table                                    retention_N  N/site     E_var/site  certificate (E_var < N(A_0) = 8.683e-6)
   (a) ln A_0 + 1.0546 l                    0.845        4.684e-6   1.115e-5    no
   (b) ln A_0 + 0.9406 l + 0.7200 f_J[A_0]  0.946        4.204e-6   9.372e-6    no (8% above)
   (d) two-hop reference                    1.005        3.925e-6   7.851e-6    yes (known case)
 Broken version (standardised coefficients used as raw): eps^2 0.987. Reading (corrected after the PI, 15:40): this is the Jacobi formula itself evaluated one hop short (l exact, f_J[G] replaced
 by f_J[A_0]), not a copy; 0.946 measures only what dropping the second hop costs. It says nothing about learning the correction.
 The copy task stays unsolved (spins-only add-on 0.06); the one-hop form only halves the per-round hop cost, which still compounds.
 Open: (1) the variational certificate fails for (b); (2) rounds >= 2 with inputs fixed at A_0's one hop (block below).

MANY ROUNDS WITH INPUTS FIXED AT A_0's ONE HOP. PAPER CHECK (lead 2026-11-18 15:15; PI: "isn't fix 2 costly for many rounds")
Derivation: round k's exact correction is a functional of A_{k-1}'s one-hop ratios; A_{k-1} = A_0 e^{R_{k-1}}, so its ratios are
 rho_y e^{R(y) - R(x)}: R at the neighbours, recursively k hops of A_0 [exact]. Naive cost grows ~ z^(k-1) per amplitude (z ~ 50 one-hop
 configurations at 6x6). Flat-cost alternative: R_k(x) = g_k(one-hop features of A_0 at x), same inputs every round; then cost per amplitude is
 one A_0 inverse + rank-1 ratios for all k [exact]; the price is that R_k is a fit, not the formula. Deciding number: the share of the
 cumulative correction ln A_k - ln A_0 that A_0's one-hop features carry, for k = 2 (and 3 if it exists).
Checked: k = 1 reduces to ARM H OPTION 2 (eps^2 0.0133, retention 0.946); a fit can only lose with k (two-hop content grows). Confidence 85% in
 the derivation; the deciding number is not derivable.
Prior results: rounds23 (rangelimit_6x6/rounds23: lA_1, lA_2, p_1, p_2, gram_round1/2); L-SWO saturation after ~3 rounds (different class: PP refit).
Known case: k = 1 must reproduce eps^2 0.0133 for (b) and 0.0105 for the cubic; random column ~1; the target itself 0.
Next rung (data, CPU): D-norm eps^2 of U_2 = ln A_2 - ln A_0 (round-2 D-norm weights) by (b)-type 2-feature fit and by the one-hop cubic;
 then FN grade of the best fit, retention against N(A_2). PASS: retention_2 >= 0.8 of the two-round gain -> flat-cost rounds are viable,
 design the ratio-input add-on. KILL: <= 0.5 -> inputs must move with the rounds (periodic consolidation needed). Prediction: PASS at 40%.
Status: DONE (paper).
 RESULT (lead 17:05, jobs 18043215/18043363, /scratch/a/A.Otaifi/ansatz_runs/rangelimit_6x6/onehop_r2/, script e55adbe7; exact grades):
 eps^2 of U_2 (p_2 D-norm, in / CV): (b2) 0.388 l + 3.26 f_J[A_0]: 0.340 / 0.341; one-hop cubic: 0.199 / 0.208; R_1 unchanged: 0.518.
 retention_2 = (N(A_0) - N)/(N(A_0) - N(A_2)), N(A_2) 2.416e-6: (b2) 0.836, R_1 0.715; certificate fails for both. Cubic ungraded (extrapolates
 to R ~ 2224 on rare rows). Known case round 1 reproduced exactly.
 Reading: PASS by the letter (0.836) but round 2 already loses 34% of its D-norm with fixed one-hop inputs (round 1: 1.3%); the two-hop content
 grows per round as derived. Fixed inputs do not carry many rounds cheaply; at best 2. Side result to the copy task (PI 15:40).

ARM H CULPRIT CHECK 1: IS THE LOSS SPENT ON THE DIPS? (lead 2026-11-18 ~16:15, PI 'where could the culprit be'; script ansatz/code/armH_loss_share.py sha 816b5197, exact tables, value term e_x^2 only)
Step-0 training value loss under the run's stratified weights (1/9 per decade -5..-13 of |A_1|^2, n_r inside), e = r* - c:
 by |e|: [0,0.1) weight 0.950 loss share 0.179; [0.1,0.5) 0.049 / 0.598; [0.5,2) 0.001 / 0.207; >= 2 0.000 / 0.016.
 by decade (loss share, rms e): -13 0.738 0.140; -12 0.194 0.072; -11 0.050 0.037; -10 0.013 0.019; -9 0.004 0.010; -8..-5 < 0.002.
 VERDICT on hypothesis 1 as stated: REFUTED: dips |r*| > 0.5 carry 22% of the loss, not most. But the deepest trained decade carries 74%,
 the gain decades -8..-11 (first-order shares, DATA CHECK) about 7%.
Trained Hm_s1 table vs target per decade: rms of the fitted add-on 0.002-0.012 (target 0.001-0.14); corr(r_fit, r*) 0.37, 0.65, 0.28, 0.27,
 0.23, 0.20, 0.17, 0.13, 0.10 for decades -5..-13; value loss 2.953e-3 -> 2.915e-3 (-1.3%). The add-on learned the right sign weakly and
 is 3-10x too small; in the decades that carry the gain its correlation with the target is 0.17-0.27.
Reading: the spins-only network barely sees the target's structure (culprit 2, inputs) more than it mis-weights it. r* is 97% l
 (ARM H OPTION 2: l alone leaves eps^2 0.026), and l = ln|1 + alpha(E_L[A_0] - E)|: the add-on must compute the pair product's local energy from spins.

ARM H WITH A SUPERVISED OPTIMISER (ADAM). PAPER CHECK (lead 2026-11-18 ~16:30; PI: 'fit a known function or data points')
Derivation: arm H is regression on an exact labelled set (r*(x) on 15.8M orbits). Damped GN with P >> B: dtheta = G^T (G G^T + lam)^-1 e interpolates
 each batch (fresh-row change K(X',X)(K+lam)^-1 e, block 'STEPS DO NOT TRANSFER'); with a trust radius of 1% rms change per step it moves r by
 <= tau per accepted step [exact]. Observed: trained add-on 3-10x too small, corr 0.1-0.27 in the gain decades (CULPRIT CHECK 1). Minibatch
 Adam is a stochastic first-order method on the population loss: each step follows an unbiased gradient estimate; with a decaying step
 it converges to a stationary point of the population loss for smooth nets (Robbins-Monro conditions; Adam: Kingma-Ba 2015, convergence for
 AMSGrad-type variants, Reddi et al. 2018; in practice the standard tool for supervised fits) [cited, conditions approx. met].
 Deciding question: does the network reduce the population (held-out) loss when optimised as a regression, i.e. is the failure the
 optimiser (GN on batches) or capacity/inputs.
Checked: the ridge scan's lam 1000 arm is GN -> gradient with tiny steps (trust cap): safe, -6e-7/step; it did not test a gradient method
 with a sensible learning rate. Confidence 80% in the reasoning.
Prior results: cx_lib Adam path (warmup-cosine, optional clip) used for the 4x4/6x6 ViT fits (B10 4x4 success: 155k-param ViT); t1s2_vit (minSR) learned
 a FULL amplitude; nothing on this add-on target with Adam.
Known case + broken version: overfit 8192 fixed stratified rows (train loss < 10% of step 0 = healthy capacity/optimiser); broken: labels
 shuffled across rows -> train loss may fall, held-out must not fall.
Next rung (run, a40-24gb, same ViT body, mean64 head, same loss/target/stratified sampler): Adam, minibatch fresh rows each step, ~2 h,
 held-out every 500 steps, then table + consol_grade. PASS: held-out -30% and retention_N >= 0.5; KILL: overfit check fails (capacity/inputs ->
 ratio inputs); BETWEEN: held-out falls but retention < 0.5 -> inputs / gain-matched decade weights next. Predictions: overfit pass 80%;
 held-out -30% at 50%; retention >= 0.5 at 30%.
Status: DONE.

ARM H: OPTIMISER OR FEATURES? READOUT BOUND. PAPER CHECK (lead 2026-11-18 ~16:50, PI: 'why should B work, what does it fix')
Measured (Hm_s1 theta_ckpt_*, ckpt_last; CULPRIT CHECK 1): body moved 8% of its norm (step 50 -> 400), w_out norm 1.39, yet the trained r
 has rms 0.002-0.012 over configurations vs target 0.001-0.14, corr 0.10-0.27 in the gain decades: the output is insensitive to x.
Hypothesis [conj, ~60%]: r = mean over 64 images of a patch sum behaves like a sum of local terms (Jastrow-like); the target is ~97% l =
 ln|1 + alpha(E_L[A_0] - E)|, the log of a global sum, nonlinear where alpha(E_L - E) = O(1); a sum of local terms cannot follow it.
 If true no optimiser fixes it; B (E_L / one-hop ratios as input) makes l a 1-D function ln|1 + alpha u| of an input, which a small head
 represents exactly, and leaves f_J (1.3% of the target, ARM H OPTION 2).
Test (CPU, data, no training): freeze the trained body (theta_last) and a random body; compute the 60 mean64 channel features on the
 stratified training rows; least-squares w_out (with offset) against r* per the run's weights; report corr and value-loss reduction per decade.
 Bound: no w_out on these features does better. Known case: the bound on the trained body must be >= the trained run's own corr (0.10-0.27);
 broken version: shuffled r* -> corr ~ 0.
 Decision: bound corr <= 0.3 in decades -8..-11 -> features are the culprit -> B (ratio/E_L inputs), Adam not run. Bound >= 0.8 -> optimiser
 culprit -> run the Adam jobs. Between: run Adam and B both. Prediction: bound <= 0.3 at 55%.
Status: DONE.
 RESULT (lead 17:30, worker; /project/theorie/a/A.Otaifi/runs/j1j2/consol_6x6/readout/fit_out.txt, scripts 20c4e2e8 / 9d5dfa28; test = held-out orbits,
 8000 rows/decade, one draw): best linear w_out, test corr in decades -8..-11: trained body 0.24, 0.25, 0.21, 0.15 (value-loss -1.6% overall);
 random body 0.11, 0.12, 0.12, 0.08 (-0.7%). Known case: trained w_out reproduces r_fit to 3.4e-7; trained readout (1.4%) is at the bound.
 Broken (shuffled r*): corr ~0. Feature spread across configurations (median std over channels): trained 0.019, random 0.12-0.18.
 Rule: bound <= 0.3 -> features. Lead's reading: the bound is for FIXED features; it shows neither the random nor the GN-trained body has the
 target, not that no body can learn it. Attention makes each token's term depend on all spins, so the 'sum of local terms' argument is
 NOT a proof (lead's error at 16:50). New clue: GN training shrank the feature spread 6x below the random body (feature collapse), an
 optimiser symptom. Capacity vs optimiser is separated only by the overfit check (Adam, 8k fixed rows), now released.

RATIO-INPUT ADD-ON NETWORK (6x6). PAPER CHECK (lead 2026-11-18 ~17:50; PI: 'fit a network to the function, solve it')
Derivation: E_L[A_0](x) = V_x + sum_b h_b(x) rho_b(x), rho_b = A_0(y_b)/A_0(x), h_b = J_b/2 on anti-aligned bonds (signs: s2 convention as in
 bonds_of) [exact]; l = ln|1 + alpha(E_L - E)| [exact]; f_J[A_0](x) = function of the kept one-hop ratios [exact]. Network: per-bond embedding
 psi(rho_b, bond type, local spins) -> sum over bonds (DeepSets, lattice-symmetric because the bond set maps to itself) -> MLP -> r.
 E_L is in the span of one linear per-bond layer + sum; l is a 1-D nonlinearity of it (2-layer MLP) [exact representability];
 f_J[G] - f_J[A_0] (two hops) only approximable: data ceiling one-hop cubic eps^2 0.0105, 2-coef formula retention 0.946, two-hop 1.005.
Checked: rho = 1 everywhere -> E_L = V + sum h (Ising-like) -> l depends on spins only; alpha = 0 -> l = 0. Representability by
 universal approximation of a 1-D function (Cybenko) on bounded input [conditions met: |rho| tabled, finite on all valid bonds except nodes;
 clip ln rho]. Confidence 85% that the network reaches >= 0.9.
Prior results: ARM H OPTION 2 (formula 0.946); readout bound (spins features 0.25 corr); B10 4x4 (spins ViT, 155k params, succeeded at 4x4).
Known case + broken: (1) the 2-coef formula table graded 0.946 (ref); (2) broken: rho shuffled across configurations within decade ->
 retention must fall to <= 0.1 (spins-only level). Network initialised generic (no formula skip): what it learns is learned.
Cost note: at 6x6 rho is a table lookup. For a plain PP, all rho from one inverse (pp2_lemma.one_swap_ratios, ~1 A_0 eval); A_0 here is
 'ppbf' (backflow): no lemma, ~NB A_0 evaluations per amplitude at 8x8. Not covered: rounds >= 2 (MANY ROUNDS block), 8x8 cost.
Next rung (run, a40-24gb, Adam, same loss/target/stratified sampler, ~1-2 h): PASS retention_N >= 0.9; BETWEEN 0.5-0.9; KILL < 0.5 (then the
 inputs argument is wrong). Prediction >= 0.9 at 80%.
Status: DONE.

ARM H WITH A SUPERVISED OPTIMISER (ADAM), RESULT ov2 (lead 2026-11-18 ~17:00; job 18044227, runs/j1j2/consol_6x6/adam/ov2, branch j1j2-adam 4ceed70)
8199 fixed rows, 3000 Adam steps, 1.06 s/step. Train loss on the fixed rows 1.029e-2 -> 1.842e-3 (17.9% of step 0; pass line 10%; curve still falling at the cosine end).
Held-out 1.143e-2 -> 1.465e-2 (+28%), rising from step 300 while train had moved only -2%.
Reading: the ViT can fit these labels (capacity/optimiser not the block; the 10% miss is run length). What it learns is row-specific: the part of r* shared between
rows is too small for the spins-only features to pick up first (matches READOUT BOUND: best linear readout correlation <= 0.25). The 'skip full' branch is overridden on this
diagnosis: adam/full (fresh rows, n_steps 3600) is the direct transfer test, prediction FAIL at 75%.

RATIO-INPUT ADD-ON NETWORK (6x6), BUILD + LEAD DECISION (lead 2026-11-18 ~17:20; branch j1j2-ratnet 08e37a9, runs/j1j2/consol_6x6/ratnet/STATUS.md)
CPU tests passed: (i) r = 0 at init exactly; (ii) symmetry 2.0e-13 over 16 ops with spin_mode sym (raw spins break it, 7.8); (iii) E_L from the inputs gives l to 7.3e-12 (ln rho shuffled 7.3, sign dropped 10.1);
(iv) 400 Adam steps mb 256: train -16.9%, held-out -11.1%, corr(r,r*) decades -8..-11 = 0.03/0.11/0.18/0.27; shuffle_rho: held-out -0.26%, corr ~0.
Decision: rho_lin ON with cap 20, scale 1 (was off; the divisor 1000 made a linear rho channel invisible), so E_L is linear in an input as the paper check argues. Runs ratnet + ratnet_shuf submit after a CPU check of the linear channel (iii-lin).
Prediction: ratnet retention_N >= 0.9 at 70%; ratnet_shuf < 0.25 at 90%.
ARM H ADAM full RESULT (lead 2026-11-18 ~18:40; job 18045200, runs/j1j2/consol_6x6/adam/full): fresh rows, 3600 Adam steps. Held-out 1.1428e-2 -> 1.1363e-2 (-0.6%); GRADE H retention_N 0.0571, certificate False.
Same as GN (Hm_s1 0.056). Predicted FAIL (75%), confirmed. Spins-only ViT add-on: the optimiser is not the block (both optimisers give 0.06); the features do not carry the ratios (READOUT BOUND, ov2 memorises).

RATIO-INPUT ADD-ON NETWORK (6x6), RESULT (lead 2026-11-18 19:45; jobs 18045653 ratnet, 18045654 ratnet_shuf; runs/j1j2/consol_6x6/ratnet/{ratnet,ratnet_shuf}; branch j1j2-ratnet 74c3417)
ratnet (seed 1, true ratio inputs): GRADE H N=5.234873e-06 retention_N=0.7285, E_FN 1.2456e-05, E_var 1.7868e-05, certificate False. Held-out 1.1428e-2 -> 5.766e-3 (-50%), flat over the last 5000 steps.
ratnet_shuf (broken: inputs from a same-decade partner row): retention_N 0.0080, held-out -2.4%. Control differs by 0.72 in retention.
Against the spins-only ViT on the same loss and grader: 0.056 (GN) / 0.057 (Adam). Pre-registered ratnet line: pass >= 0.9, kill < 0.5 -> BETWEEN. Against the PI's task line (two-seed mean >= 0.5): seed 1 alone above it; seed 2 needed.
Open: the gap 0.73 vs the explicit one-hop formula's 0.946 (fit quality vs inputs) -> ceiling check, held-out loss of r = l + 1.307 f_J[A_0] on the same held-out rows.

RATNET SEED 2 + CEILING, RESULT (lead 2026-11-18 ~20:50; job 18049191 runs/j1j2/consol_6x6/ratnet/ratnet_s2; ceiling: ratnet/ceiling/, script ratnet_ceiling.py 36642bb)
Seed 2 retention_N 0.7054 (N 5.3446e-6). Two-seed mean 0.717 -> PI task line (>= 0.5) PASSED at 6x6, one round. Lead line (>= 0.9) not met.
Ceiling, held-out loss on seed-1 rows (r=0 reproduces 1.1428e-2; ratnet table reproduces 5.766e-3): r=0 1.14e-2, ratnet 5.77e-3, one-hop formula (b) 2.32e-4 (25x lower).
Decade -13: 7.15e-2 / 3.76e-2 / 1.58e-3; decade -12: 2.30e-2 / 1.03e-2 / 3.65e-4.

LOG-DOMAIN POOLING FOR THE RATIO NETWORK, PAPER CHECK (lead 2026-11-18 ~20:50)
Derivation: (b) = a l + b f_J[A_0] is a function of ratnet's inputs (exact): with rho_b = A_0(y_b)/A_0(x) and sign classes,
  E_L - E = (D0 - E) + S_blk - S_kept,  S_c = sum_{b in c} |h_b| rho_b;  f_J[A_0] = ln S_kept - ln((H_xx - E') + S_viol)  (block 'f_J = ln[K/((H_xx-E')|B|+Vs)]' divided by |B(x)|).
  So every term is ln of a sign-class sum of |h_b| rho_b, rho_b spanning e^{+-30}. ratnet = MLP(mean_b phi_b) with phi a GELU MLP of (ln rho/10, min(rho,20), ...):
  a sum of bounded per-bond outputs cannot carry S_c over many decades, so ln S_c is lost exactly where one bond has rho >> 20 (48% of random orbits; most of the deep decades that carry 74% of the loss).
  Deciding term: the pooling. Fix C: pool in log space per channel k, P_k = logsumexp_b (ln rho_b + g_k(feat_b)) with g_k learned (masked bonds -> -inf), then MLP(P_k, D0, ...).
  With g_k = ln|h_b| + ln 1[b in class c] (a large negative bias off-class), P_k = ln S_c exactly; (b) is then a smooth function of (P_kept, P_blk, P_viol, D0). Exact representability.
Checked: limits: all rho equal -> P = ln rho + ln sum e^{g}; one dominant bond -> P = ln rho_max + g (max pooling); constant shift of ln rho shifts P by the same (correct for a ratio sum).
  logsumexp pooling = softmax-attention normaliser (standard; e.g. log-sum-exp pooling in set networks) - representability is by construction, not a cited theorem. Confidence the derivation is right 85%;
  that Adam finds it: not derivable.
Prior results: none in this project; spins-only and sum-pooled nets above (0.06 / 0.72).
Known case: TO DO (CPU): hand-set C weights reproduce the (b) table (onehop_grade/tables/t_b.npy) to 1e-8 on 4000 rows; broken: sign split removed (one channel) misses by O(1).
Data check: TO DO: ratnet seed-1 residual loss per row, rows with max rho >= 20 vs < 20, within each decade. Diagnosis holds if >= 3x higher in the large-rho rows in decades -11..-13.
Next rung: C, two seeds + broken (shuffled inputs), same spec as ratnet. Pass retention_N >= 0.9 (two-seed mean); kill < 0.75 (no better than sum pooling); between -> fit-quality study. Cost ~2 A40-h. Prediction >= 0.9 at 55%.
Known case RESULT (worker, ratnet_pool_check.py b80c140, ratnet/poolcheck/): two sign classes in the (b) code (kept s2s2<0; blocked = violating s2s2>0). Hand-set pooling reproduces t_b to 2.9e-12 (4000 random orbits), 1.7e-14 (held-out); broken (one merged channel) misses by max 7.42, median 0.62. PASS.
Data check RESULT (same files): rows with max rho >= 20 = 8% of held-out rows, 75% of ratnet's loss (51% for (b)). Ratio big/small per row, ratnet vs (b): -11 8.18 vs 2.00; -12 2.50 vs 2.08; -13 17.2 vs 2.62. Pre-written rule (>= 3 in each of -11..-13) MISSED at -12. Lead reading: the diagnosis explains the largest share (75%) but not all: small-rho rows are still 12x above (b) (1.54e-3 vs 1.24e-4), worst at -12. Estimate if big rows reached (b)'s ratio: total ~1.5e-3, 4x below ratnet. Proceed with C (cheap, 2 A40-h); prediction lowered: >= 0.9 at 40%, > 0.75 at 70%.
Status: DONE (known case pass; data check done, rule missed at -12 as stated).

LOG-DOMAIN POOLING, RESULT (lead 2026-11-19 ~00:10; jobs 18050224 lse_s1, 18050225 lse_s2, 18050226 lse_shuf; runs/j1j2/consol_6x6/ratnet/lse_*; branch j1j2-ratnet 86bb141)
retention_N: lse_s1 0.8590, lse_s2 0.8288 (two-seed mean 0.844); lse_shuf (broken) 0.0072. Sum pooling was 0.717. Pre-registered: pass >= 0.9, kill < 0.75 -> BETWEEN (prediction > 0.75 at 70% held).
Held-out (seed-1 rows): r=0 1.14e-2, sum pooling 5.77e-3, lse 3.18e-3, formula (b) 2.32e-4.
Where the rest sits (poolcheck_lse_s1/part1_table.md; header there says 'ratnet_s1' but the table is lse_s1: L 3.28e-3 vs ratnet 5.77e-3; c handling differs 3% from L_ho_final):
  decade -13 rows with max rho >= 20 (154 of 4104 rows): 6.16e-2 per row (sum pooling 9.98e-2, formula 2.74e-3) = ~70% of the remaining loss. Small-rho rows improved most (-11: 1.11e-3 -> 2.33e-4).
Lead reading: exact representability did not make those rows learnable. Hypothesis (unchecked): for big rows l ~ ln|alpha| + P_max + ln|1 - e^{-|P_blk - P_kept|}| has a log singularity on the cancellation line P_blk = P_kept,
which an MLP on (P/10) inputs fits badly. Cheap check: the lse_s1 residual of those rows against |P_blk - P_kept|.

CANCELLATION CHECK, RESULT (worker, ratnet_cancel_check.py 8039635, runs/j1j2/consol_6x6/ratnet/cancel_lse_s1/)
Big-rho rows of -12/-13 (n 256): rows with |P_blk - P_kept| < 0.3 are 23% of them and carry 90.3% of lse_s1's loss there (6 rows with |Delta| < 0.1 carry 75.5%, mean loss 1.24 vs (b) 1.7e-2). Pre-written rule (>= 50% on <= 25%) HOLDS.
|u| = |1 + alpha(E_L - E)| >= 0.1 on all 912 rows of -12/-13: the target is smooth there (mean |l| 0.22). My stated mechanism (log singularity of l) is WRONG; the failure is in the inputs-to-target map.
Corrected diagnosis (equation): E_L - E = (D0 - E) + e^{P_blk} - e^{P_kept}. With P ~ ln 20 .. ln 10^3 and Delta -> 0 the MLP must form e^P (e^Delta - 1): output gain ~ |alpha| e^P along Delta, so an O(0.01) error in its
internal Delta becomes O(1) in l. The linear rho channel that would do the sum before any nonlinearity is capped at 20 - exactly on these rows (they are defined by max rho >= 20).
Fix D (paper): add signed linear pooling, uncapped: z_k = asinh( (b_k + sum_b c_k(feat_b) rho_b) / s ), c_k learned per bond (can take h_b sgn_b), s = 10. The cancellation then happens in the sum, exactly;
with c = h sgn, b = D0 - E + 1/alpha: z = asinh((E_L - E + 1/alpha)/s), and l = ln|alpha s sinh z| is smooth in z for |u| >= 0.1. Known case + broken version: CPU test (hand-set weights give l to 1e-8; cap 20 restored -> misses on the 59 rows).
Next rung: D = lse + 4 signed-linear channels, 2 seeds + broken. Pass two-seed mean >= 0.9; kill <= 0.844 (no better than lse); prediction >= 0.9 at 55%. Cost ~1.5 A40-h.

FIX D (SIGNED-LINEAR POOLING), RESULT (lead, 2026-11-19 ~02:30; verified from out/progress.txt GRADE lines and delta_today VERDICT lines)
Ran: lse + 4 uncapped signed-linear channels (ratnet_lib 4bd298f), 55000 steps, cip a40-24gb, ~1 A40-h each.
  lin_s1 18055557 retention_N 0.8956 (held-out 1.143e-2 -> 1.885e-4); lin_s2 18055558 0.8929; broken lin_shuf 18055559 0.0065.
  Two-seed mean 0.8943 -> BETWEEN (pass 0.9, kill 0.844). Broken < 0.1 as required. Seed spread 0.003.
Diagnosis confirmed: big-rho rows of decades -12/-13 total loss 9.87 (lse_s1) -> 0.353 (lin_s1); |Delta| < 0.1 rows 1.24 -> 1.21e-2 mean loss (formula (b) 1.72e-2);
  |Delta| < 0.3 share 90.3% -> 49.8%. Path: runs/j1j2/consol_6x6/ratnet/cancel_lin_s1 (column labelled lse_s1 is lin_s1).
Open (hypothesis, not a claim): lin_s1's held-out loss (1.885e-4) is below formula (b)'s (2.32e-4) yet its retention is lower (0.896 vs 0.946): the loss weights
  rows per decade, the grade weights them by their FN-energy effect, so the remaining 0.05 sits where the two weightings differ. Test (CPU, minutes): per-decade loss of lin_s1 vs (b).
Lead decision: BETWEEN, take lin_s1 as the round-1 add-on and move to round 2 (critical path); the 0.05 gap to (b) is not on it.

ROUND 2 OF THE COPY (RATIO-INPUT ADD-ON ON A FROZEN COPY), PAPER CHECK (lead, 2026-11-19 ~03:30; PI concern: does a 0.89 copy hold over rounds?)
Derivation:
 Notation: chi_1 = A_0 e^{r_1}, r_1 = lin_s1 (table /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ratnet/lin_s1/la_table.npy). Round map M (rangelimit_rounds.py): from a start a,
 G[a] = a |1 + alpha_a (E_L[a] - E[a])| (alpha_a by line search), M[a] = G[a] e^{1.307 f_J[G[a]]} (f_J with E' = E_FN[s2 G[a]]). Exact teacher A_2 = M[A_1]; round-2 teacher
 from the copy  T_2 = M[chi_1]; round-2 add-on target  r_2* = ln T_2 - ln chi_1 = l[chi_1] + 1.307 f_J[G[chi_1]].
 (1) Representability [exact]: r_2* is the same functional of chi_1's one-hop and two-hop ratios as r_1* = l[A_0] + 1.307 f_J[G[A_0]] is of A_0's. ratnet reads ln rho_b from
     base_table by neighbour lookup (ratnet_lib.py l.8), so round 2 = round 1's code with base_table -> chi_1's table, target_la -> ln T_2. No new knob. The
     inputs refresh with the round, which the fixed-input check showed is needed (MANY ROUNDS: 0.836 with A_0's inputs, eps^2 0.34 vs 0.013).
 (2) Two error sources in round 2, separated by the grade [exact, definitions]: per-round copy quality q_2 = (N(chi_1) - N(chi_2)) / (N(chi_1) - N(T_2)), and teacher
     degradation N(T_2) vs what M would give from A_1-quality starts. Cumulative retention_2 = (N(A_0) - N(chi_2)) / (N(A_0) - N(A_2)), N(A_2) 2.416e-6 (rounds23 log).
 (3) Compounding [conj, 55%]: project M onto the grade N. Exact rounds give N 8.683 -> 3.950 -> 2.416 (e-6/site). An affine contraction N -> N_inf + kappa (N - N_inf) through
     these three points has kappa 0.324, N_inf 1.680e-6. A copy keeping q of each round's gain is then N -> N_inf + (1 - q(1 - kappa)) (N - N_inf): kappa_eff = 0.396 at q 0.894.
     Check: round 1 predicts N(chi_1) 4.453e-6, measured 4.444e-6 (by construction of q). So a copy error does not add up round by round: it slows the contraction toward the
     same fixed point (rounds needed x ln kappa / ln kappa_eff = 1.22), and the stationary excess is bounded by eps / (1 - kappa) (perturbed contraction, below).
     Predictions: N(T_2) 2.576e-6 (affine) or 2.72e-6 (ratio model N(A_2)/N(A_1) = 0.612); with q_2 = q_1: N(chi_2) 2.78e-6, cumulative retention_2 0.94.
 Deciding term: the teacher's local energy E_L[chi_1] = D0 + sum_b h_b sgn_b rho_b e^{r_1(y_b) - r_1(x)}. A copy error dr enters as delta E_L = sum_b h_b sgn_b rho_b (dr(y_b) - dr(x)):
     amplified by rho_b, i.e. on exactly the big-ratio rows where round 1's residual sat (cancellation rows, decades -12/-13). If the error grows there, T_2 gains less than (3) says.
Checked: (1) limit q = 1 (chi_1 = A_1): T_2 = A_2 exactly, the procedure must reproduce lA_2 and N 2.416095e-6. (3) limit q = 0: chi_1 = A_0, kappa_eff = 1, no progress, correct.
 Sign: kappa < 1 needed; the three exact points give a convex sequence (gains 4.733, 1.534), consistent. Theorem: inexact fixed-point iteration, x_{k+1} = T(x_k) + e_k with T a
 kappa-contraction gives |x_k - x*| <= kappa^k |x_0 - x*| + sup|e| / (1 - kappa) (Ostrowski 1967, from knowledge; standard in inexact Picard iteration). Condition NOT checked:
 M is a contraction in a metric we can compute; the 1D projection onto N is a model, hence [conj]. Confidence: (1), (2) 95%; (3) 55%.
Prior results: MANY ROUNDS (fixed A_0 inputs): retention_2 0.836, R_1 unchanged 0.715 (rangelimit_6x6/onehop_r2, exact grades) - different inputs (not refreshed). rounds23:
 exact M twice, per-round formula fit eps^2 0.00145 / 0.00389 (fit (ii) on that round's own features: round 2 is 2.7x harder for the formula, tax/gain 2.6% / 2.9%).
 L-SWO (October, full PP refit, different class) saturated after ~3 rounds. Literature: searched in session memory only (iterated distillation / self-training error accumulation
 is the ML analogue; no theorem on our map). recall.sh (round 2, second round): label-free loop 10-31 (docs/reports/2026-10-31_label_free_loop_6x6.md): stored share r fell 0.37 -> 0.11 over rounds under s1, but round 2 unconverged and a full PP refit with spins-only inputs; B20 (distil the k-stack, spins-only students) <= 10%; C13 (4x4 no-hop refit) raises E_FN every round. All three are spins-only stores, the class arm H showed cannot rebuild the ratios (0.057); none binds a ratio-input add-on. Their lesson kept: a declining per-round share is the failure to watch, hence q_2 is the graded quantity.
Known case: TO DO (data rung, CPU): start = lA_1 table must give back lA_2 to <= 1e-10 and N 2.416095e-6; broken: f_J with E' = E_FN[s2 a] instead of E_FN[s2 G[a]] must miss by O(0.1).
Not covered: rounds >= 3; 8x8 cost (inputs of round k need chi_{k-1} at the NB neighbours, i.e. r_{k-1} there, which itself needs NB ratios: ~NB^2 A_0 evaluations per amplitude
 unless r is cached or the ratios are rank-1 updates); seed scatter of round 2 (round 1: 0.003).
Next rung 2 (existing data, CPU): build T_2 = M[chi_1] from the lin_s1 table, N(T_2) by the same line search. Rule: N(T_2) <= 3.0e-6 (the teacher from the copy keeps >= 0.94 of the
 exact round-2 gain 1.534e-6) -> rung 3. N(T_2) > 3.5e-6 -> the copy damages the next teacher: stop, locate rows (delta E_L above) before any fit. Between -> rung 3 with the reading noted.
 Prediction: 2.58-2.72e-6 at 65%. Cost ~1 h CPU.
Next rung 3 (run): r_2 on frozen chi_1, lin_s1's spec with base_table chi_1 and target ln T_2, 2 seeds + broken (shuffled inputs). PASS q_2 >= 0.85 (the copy holds round over round);
 KILL q_2 < 0.70 (two seeds); between -> diagnose which rows. Broken must give q_2 < 0.1. Report cumulative retention_2 too (predicted 0.94 at q_2 = q_1). Prediction: PASS at 60%.
 Cost ~3 A40-h.
Status: NOT DONE: known case (rung-2 script) to run.

ADDITIVE TARGET READOUT CHECK, RESULT (lead 2026-11-19, step-back hypothesis 'the log form puts -ln A_0 into the target'; runs/j1j2/consol_6x6/readout_additive/{readout_additive.py,out.txt})
Same arm-H features (trained ViT body / random body, 60 features, 8000+8000 rows per decade of |A_1|^2). Test corr of the LS readout, decades -8..-11:
 trained body: LOG r* 0.244 0.246 0.211 0.150 (known case = readout/fit_out.txt, reproduced exactly); ADD (A_1 - lam A_0)/s_d 0.239 0.253 0.210 0.146; ADD with A_0 as extra column 0.333 0.270 0.234 0.185 (per decade); shuffled ~0.
REFUTED: the additive target is no easier. Reason, from the same data: |r*| > 1 on 0.000 of rows in every decade -5..-12 (0.001 at -13). In the decades that carry the gain r* is small, so
 Delta*/A_1 ~ r* and the two targets are the same function. The O(1-7) excursions (ARM H DATA STEP) sit below the graded decades. The step-back mechanism was wrong for the rows that matter.
What the target is on those rows: r* ~ alpha (E_L[A_0] - E) + 1.307 f_J, E_L[A_0](x) = D0 + sum_b h_b sgn_b A_0(y_b)/A_0(x): A_0's change across all ~50 bonds. A network that does not contain A_0 must rebuild A_0's
 bond-by-bond structure to produce it (readout corr <= 0.33 for a trained ViT body; ratio inputs supply it and reach 0.894).

FLAT-COST ROUND 2: ADD-ON ON THE FROZEN BASE'S OWN RATIOS, PAPER CHECK (lead, 2026-11-19 ~09:40, takeover; PI 09:20: 'copy the new network and its corrections for the second round; freeze or not is your choice')
Question: can round 2 be stored at a cost per amplitude that does not grow with the round number?
Derivation:
 (1) Where the nesting comes from [exact]. Round k's correction is a function of the previous state's ratios. If the add-on reads chi_{k-1}'s ratios, then chi_{k-1}(y_b) needs r_{k-1}(y_b), which
     needs that neighbour's own ratios: z^(k-1) base evaluations per amplitude (z ~ 50 nonzero bonds at 6x6). The nesting is caused by refreshing the INPUTS, not by the ratio inputs as such.
 (2) Flat-cost class [exact cost]: ln psi_k = ln B + R_k(x), R_k = g_k(B's one-hop ratios at x, spins), the same inputs every round, g_k a trained network (ratnet). Cost per amplitude for every k:
     one B evaluation + z one-hop ratios of B (for a pair product, rank-2 updates of B's own inverse: O(z) after the O(N^3) B already pays; with the 6x6 backflow, low-rank updates of the
     window) + one network. The neighbours of chi_{k-1} are needed only to BUILD round k's training target (one hop, z fixed-cost evaluations), not to evaluate the stored state.
     This is the 'fixed inputs' design of MANY ROUNDS, now with a learned per-bond network instead of 2-5 hand features.
 (3) What it can and cannot represent [exact]: round 1's target l[B] + c f_J[G_0] is a function of B's one- and two-hop ratios; a one-hop-input network fitted it to 0.894 (lin_s1), formula (b) on
     one-hop features 0.946. Round 2's cumulative target U_2 = ln T_2 - ln B also needs chi_1 at neighbours, i.e. B's ratios up to two (and four, through f_J) hops: content outside the
     inputs grows per round. Deciding number: the share of U_2 a learned function of B's one-hop ratios holds, graded by FN energy.
 (4) Why it is a copy, not a recomputation (PI 11-18 objection): the stored object is g_k's weights; its inputs cost the same as one local-energy evaluation of B in every round, which any
     VMC/FN step already pays. What would be 'computing' is an input cost that grows with k; (2) has none.
Checked: k = 1 is lin_s1 exactly (target T_1 = A_1, base B = A_0): reproduces 0.8956 by construction. Limit 'g_2 = g_1' (no round-2 learning) gives chi_1: cumulative retention_2 =
 (N(A_0) - N(chi_1))/(N(A_0) - N(T_2)) = 4.239e-6/(8.683e-6 - N(T_2)), the floor any round-2 fit must beat. Cost claim (2): the Sherman-Morrison/determinant-lemma ratio
 det(I + V^T M^{-1} U) is the standard VMC local-energy route (Ceperley-Chester-Kalos 1977; Sorella's Lanczos-step VMC, PRB 64, 024512 (2001), from knowledge, not re-read).
 Confidence: (1), (2), (4) 90%; (3) the deciding number is not derivable.
Prior results: MANY ROUNDS (fixed one-hop inputs, hand features): (b2) retention_2 0.836, R_1 unchanged 0.715, one-hop cubic eps^2 0.199 vs (b2) 0.340 (more one-hop content than 2 features
 carry; cubic not gradable, extrapolated). ADDITIVE TARGET READOUT CHECK (today): spins-only features hold <= 0.33 of r*. TRIED_IDEAS AR2 (local log-additive add-on, 8x8): 0.02-0.13 -
 spins-only, does not bind a ratio-input net.
Known case: the network code is lin_s1's (ratnet_lib 4bd298f), unchanged; only target_la changes. The target T_2 = M[chi_1] is built by round_map.py with its own known case (lA_1 -> lA_2,
 worker, running) and broken version (E' of the start). Network broken version: shuffled inputs (lin_shuf: 0.0065).
Not covered: rounds >= 3; 8x8 input cost with backflow (~NB B evaluations unless low-rank updates are written); whether two-hop content of round 2 needs two-hop inputs (then: add pairs of
 bonds as tokens, still flat in k).
Next rung (run): ratnet lse_lin, lin_s1 spec, base_table = A_0 (unchanged), target_la = T_2; 2 seeds + shuffled; ~3 A40-h (j1j2 GPU use since 11-18: ~24 A40-h of 40).
 Quantity: cumulative retention_2 = (N(A_0) - N(chi_2))/(N(A_0) - N(T_2)). PASS >= 0.85 (two-seed mean): round 2 copied at flat cost. KILL <= floor + 0.05 (no better than not learning round 2).
 Between: the one-hop inputs miss round 2's two-hop content -> pair-of-bond tokens (paper first). Broken < 0.1. Prediction: PASS at 45%, between 45%, kill 10%.
Status: DONE for the paper; the run waits for the worker's T_2 known case (pass) and N(T_2).
 REVIEW + TEACHER (lead 2026-11-19 ~10:05). Teacher from the copy: N(T_2) 2.566723e-6 (worker round_map.py, /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/round2/teacher/round_map.json, alpha -0.077;
  start N(chi_1) 4.444332e-6 reproduced). Prediction written 08:47 in 'ROUND 2 OF THE COPY': 2.576e-6 (affine contraction) -> measured within 0.4%: the copy does not damage the next teacher.
  Known case lA_1 -> lA_2: worker report pending. Astra (cj-c2098bb95b57, result.md there): flat cost >99%; sufficiency of one-hop inputs unmeasured; round 2 from a fixed-input copy reaches 3 hops of B;
  predicts the planned run at cumulative 0.81 (70% range 0.73-0.89). CORRECTION (Astra, accepted): my pass line 'cumulative >= 0.85' keeps only 51% of round 2's own gain (unchanged chi_1
  already gives 0.693, the hand (b2) table 0.857). Pass line replaced by the per-round share q_2 = (N(chi_1) - N(chi_2))/(N(chi_1) - N(T_2)), denominator 1.878e-6, PASS >= 0.85 (round 1: 0.894),
  i.e. N(chi_2) <= 2.848e-6. On that line the hand (b2) table gives q_2 ~ 0.53 (N 3.4415e-6, target A_2 not T_2). GPU run withdrawn until the moment check below.

WORD-MOMENT INPUTS: WHAT ROUND k NEEDS, PAPER CHECK (lead 2026-11-19 ~10:10)
Derivation: split H = D + H_K + H_B (diagonal, kept bonds s_x s_y < 0, blocked bonds s_x s_y > 0 under s2). For a word w over {H_K, H_B, H} define the moment mu_w(x) = (w B)(x)/B(x).
 Round 1 [exact]: E_L[B] = mu_H. G = B u, u = 1 + alpha(mu_H - E). Then (H_K G)/G = [mu_K + alpha(mu_{K H} - E mu_K)]/u and (H_B G)/G = [mu_B + alpha(mu_{B H} - E mu_B)]/u, so
  r_1* = ln|u| + c [ ln((H_K G)/G) - ln(Hxx + (H_B G)/G - E') ] is a CLOSED function of the six moments {mu_H, mu_K, mu_B, mu_KH, mu_BH} (+ D, E, E', alpha): words of length <= 2.
  (fJ_and_VsK in rangelimit_rounds.py computes SK = sum_kept |h| G(y)/G(x), VA = sum_blk h G(y)/G(x): the same objects; signs as in the code.)
 Round k [exact, same algebra]: closed function of moments of words of length <= 2k (round 2: length <= 4). A network reading moments up to length L has a cost ~ z^L per amplitude, FLAT in k,
  and is exact for k <= L/2; beyond that it must approximate.
 Consequence: ratnet's one-hop inputs = length-1 moments (resolved per bond). Round 1 needs length 2: lin_s1's 0.894 already approximates the two-hop part (formula (b), length 1: 0.946 ...
  (b) is one hop short). Round 2 needs length 4; length-2 inputs cover all of round 1's correction exactly and the leading part of round 2's.
Checked: alpha -> 0, c -> 0: r* -> 0. c = 0: r_1* = ln|1 + alpha(mu_H - E)| (one hop) [ok]. mu_K + mu_B + D = mu_H [identity, a test]. Confidence in the algebra 90% (signs follow the code, to be
 checked by the known case below).
Known case (CPU, to run): from the A_0 table, compute mu_w for the length-<=2 words by sparse matvecs (BS.build K split by s2); rebuild lA_1 from them with the closed form; must match rounds23/lA_1
 to <= 1e-9. Broken: drop the alpha(mu_{K H} - E mu_K) term -> must miss by O(0.1).
Next rung (existing data, CPU): D-norm fits (rangelimit fits(): cubic polynomial, CV) of U_2 = ln T_2 - ln B on three feature sets: L1 = {mu_H, mu_K, mu_B, ln B}; L2 = L1 + {mu_KH, mu_BH, mu_KK, mu_BB};
 L2+ = L2 + the exact round-1 closed form r_1* evaluated at x. Then FN grade of each fitted table -> q_2. Shuffled-target control. Decides the input level: L1 q_2 >= 0.85 -> one-hop ratnet run;
 only L2 >= 0.85 -> length-2 inputs (cost ~z^2 per amplitude, flat in k); neither -> the polynomial fit is the limit, not the inputs: a network run would then test capacity (state which).
 Prediction: L1 q_2 0.4-0.6; L2 0.75-0.9 [own, 50%]. Cost ~2 h CPU.
Status: NOT DONE: known case to run.
 ROUND MAP KNOWN CASE (worker, round_map.py 217d0ff, /scratch/a/A.Otaifi/ansatz_runs/rangelimit_6x6/roundmap_known; lead reading 2026-11-19 ~10:30): lA_1 -> lA_2 max|diff| 7.5e-10 (6 of 15.8M rows
  > 1e-10, all at ln a ~ -35, round-off in ln SK - ln den; mean 2e-16), N(lA_1) 3.950335e-6, N(lA_2) 2.416095e-6 reproduced. Lead: PASS (the 6 rows carry no FN weight; the 1e-10 line was
  mine and too tight for e^-35 rows). Broken version (E' of the alpha=0 solve) misses by only 1.4e-5: E' moves 1e-4, too weak to discriminate - not a failing test; the WORD-MOMENT known case
  (closed form vs lA_1, broken = dropped alpha terms) is the independent check of the same formula. Status of 'ROUND 2 OF THE COPY' known case: DONE with that caveat.
 WORD-MOMENT FIT, RESULT (worker word_moments.py de2d487 on j1j2-ratnet; /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/round2/moments/{grade,fit}.json; lead read 2026-11-19 11:35).
  Identity mu_H = Hxx + mu_B - mu_K: 4.3e-11. Known case: the signed closed form FAILS (max 17.2) because lanc builds G = B|u| and u < 0 on 9.2% of rows; with a per-bond gather of |u(y)|
  it rebuilds lA_1 to 3.3e-9 (1 row of 15.8M above 1e-9). So r_1* is a closed function of length-2 moments only where u > 0 on x and all neighbours (5.7% of rows, 0.77 of the round-1 weight);
  elsewhere it needs |u| at each neighbour, i.e. E_L at y: a kink a smooth readout of moments cannot make. Broken (alpha terms dropped): misses by 2.5. Algebra block: DONE with this correction.
  Cubic D-norm fits of U_2 (CV eps2) and FN grades:  L1 0.273 -> N 3.843e-6, q_2 0.320 | L2 0.181 -> 3.334e-6, q_2 0.591 | L2+ (exact r_1*) 0.127 -> 3.042e-6, q_2 0.747 |
  shuffled L2+ 1.001 -> 1.198e-5, q_2 -4.0. Prediction (L1 0.4-0.6, L2 0.75-0.9) both LOW. Rule branch: neither >= 0.85 -> capacity question.
  Lead reading: the cubic is a weak learner here (raw moments with tails 1e4 sd; fit clipped on 1.9-11k rows); "inputs insufficient" is NOT shown. Calibration missing: the same cubic on
  round 1 (U_1 = lA_1 - ln B, exactly a function of these inputs except the |u| kink) against ratnet's 0.894 on the same target.
CAPACITY CALIBRATION (lead 2026-11-19 11:40, CPU, existing data). Same code, same features: cubic L1, L2 fits of U_1 (weights p_1), round-1 retention_N; plus asinh(x/sd) features for L2 on U_1 and U_2.
  Rule: cubic-L2 retention on U_1 <= 0.7 (<= 0.8 x ratnet) -> the cubic is the limit; q_2(L2) 0.59 is a floor and the one-hop ratnet on T_2 is the next rung (data check DONE by this block).
  Cubic-L2 on U_1 >= 0.85 -> the cubic is adequate, so round 2 needs inputs beyond length 2 (Astra's joint ratio tokens / |u(y)| gathers); no one-hop run. Between -> asinh arm decides.
  asinh-L2 on U_2 >= 0.85 -> length-2 inputs suffice in a smooth basis. Prediction: cubic-L2 on U_1 0.55-0.75 [own, 60%]. Cost ~2.5 h CPU.
 CAPACITY CALIBRATION, RESULT (worker capacity_fit.py 703361e on j1j2-ratnet; /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/round2/capacity/<arm>/{fit,grade}.json; lead re-read grade.json 13:55).
  Sanity: old L1-on-U_2 CV eps2 0.2726 reproduced. Same cubic, same raw features, on ROUND 1 (U_1, p_1): L1 N 4.512e-6 -> retention_N 0.881; L2 N 4.190e-6 -> 0.949 (above ratnet's 0.894);
  shuffled L2 1.058e-5 -> -0.40. alpha -0.1 is a grid point for all three (parabola not refined: N is an upper bound, retention a lower bound). asinh features: U_1 L2 0.849, U_2 L2 q_2 0.569 (no help).
  RULE BRANCH: cubic-L2 on U_1 >= 0.85 -> the cubic is adequate; round 2 needs inputs beyond length 2. The same learner on the same B-inputs keeps 0.95 of round 1 and 0.59 (0.75 with exact r_1*) of round 2.
  Reading [measured + the WORD-MOMENT algebra]: each round extends the reach by ~one hop of B (round k ~ words of length 2k; the |u| kink adds per-neighbour E_L). An add-on on FIXED B-inputs cannot hold
  round after round at flat cost: the PI's 11-19 objection holds in the data. Caveats: 6x6 only; pooled moments, not resolved per-bond tokens (Astra's 'complete one-hop multiset' arm not run);
  weights p_1 = rounds23/p_1.npy (differs from rangelimit_6x6/p.npy by 3% median; not checked which round 1 used). False alarm: VERDICT lines 13:21/13:23 for capacity_ws40_278219 were an OOM kill of
  one of three parallel grades (32.9 GB each); arm A regraded alone (gradeA2.log, exit 0).

REFRESH THE BASE EACH ROUND (FULL REFIT, NO MOVING 1/B), PAPER CHECK (lead 2026-11-19 15:05)
Why: CAPACITY CALIBRATION shows an add-on on FIXED B-inputs cannot hold round 2 (reach grows ~1 hop per round). Flat cost per amplitude then needs the inputs to refresh without nesting:
 store chi_k = B_k e^{g_k(Phi_{B_k})}, B_k of the fixed ppbf class, g_k a fixed-size ratio head (ratnet) on B_k's OWN one-hop ratios. Round k+1: teacher T_{k+1} = M[chi_k] by neighbour lookups
 (training-time cost z^2 B-evals per row, paid once per round; stored amplitude cost = 1 + z B-evals, flat in k). Then fit (B_{k+1}, g_{k+1}) to T_{k+1}. The head's reach is relative to the
 newest base, so it never grows; the base must absorb what the previous head held. Deciding term: can the ppbf class absorb chi_1 = B e^{g_1} (round 1 only)? That is the 11-17 spec arm P.
Derivation, loss [exact]: log loss L = sum_x w_x (ln T_theta(x) - ln t(x) - c)^2, grad = 2 sum w e (dT/dtheta)/T: diverges where T_theta -> 0 at a row with t != 0 (the PI's 1/B).
 Spec P (11-17) residual r = e^{e} - 1 = t/(T e^c) - 1 has dr = -(1+r) dT/T = -(t e^{-c}) dT/T^2: STILL a moving 1/T, now squared. CORRECTION: spec P as written does not remove the 1/B.
 Correct form (relative to the FROZEN target): rho = T e^{c}/t - 1, L_rel = sum w rho^2, d rho = e^{c} (dT/dtheta)/t: only the fixed 1/t; bounded wherever t > 0 (the teacher table has no zeros
 in the graded set: min |t| checked in the known case). Bond terms the same way: rho_xy = (T_x/T_y)/(t_x/t_y) - 1 has a moving 1/T_y; use rho_xy = rho_x - rho_y instead (first-order equal).
Checked: |rho| << 1 -> rho = e + O(e^2), so L_rel = L_log + O(e^3): same minimiser to first order, same Hessian at an exact fit; the losses differ only on rows with large error, which is where the
 log loss's gradient blows up (T -> 0) and L_rel saturates at w per row. Limit c -> c*, theta frozen: L_rel stationary in c at sum w rho e^{c}T/t = 0 [ok]. Theorem: none needed (algebra). Confidence 90%.
 What it does NOT change: first-order capacity (Theta v2 tangent test 0-4% is loss-independent at first order). It can only help if October's nonlinear log-loss refits (11-18%, L-SWO 37%)
 were held back by rows with T_theta near 0. Hypothesis, not shown.
Prior results: L-SWO 10-30 full PP refit, log loss 37%/round, saturates after ~3 rounds; Theta v2 11-01 tangent 0-4%; nonlinear refits 11-18%; precedent ppbf log-loss copy of a round-1 target
 E_FN 2.09e-5 vs 1.22e-5 with rows below 1e-9 spliced exact (loss sits in near-node rows: supports the hypothesis); arm P never run (PI scope 11-18). Second PP holds 0.24-0.47 (P23).
Known case (CPU, to run): 4x4 exact, same small class from the same warm start, three losses: log, spec-P (t/T - 1), L_rel (T/t - 1); identical optimiser and steps. Target: a round-1 teacher on 4x4.
 Pass: L_rel reaches lower held-out D-norm eps2 than log, and its largest single-row share of the gradient norm stays < 10% at every step. Broken version = spec-P form: must show a row share
 > 50% at some step (the 1/T^2 blow-up) or a worse eps2. If L_rel and log end equal within 5% -> the 1/B hypothesis has no lever on this class; say so before any GPU run.
Data check (existing tables, CPU): the precedent log-loss copy table vs its target: share of the D-norm residual in rows with |T/t - 1| > 1 (where the losses differ). < 10% -> the loss cannot matter
 at 6x6; skip the run. >= 30% -> run.
Next rung (GPU, only after both checks): ppbf full refit of A_1 with L_rel, warm start A_0, 2 seeds (spec P otherwise unchanged), <= 4 A40-h. Pass retention_N >= 0.5 (then design the joint
 base + head round 2, pass q_2 >= 0.85); kill both < 0.25 -> the ppbf class cannot absorb even one round: flat-cost multi-round storage needs a different base class (PI question).
 Between -> base + head jointly on A_1 decides whether the head covers the rest. Prediction: 0.3-0.5 [own, 55%]; >= 0.5 at 30%.
Status: NOT DONE: known case and data check to run.

APPROXIMATE COPY OVER ROUNDS (PI 2026-11-19: 75-80% per round is enough if it does not degrade badly over rounds). Lead, CPU, existing machinery.
 Store chi_k = B e^{h_k}, h_k = cubic(L2 moments of B, h_{k-1}(x)) refit each round (inputs fixed in k: z^2 B-evals + the previous head at x; flat cost). Start chi_1 = lin_s1 copy.
 Round 2: fit U_2 on L2 + (ln chi_1 - ln B) -> chi_2, q_2. Round 3: T_3 = M[chi_2] (round_map.py), fit U_3 = ln T_3 - ln B on L2 + (ln chi_2 - ln B) -> chi_3, q_3 = (N(chi_2) - N(chi_3))/(N(chi_2) - N(T_3)).
 Pass: q_2, q_3 >= 0.75 and q_3 >= q_2 - 0.05 (no degradation). Kill: q_3 < 0.6 or q_3 < q_2 - 0.15. Prediction: q_2 0.70-0.75 (L2+ with exact r_1* gave 0.747), q_3 0.65-0.75 [own, 50%]. ~4 h CPU.

INSIGHT: ROUNDS ARE PROJECTIVE IMAGINARY-TIME STEPS; COPY LOSS SLOWS, DOES NOT MOVE, THE END POINT (lead 2026-11-19 ~16:00)
 (1) [exact] G_k = |(1 + alpha(H - E)) A_k| (sign s2): the round's first factor is a power/imaginary-time step applied to the stored network; e^{c f_J[G]} is a local correction of G.
  Hence reach grows by one hop per round for the same reason Lanczos-step states cost z^p. The field's standard fix: rebuild the target from the FROZEN previous network
  each step and refit the WHOLE network to it, so the stored cost never grows: Ledinauskas & Anisimovas arXiv:2307.15521 (2D J1-J2; first-order fit of the NQS to the target
  from the Schroedinger step, target held until the energy drops; abstract only read), Kochkov & Clark SWO arXiv:1811.12423 (abstract only). Our add-on-on-frozen-B is the outlier.
 (2) [derived from the contraction model, 55%; model predicted N(T_2) to 0.4%] N(T_{k+1}) - N_inf = kappa (N(chi_k) - N_inf) and N(chi_{k+1}) = N(chi_k) - q (N(chi_k) - N(T_{k+1}))
  => N(chi_{k+1}) - N_inf = (1 - q(1 - kappa)) (N(chi_k) - N_inf). With constant q the copied rounds converge to the SAME N_inf, factor 0.49/round at q 0.75 vs 0.32 exact.
  A floor appears only if the copy has an absolute error eps that does not shrink with the round's gain: N_floor - N_inf ~ eps/(1 - kappa). So the deciding number is the class's
  absolute error on the converged state, not q per round. Consequence for the design: refit the whole network each round (literature route) and measure eps near convergence.
 REFRESH-THE-BASE CHECKS, RESULT (worker 2a3ba2d on j1j2-ratnet; /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/refit_loss/{partA,partB}; lead read 2026-11-19).
  Data check (precedent ppbf copy c1_cplx_s1 vs pfaffian_d/o2b_round_1 target, E_FN 2.092e-5, spliced 1.220e-5): D-norm share of the residual in rows where the copy is TOO SMALL (t/T - 1 > 1,
  where the log loss's 1/B acts) is 1.8% (FN weights, mr_residual.py) / 10.9% (target weights); rows where the copy is TOO LARGE (T > 2t) hold 30.4% / 8.4%.
  INSIGHT: October's full-refit loss sat where the target is small and the copy failed to follow it down (near the target's nodes), not where the copy -> 0. The 1/B divergence is not what
  limited that copy; a loss change alone (L_rel) is not predicted to fix it. The deciding property of a refit class is whether it can place the target's near-zeros.
  4x4 known case: inconclusive (plain PP could not fit the 4x4 base, eps2 ~ 1 for all three losses); spec-P broken line met (row share > 0.5). L_rel's top row = the smallest-t row in every run:
  the fixed 1/t is large exactly at near-nodes under decade weights. Paper check 'REFRESH THE BASE EACH ROUND': NOT DONE (known case not decisive); no GPU refit with L_rel on this evidence.
