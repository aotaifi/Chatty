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
 PAPER READ (lead 2026-11-19, arXiv:2307.15521 sections 1-2.6 read in full): target psi_T = (1 - dtau H) psi_fixed from a FROZEN copy (their Eq. 6-7); two steps per target rejected because
  the cost becomes quadratic (= our nesting z^k). Loss: overlap L = -log |<psi|psi_T>|^2/(<psi|psi><psi_T|psi_T>) (Eq. 8), gradient Eq. 9 with ratios psi_T/psi averaged under |psi|^2: no per-row 1/psi
  blow-up (rows enter weighted by |psi|^2). dtau from the energy-optimal Euler step (Eq. 11) = our alpha line search. KEY: the target is held only until the trained network's energy falls below
  <E> - sigma_E of the frozen one (Alg. 1), then rebuilt: each step needs a PARTIAL copy that lowers the energy, never a full copy; whole network refit, cost flat; competitive with DMRG on 2D J1-J2.
  Mapping to our task: our round = their epoch with the FN grade N in place of <E>; our q per round need not be high, only N(chi_{k+1}) < N(chi_k) beyond noise (consistent with the contraction
  algebra above). Difference: their psi is complex and learns sign changes; we fix s2 and take |u|, which makes kinks at u = 0.
DECISION (lead 2026-11-19, after reading 2307.15521): adopt the frozen-target loop. Round k: target T_k = M[chi_{k-1}] from the frozen previous copy (round_map.py); the ratnet head refit FROM
 SCRATCH on T_k (B frozen, head e^g -> no 1/B in any gradient; B-ratio inputs, lse_lin pool as lin_s1); accept a partial copy: round passes if N(chi_k) < N(chi_{k-1}) by > 5% of the step
 N(chi_{k-1}) - N(T_k). 6x6, rounds 2 and 3, ONE seed (PI: a test; round-1 seed scatter 0.003, far below the 5% line), est. ~4 A40-h. Kill: round 3 gains < 5% of its step (loop stalls). Launch gate (data check): the CPU cubic version of
 the same loop (round3_approx, running) shows N falling at round 3. Doubt [own, 40%]: fixed sign + |u| kinks at u = 0; watched via N per round.
 U-ZEROS CHECK, RESULT (worker, /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/u_zeros/{u_zeros.json,run.log}; lead read 2026-11-19). Precedent copy c1_cplx_s1 vs o2b round-1 target (= FN(g_1),
  g_1 = a_1|1 + alpha_V(E_L - E_b)|, alpha_V -0.1179). Near-zero |u| is a SUFFICIENT cause of copy-too-large (|u| < 0.05: > 99.8% of rows T > 2t; 0.05-0.2: 97%) but not the main one:
  u < 0 holds 12.5% (FN wts) / 10.7% (target wts) of the D residual, 0 <= u < 0.2 7.8% / 0.7%, u >= 0.2 79.7% / 88.5%. Round 1 on B: weight of u < 0 rows p_1 1.3e-6 at alpha* -0.1,
  3.2e-8 at alpha*/3, 2.7e-9 at alpha*/5 (rows 9.2% -> 0.4% -> 0.04%). Reading: a smaller step removes the kinks (cheap knob for the loop if a round stalls at u ~ 0), but ~80% of the
  old full-refit residual is elsewhere, so step size alone would not have rescued that copy.
 LOOP, CPU CUBIC HEAD, PARTIAL (lead read logs 18:10, /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/round3_approx/{r2grade,map3,r3fit}.log): round 2 with the STORED previous head (not exact r_1*):
  N(chi_2) 3.060452e-6, q_2 0.737 (alpha -0.0707). Round-3 teacher from that copy N(T_3) 2.127967e-6. Contraction model (kappa 0.324, N_inf 1.680e-6) predicted 1.680 + 0.324(3.060 - 1.680)
  = 2.127e-6: hit to 0.05% (second confirmation after T_2 at 0.4%). Round-3 fit CV eps2 0.088 (round 2: 0.127). Grade of chi_3 pending.
 LOOP, CPU CUBIC HEAD, RESULT (worker c3da637/ab2be2e, /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/round3_approx/summary.json; lead checked against logs): N(chi_2) 3.0605e-6 (q_2 0.737),
  N(T_3) 2.1280e-6, N(chi_3) 2.6672e-6 (q_3 0.422). Rule: KILL by the per-round line. DIAGNOSIS (equation): absolute copy error eps_k = N(chi_k) - N(T_k): round 2 0.494e-6, round 3 0.539e-6;
  ratnet round 1 (lin_s1) 4.444 - 3.950 = 0.494e-6. eps is ~constant per round while the step shrinks by kappa 0.32, so q must fall; it is not degradation. With constant eps the loop's
  fixed point is N* = N_inf + eps/(1 - kappa) = 1.680 + 0.52/0.676 = 2.45e-6 (exact rounds: A_2 2.416e-6, N_inf 1.680e-6). The per-round q line was the wrong test; the deciding number
  is eps. To go below ~2.4e-6 eps itself must shrink. PREDICTION for ratnet loop r2_s1 (job 18070490, before its grade): N = N(T_2) + 0.5e-6 = 3.07e-6 +- 0.1e-6 (q_2 ~0.73).
 LOOP, RATNET r2_s1, RESULT (job 18070490, /project/theorie/a/A.Otaifi/runs/j1j2/consol_6x6/loop/r2_s1/out/grade.log, lead read 2026-11-20 08:15): N(chi_2) 3.3865e-6 (alpha -0.0847),
  q_2 0.563, eps_2 0.82e-6. PREDICTION 3.07 +- 0.1 MISSED by 3 sd. Why (lead, from existing fits): the ratnet was refit from scratch (fresh net_seed) on one-hop inputs only and lands where the
  cubic without the previous head lands (L2 q_2 0.591, N 3.33e-6); the cubic loop's L2h had h_1 = ln chi_1 - ln B as a 9th input (0.737). Carrying the previous head forward is worth 0.15 of
  round 2: a refit must keep it (warm start or input), as 2307.15521 keeps one network training. My 3.07 assumed the ratnet >= L2h; it had less input. r3 teacher 18070867 (on ratnet r2) not needed.
 CUMULATIVE (lead, derived): share of the exact rounds' energy gain kept, (N(A_0) - N(chi_k))/(N(A_0) - N(A_k)): k=1 0.896, k=2 0.897 (cubic chain), k=3 0.889 (A_3 = 1.918e-6 from the
  contraction model, not measured). Constant eps gives the limit (7.00 - 0.77)/7.00 = 0.89 for all k: the copy does not decay in this measure; the energy stalls at ~2.45e-6.

FIXED-REFERENCE TWO-HOP INPUT, PAPER CHECK (lead 2026-11-20 08:30)
Derivation [exact]: chi = B e^{h(x)}; E_L[chi](x) = D_x + sum_b H_b s_x s_yb rho_b(x) exp[h(y_b) - h(x)], rho_b = B(y_b)/B(x). Round map: U_{k+1} = ln T_{k+1} - ln B = h_k + ln|1 + a(E_L[chi_k] - E)|
 + 1.307 f_J[G_{k+1}]. A head on x's one-hop inputs sees h_k(x) but not h_k(y_b) (two hops of B, since h_k is a function of y_b's one-hop ratios). The missed term each round is the
 neighbour spread of h_k beyond what x's inputs predict; as h_k converges it converges -> eps constant [conjectured link, consistent with eps 0.49/0.49/0.54]. Deciding term: h_k(y_b) in E_L.
 Fix without nesting: give every round's head the SAME frozen two-hop features of the reference chi_1 = B e^{h_1} (lin_s1): l_1(x) = ln|1 + a_1(E_L[chi_1](x) - E_1)| (a_1 = -0.1, round2/teacher
 alpha_start), plus h_1(x). Inputs are fixed functions of B: cost per amplitude z^2 ratio evaluations (one-time factor z over now), flat in k; no chain h_k(h_{k-1}) as in L2h.
 Then round 3's miss is only the neighbour spread of the increment h_2 - h_1 (not of h_2): should be smaller [hypothesis].
Checked: round 2 with inputs (L2, h_1, l_1, f_1 = f_J[G_2]) contains U_2 = h_1 + l_1 + 1.307 f_1 exactly (linear terms) -> must grade at N(T_2) = 2.5667e-6 (known case below). Limit a -> 0: l_1 -> 0,
 class reduces to L2 + h_1 (no chain), measured as arm A. Confidence in the derivation 85%; in the size of the gain 50%.
Prior results: CAPACITY CALIBRATION (cubic on fixed one-hop inputs: 0.95 of round 1, 0.59 of round 2; L2+ with exact r_1* 0.747; L2h 0.737); spin-input ViT add-on (arm H) failed to rebuild B's
 ratios from spins (STORAGE 'ARM H STEPS DO NOT TRANSFER'); 2307.15521 sec 2 (one network kept training). Nothing measured with neighbour-gathered features.
Known case: round 2, arm K = L2 + h_1 + l_1 + f_1 must grade N within 1% of 2.5667e-6. Broken: arm S = arm B with l_1 rows shuffled must grade no better than L2h (3.06e-6).
Next rung (CPU, worker, /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop/): arm B = L2 + h_1 + l_1 on T_2 -> chi_2^B; T_3^B = M[chi_2^B]; arm B on T_3^B -> chi_3^B. Also arm A = L2 + h_1 (no chain), rounds 2-3.
 eps_3 = N(chi_3^B) - N(T_3^B). PASS eps_3 <= 0.30e-6 (fixed point <= 2.12e-6) -> GPU ratnet with these inputs, warm-started. KILL eps_3 >= 0.45e-6 -> neighbour spread is not the lever.
 Between: report. Cost ~6-8 CPU-h on workstations. Prediction: eps_2(B) 0.25e-6 (0.1-0.4), eps_3(B) 0.30e-6, 50%.
Status: DONE (known case is part of the run, read before arm B's numbers).
 FIXED-REFERENCE TWO-HOP INPUT, RESULT (worker 24517de on j1j2-ratnet; /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop/<arm>/{r2fit,map3,r3fit}; lead re-read every grade.json 2026-11-20):
  known case K (L2+h_1+l_1+f_1) N 2.566723e-6 = N(T_2); feature identity 3.6e-15. Broken S (l_1 shuffled) 3.0669e-6 = L2h level: l_1 carries the gain.
  arm A (L2+h_1, no chain): N(chi_2) 3.0605, N(T_3) 2.1280, N(chi_3) 2.7113 -> eps 0.494 / 0.583. arm B (L2+h_1+l_1): N(chi_2) 2.7760, N(T_3^B) 2.0394, N(chi_3) 2.4322 -> eps 0.209 / 0.393 (e-6).
  VERDICT BETWEEN (eps_3 0.393; pass <= 0.30, kill >= 0.45). Prediction eps_2 0.25 / eps_3 0.30: eps_2 right, eps_3 high. Cumulative kept share (A_3 derived 1.918): B 0.943 / 0.924.
  Reading [equation]: in round 2, l_1 IS the round's own factor ln|u_2| (reference = current head), so eps_2 0.209 is the f_J part alone. In round 3 the needed factor uses E_L[chi_2^B]; the
  reference h_1 is one round stale. eps_3 - eps_2 = 0.18e-6 = cost of the stale reference after one round; it grows as h_k drifts from h_1. Lever confirmed (2.4x / 1.4x); staleness is the new term.

REFRESHED ONE-HOP REFERENCE, PAPER CHECK (lead 2026-11-20)
Derivation: round k needs ln|1 + a(E_L[chi_{k-1}] - E)|, i.e. h_{k-1} at neighbours. h_{k-1} itself uses l (two-hop), so its exact neighbour values are three-hop (nesting). Replace it by its
 one-hop projection h'_{k-1} = cubic fit of (ln chi_{k-1} - ln B) on (L2, h_1) [fixed one-hop inputs]; then l'_{k-1} = ln|1 + a'(E_L[B e^{h'_{k-1}}] - E')| needs only two-hop ratios: cost z^2 per
 amplitude, flat in k. Round-k inputs: L2 + h_1 + l_1 + l'_{k-1} (fixed count; l' replaced each round, never accumulated). Miss left: the neighbour spread of h_{k-1} - h'_{k-1} (its two-hop part),
 instead of h_{k-1} - h_1. Checked: round 2 limit: h'_1 = h_1 exactly -> l'_1 = l_1, reduces to arm B. Confidence 80% in the algebra, 50% that the gain is >= half the staleness.
Known case: projection of h_1 itself must return h_1 (cv eps2 ~0) and l' = l_1 to 1e-9. Broken: l' column shuffled -> no better than arm B.
Next rung (CPU, same worker): arm R round 3 on the SAME target T_3^B (chi_2 = chi_2^B), so eps_3(R) compares with B's 0.393 directly; then round 4 for B and R.
 PASS eps_3(R) <= 0.30e-6 -> round 4 decides the trend. KILL eps_3(R) >= 0.37e-6 -> staleness is not one-hop-recoverable; stay with B. Prediction eps_3(R) 0.30e-6, 50%. Cost ~5 CPU-h.
Status: DONE (known case part of the run).
 REFRESHED REFERENCE, KNOWN CASE (worker, ref2hop/known/check.log): projection of h_1 on (L2, h_1) exact (cv eps2 -4.5e-16) up to the D-norm gauge constant -0.0652; the clip to the target range
  moved 805 rows (weight 2e-9); l_part vs ref max 1.27 on rows of weight 3e-11, weighted rms 1.8e-6, alpha_start -0.1. Lead: my 1e-9 bar ignored the clip (a clipped row changes E_L at its
  neighbours); accepted at weighted rms < 1e-5 provided the rows off by > 1e-6 lie within one hop of the 805 moved rows (worker checks before arm R). Round-k projection uses round-k weights.
  Localisation check: 55899 of 57329 off rows within one hop of the moved rows; 1430 farther, weight 8.0e-12, max 1.0e-3. Lead: consistent with the global mean E shifting with the 805
  moved rows (d = E_L - E everywhere; ln|1 + a d| is sensitive only where u ~ 0, i.e. near-node rows of negligible weight). Accepted; arm R relaunched. No further known-case refinement.
 REFRESHED ONE-HOP REFERENCE, RESULT (worker, 24517de; /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop/{R,Rs,B4}; lead re-read grade.json and round_map.json 2026-11-20, inputs prev = lin_s1 fixed):
  arm R (L2 + h_1 + l_1 + l'_{k-1}, l' rebuilt each round from the one-hop projection of the latest copy): N(chi_3) 2.2090 (T_3^B 2.0394, eps 0.170), N(chi_4) 2.2090 -> 1.9482 (T_4^R 1.6955, eps 0.253).
  arm B round 4: N(chi_4) 2.2379 (T_4^B 1.8111, eps 0.427). Broken Rs (l'_2 shuffled) 2.4318 = B (0.392): l' carries the gain. PASS (eps_3 0.170 <= 0.30; prediction 0.30 at 50%: better).
  Copy error per round, R: 0.209 / 0.170 / 0.253 (B: 0.209 / 0.393 / 0.427). Per-round q_R: 0.77 (r3), 0.51 (r4). Copy at round 4 (1.948e-6) is below EXACT round 2 (2.416e-6).
  Contraction model broke at round 4: predicted N(T_4^R) = 1.680 + 0.324(2.209 - 1.680) = 1.851, measured 1.6955 (also B: 1.924 vs 1.811); rounds <= 3 matched to 0.5%. So N_inf 1.680 and
  derived A_3 1.918 / A_4 1.757 are not trusted: exact A_3, A_4, A_5 by round_map on rounds23/lA_2.npy are running (CPU) before any comparison with exact rounds is quoted.
 EXACT ROUNDS 3-5 AND ARM R ROUND 5, RESULT (worker 24517de; ref2hop/exact/map3-5, ref2hop/R5; lead re-read round_map.json, grade.log, fit.json 2026-11-21):
  exact: N(A_2) check 2.4160953 (= rangelimit 2.416095); N(A_3) 1.793102, N(A_4) 1.423426, N(A_5) 1.200171 (e-6). arm R round 5: N(T_5^R) 1.558138, N(chi_5^R) 1.931361, eps_5 0.373.
  Copy vs exact, rounds 1-5: cumulative share (N_A0 - N_chi)/(N_A0 - N_A) 0.896 / 0.943 / 0.940 / 0.928 / 0.902; ratio N_chi/N_A 1.13 / 1.15 / 1.23 / 1.37 / 1.61.
  Per-round q: 0.90 / 0.89 / 0.77 / 0.51 / 0.04. The copy STALLS at round 5 (1.948 -> 1.931) while its target gained 0.39. eps by round 0.209 / 0.170 / 0.253 / 0.373: degrades.
 DIAGNOSIS (lead 2026-11-21, R5/r5fit/tables/t_R.npy vs R5/map5/lA_next.npy):
  [measured] the cubic head u extrapolates on rows the D-norm weight (n p^2) ignores: raw fit range [-3995, 491] vs target U range [-15.4, 7.6]; 5638 rows clipped at the top carry 3.5e-5 of
  chi_5's norm vs 4.2e-10 under T_5; top rows by KL(chi||T): u - U = +9 to +11 (amplitude x e^10) at p_T ~ 1e-14. |u-U| quantiles 0.5/0.99/0.999: 0.015 / 2.7 / 5.8.
  Clipped rows by round 1201 / 2824 / 6237 (R3-R5); E_var/site 3.9e-6 (chi_3) -> 3.2e-5 (chi_4) -> 4.2e-5 (chi_5), while the D-norm residual eps2*D_UU stays 4.7e-5 / 4.9e-5 / 8.7e-5:
  the error that grows is where the loss does not look.
  [equation, conjectured as the compounding term] a spike chi(x0) -> e^D chi(x0) enters every neighbour y as E_L[chi](y) += H_{y x0} (e^D - 1) chi(x0)/chi(y); the round map multiplies T_{k+1}(y)
  by |1 + a(E_L(y) - E)| and FN adds H_{y x0} chi(x0)/chi(y) to H_FN(y,y) on sign-violating bonds. So spikes in round k inflate round k+1's target and the grade near them: error compounds.
 SPIKE TEST, PASS/KILL (lead 2026-11-21, written before the run): regrade fixed tables for chi_4^R (T_4^R, eps 0.253) and chi_5^R (T_5^R, eps 0.373), targets unchanged.
  Tags: FB0 = u set to 0 (fall back to B) where u hit the top clip; FBlo = u set to lo there (both production-legal: the flag is raw head output > trained max);
  ORh = u set to U on top-clipped rows, ORs = u set to U where u - U > 3 (oracles, test only: how much of eps the spikes carry).
  KILL spike hypothesis: ORs eps_5 >= 0.33 (spikes carry < 0.04 of 0.373). PASS hypothesis: ORs eps_5 <= 0.25. PASS fix: FB0 or FBlo eps_5 <= 0.26 and eps_4 <= 0.20 -> rerun rounds 4-6
  with the fallback (and a spike penalty in the loss) to see whether eps stays flat. Between: report. Prediction: ORs eps_5 0.22 (60%); FBlo beats FB0; fix passes 45%. Cost ~2 CPU-h.
 SPIKE TEST, RESULT (worker; ref2hop/spike/r{4,5}/grade.json, lead re-read 2026-11-21): eps (e-6) vs fixed T_k^R, unmodified r4 0.253 / r5 0.373:
  FB0 0.202 / 0.294; FBlo 0.202 / 0.294; ORh 0.202 / 0.294 (all three equal to 1e-10: any value on the top-clipped rows below the spike gives the same N); ORs 0.179 / 0.250.
  E_var/site: unmodified 3.2e-5 / 4.2e-5, FB0 8.5e-6 / 9.3e-6, ORs 2.8e-6 / 2.8e-6. VERDICT: hypothesis PASS at the line (ORs eps_5 0.2497 <= 0.25): spikes carry 0.07 / 0.12 of eps
  (a third at r5); fix BETWEEN (FB0 0.294 > 0.26, eps_4 0.202 > 0.20): the range flag catches the top-clipped spikes only; in-range spikes (u - U > 3, 118k rows at r5) carry 0.045 more.

COPY-MEASURE WEIGHTS FOR THE COPY FIT, PAPER CHECK (lead 2026-11-21)
Derivation: (1) [exact] for H stoquastic with ground state T > 0 and psi = T f: <psi|H - E0|psi> = 1/2 sum_{x!=y} |H_xy| T_x T_y (f_x - f_y)^2 (ground-state transform; Dirichlet form of the
 generator). (2) [exact] copy chi = T e^d, f = e^d: (f_x - f_y)^2 = e^{d_x + d_y} 4 sinh^2((d_x - d_y)/2). So the cost of a copy error is sum |H| chi_x chi_y 4 sinh^2(Dd/2): weighted by the
 COPY's measure. (3) [approx, 2nd order in d] the D-norm in capacity_fit uses n HW p_x p_y (d_x - d_y)^2 with p from the TARGET: it drops e^{d_x+d_y} and the sinh. For a spike d = 10 next to d ~ 0
 the true cost is e^{20} T T, the D-norm charges 100 T T: under-weighted by ~5e6. This is the deciding term; it explains why spikes sit exactly where p_T ~ 1e-14 and grow unchecked.
 (4) Remedy [approx]: weights p_mix = sqrt((p_T^2 + p_chi^2)/2) (sum n p_mix^2 = 1), p_chi = copy amplitude from a first pass; restores e^{d_x+d_y} on spike bonds (sinh factor still missing,
 under by ~220x at d = 10, shrinking as d falls); one more fit pass, flat inference cost.
Checked: d = 0 limit: p_mix = p_T, D-norm unchanged. Constant shift d: both forms give zero (gauge). Small d: e^{d_x+d_y} 4 sinh^2 -> (Dd)^2, recovers the D-norm. Sign of the error: positive spikes
 (copy too large) are the under-weighted ones, negative ones are over-weighted relative to their cost: matches the data (top-clipped, not bottom-clipped, rows carry the excess mass 3.5e-5 vs 1.5e-25).
 Applicability: our grade is FN energy of a one-step map, T_k is not the ground state of H_FN: (1) holds only as the near-target expansion. Confidence 75% that reweighting removes most spike cost.
Prior results: recall.sh j1j2 "copy measure|reweight|own distribution|spike|Dirichlet": PLAN 691-693 (B refit per-row trust region |dlnB| <= 0.5, a different fit: pfaffian base, not the copy head);
 delta 11-05 T-EDGE "reweighting worsens the tax" (QMC sampling reweighting for the sign, not fit weights). Neither binds here. Literature: supervised wave-function fits sample from both psi and
 target (Kochkov & Clark arXiv:1811.12423, from memory, not re-read); fitted value iteration with extrapolating regressors diverges (Gordon 1995, from memory).
Known case + broken: W5f (refit on fixed T_5^R with p_mix) must cut top-clip mass and eps_5 to <= ORs 0.25; broken W5s (p_chi permuted, rng 777) must NOT (eps_5 >= 0.33).
Next rung (CPU, arm W = arm R inputs + p_mix two-pass fit + FB0 guard): W4 on T_4^R, then the chain W5, W6 from chi_4^W (map, proj, mapP, fit x2, guard, grade), plus exact map6 for N(A_6).
 PASS eps_5^W <= 0.22 and eps_6^W <= 0.22 -> spikes were the degradation; next the network copy with this loss. KILL eps_5^W >= 0.30 -> the rise is not spikes (capacity/input reach).
 Between: report. Prediction eps_4^W 0.17, eps_5^W 0.20, eps_6^W 0.21 (50%). Cost ~20 CPU-h over ~15 h wall.
Status: DONE (known case runs first, read before the chain's numbers).
 AMENDMENT (lead 2026-11-21, after the worker's stop): p_out of round_map is the FN ground state phi at the best alpha, not exp(lA_next) (rel. diff median 2e-3, max 1.5; W/wtools.py check).
  In (1) T is phi, so the copy measure is phi e^{d} with d = u - U: p_c = p_T exp(t - lA_next), normalised (sum n p_c^2 = 1). p_mix = sqrt((p_T^2 + p_c^2)/2). Identity check: t = lA_next gives p_c = p_T.
 ARM W KNOWN CASE, RESULT (worker; ref2hop/W/{w4fit,w5f,w5s} grade logs + fit.log, lead read 2026-11-21 09:30): eps vs fixed T_k^R (e-6), raw / guarded:
  W4 (p_mix, r4): 0.214 / 0.209, E_var 4.3e-6, clipped 3906, raw range [-1191, 2628]. W5f (p_mix, r5): 16.8 / 4.02 (!), E_var 8.6e-3, clipped 113453, D-norm eps2 0.156 (R5: 0.022).
  W5s (permuted ratio, broken): 0.376 / 0.295 = no change from R5. KNOWN CASE FAILED: W5f must reach <= 0.25; it is 16x worse. Copy-measure reweighting, as a second LSQ pass, is dropped.
  Diagnosis [equation]: the head is a global cubic in standardised inputs Z; a spike row has |Z| far outside the bulk (raw output ~1e3 = Z^3 terms). Re-weighting those rows makes them high-leverage
  points: the cubic bends to them, the bulk fit degrades (eps2 0.022 -> 0.156) and extrapolation moves to 113k other rows. The loss weights were never the lever; the extrapolation of a polynomial
  in unbounded inputs is. Remedy class: bounded input growth (asinh transform, existing option TRANSFORM=asinh) or winsorised Z; at r1-r2 asinh gave no retention gain (line ~1053) when spikes were few.
 ASINH HEAD TEST, PASS/KILL (lead 2026-11-21, before the run): arm R inputs, p = p_T, TRANSFORM asinh, FB0 guard, fixed targets T_4^R, T_5^R (tags A4, A5).
  PASS eps_4 <= 0.19 and eps_5 <= 0.25 (oracle-spike-free level without the oracle) -> asinh chain rounds 5-6. KILL eps_5 >= 0.33. Between: report. Prediction eps_5 0.26 (50%), clipped rows < 1000.
 ARM W CHAIN + ASINH, RESULT (worker; ref2hop/W, collect.py; lead re-read grade logs and round_map.json 2026-11-21): eps (e-6), raw / guarded:
  A4 asinh 0.313 / 0.194, A5 asinh 0.588 / 0.318 (clipped 41k / 45k): BETWEEN (pass eps_4 <= 0.19, eps_5 <= 0.25; kill >= 0.33). Not the fix.
  Chain from the cleaner chi_4^W (N 1.9047, E_var 4.3e-6): T_5^W 1.5164 (better than T_5^R 1.5581), pass-1 copy N 1.9503, eps 0.434 (R5 from the spiky chi_4: 0.373). Pass-2 reweighted fits broke
  (2.757 guarded). Exact N(A_6) 1.117349 (exact/map6; round-6 gain 0.083: the exact rounds converge near 1.1).
  READING: the copy lands at N 1.93-1.95 whatever its target (T_5 1.558 or 1.516), and the oracle spike-free tables at 1.81-1.87: a FLOOR of the cubic family on these inputs, not compounding
  (a cleaner previous copy did not lower the next copy error). Copy keeps (8.683 - 1.93)/(8.683 - 1.117) = 0.89 of the exact rounds' total gain at their convergence; its N is 1.7x theirs.
 STEP BACK (lead 2026-11-21; second failed fix of the cubic copy: range guard partial, reweighting broke, asinh between):
  1. Tool: a bounded learner on the SAME inputs (tanh MLP or trees) separates capacity from information in ~1 CPU-h; more cubic engineering does not.
  2. Same form: approximate power iteration with a supervised step (Giuliani, Vicentini, Rossi, Carleo arXiv:2303.08902, abstract read: works if the learning step is efficient; uses kernel
     ridge regression, a bounded smoother) and approximate value iteration (floor ~ step error/(1 - contraction); averagers are stable, extrapolating regressors are not: Gordon 1995, from memory).
  3. Switch: from the cubic proxy to a bounded network on the same inputs, which is also the PI's task (copy into a network).
BOUNDED NETWORK ON THE COPY INPUTS, PAPER CHECK (lead 2026-11-21)
Derivation: [exact] f(z) = w2 . tanh(W1 z + b1) + b2 obeys |f| <= |b2| + sum|w2| for all z: no growth outside the trained range (a cubic grows as |z|^3, the spikes' source). [exact] universal
 approximation on compacts (Cybenko 1989): capacity grows with width, so at width 64-128 the network can only lose to the cubic through optimisation, not expressivity on the bulk.
 [approx] loss: pointwise weighted MSE of u - U with weights n p_T^2 (the D-norm is its bond-difference form; constants are a gauge in both). Deciding quantity: N of the copy on a fixed target.
Checked: width 0 limit -> constant (worse than cubic, as it must be); tanh linear regime reproduces a linear fit. Confidence 85% the network shows no spikes; 50% it beats the cubic floor.
Prior results: recall j1j2 "MLP|tanh|bounded|trees": the ratnet (a network) r2 from scratch 3.3865 lacked the previous head and l inputs (LOOP RATNET r2 block): different inputs, does not bind.
Known case + broken: NN on fixed T_4^R must reach eps_4 <= the cubic's 0.253 (R4); broken: l'_3 column permuted (rng 777) must lose >= 0.1 (l' carried 0.22 at round 3, Rs block).
Next rung: NN5 on fixed T_5^R. PASS eps_5 <= 0.25 -> capacity was the floor: NN chain rounds 5-6 (and the GPU network copy). KILL eps_5 >= 0.35 -> information floor: the inputs must grow
 (joint two-exchange tokens, l' at several alpha). Between: report. Prediction eps_5 0.27 (50%), E_var < 1e-5. Cost ~3 CPU-h on workstations.
Status: DONE.

BOUNDED NETWORK ON THE COPY INPUTS, RESULT (lead 2026-11-21; runs /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop/NN/{NN4r,NN4sr,NN5r}/fit/grade.json, nn.json; random 10% holdout rng 2, seed 0)
  fit   target N   copy N     eps     E_var_site  train/val loss     rows outside U range
  NN4   1.695513   1.910675   0.215   3.80e-6     3.63e-6/3.52e-6    0
  NN4s  1.695513   2.154809   0.459   4.94e-6     7.28e-6/7.47e-6    0     (l' permuted rng 777)
  NN5   1.558138   1.731453   0.173   3.93e-6     4.71e-6/4.91e-6    0     (cubic on the same target: 1.931361, eps 0.373)
  (e-6; deterministic full-space grades, no chains; guard flagged 0 rows, network output in [-1.2, 1.4] inside U's [-15.4, 7.6])
  Known case PASS (0.215 <= 0.253); broken PASS (l' permuted loses 0.244 >= 0.1); NN5 PASS (0.173 <= 0.25; predicted 0.27 at 50%: better).
  Reading: the cubic floor N 1.93 was the cubic's capacity (spikes + bulk), not missing information in (L2, h_1, l_1, l'). Cumulative share at NN5: (8.683-1.731)/(8.683-1.117) = 0.919 (cubic 0.89).
  Caveat: NN5's input chi_4 is the cubic arm-R copy; a network-only chain is not yet measured.

NETWORK-ONLY CHAIN ROUNDS 5-6, PASS/KILL (lead 2026-11-21)
Model [approx, fitted to arm R]: N(M[chi]) - N* = c (N(chi) - N*), c = 0.56, N* = 1.05 (R: 1.948->1.558 and 2.209->1.696 give c 0.566, 0.557). With a constant copy error eps per round
 the chain converges to x* = N* + eps/(1-c): eps 0.17 -> x* 1.44 (share 0.958 of the exact gain). Compounding would show as eps_k growing with k.
Chain: chi_4^N = NN4 (1.9107) -> map5 -> proj5 -> mapP4 -> NN fit -> chi_5^N -> map6 -> proj6 -> mapP5 -> NN fit -> chi_6^N (same pipeline as W/chain_w4.sh, NN fit in place of armfit).
PASS: eps_5^N, eps_6^N <= 0.25 and N(chi_6^N) < N(chi_5^N) -> no degradation: GPU network copy next. KILL: eps_6^N > eps_5^N + 0.1, or N(chi_6^N) >= N(chi_5^N) -> copy errors compound:
 diagnose which inputs went stale (l' from a projection h' of a network copy) before anything else. Between: report.
Prediction: T_5^N 1.53, chi_5^N 1.70, T_6^N 1.41, chi_6^N 1.58 (+-0.06 at 50%). Cost ~3.8 h per round on one workstation (map 26 min, proj 68, mapP 26, fit 96, grade 12).

NETWORK-ONLY CHAIN ROUNDS 5-6, RESULT (lead 2026-11-22; /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop/NNc, map<k>/round_map.json, n<k>fit/grade.json R; nmatch rel 0 both rounds)
  k  N_start   target N_out  copy N(chi_k)  eps     train/val loss    proj eps2 (cubic h', clipped rows)
  5  1.910675  1.456268      1.688596       0.232   3.52e-6/3.46e-6   0.0618 (118465)
  6  1.688596  1.293669      1.589341       0.296   4.25e-6/4.16e-6   0.0735 (202289)
  (e-6; output inside U's range, 0 guard rows.) Verdict: BETWEEN (eps_6 0.296 > 0.25; not kill: N falls 0.099, eps grew 0.063 < 0.1).
  Prediction chi_5 1.70, chi_6 1.58 met (1.689, 1.589); targets fell faster than the c-model (1.456 vs 1.53). Cumulative share round 6: (8.683-1.589)/(8.683-1.117) = 0.938.
  Per-round copy fraction (N_start-N(chi))/(N_start-N_out): 0.49, 0.25. Exact rounds from our copy still gain ~0.4 per round; the copy keeps a falling part of it.

COPY ERROR GROWTH IN THE NETWORK CHAIN, DIAGNOSIS + PASS/KILL (lead 2026-11-22)
Equation: l'_{k-1}(x) = ln|1 + a'(E_L[B e^{h'}](x) - E')| with E_L[B e^{h'}](x) = D_x + sum_b H_xb rho_b(x) e^{h'(y_b)-h'(x)}, while the round map uses h = h_{k-1} (the network head).
 [exact] error in the sum = sum_b H_xb rho_b e^{h'(y_b)-h'(x)} (e^{d(y_b)-d(x)} - 1), d = h_{k-1} - h'_{k-1}: the neighbour spread of the projection miss d.
 Two candidate terms for eps growth: (a) d grows: the cubic projection's eps2 0.062 -> 0.073 and its clipped (spiking) rows 118k -> 202k, the same cubic failure the copy had;
 (b) capacity: the 64-64 net underfits a target with more structure (train = val, loss 3.52 -> 4.25e-6).
Tests on the fixed target T_6^N (NNc/map6), one seed each, CPU workstations in parallel, ~4 h:
 (a) NN projection: h' = tanh net on (L2, h_1) [9 inputs, same weights], then mapP5 from it, then the 64-64 copy. PASS eps_6 <= 0.25 -> use it in the chain; >= 0.28 -> not the term.
 (b) width 128-128 copy, same inputs. PASS eps_6 <= 0.25 -> use it; >= 0.28 -> not the term.
 Both >= 0.28 -> information beyond the 11 inputs: grow them (joint two-exchange tokens), paper check first.
 (c) chain round 7 unchanged (from n6fit): eps_7 <= eps_6 + 0.03 and N(chi_7) < N(chi_6) -> levels off without degrading; eps_7 > eps_6 + 0.1 or N(chi_7) >= N(chi_6) -> degrades.
Prediction: (a) eps_6 0.24 (50%), (b) 0.28 (50%); I bet on (a) at 55%. (c) N(chi_7) 1.54, eps_7 0.32.

COPY ERROR GROWTH, RESULT (lead 2026-11-22; /scratch/a/A.Otaifi/ansatz_runs/consol_6x6/ref2hop/NNd/{a,b,c}; target T_6^N 1.293669, e-6)
  arm                                   proj eps2 (clipped rows)  copy N     eps_6   train/val loss
  NNc (cubic proj, 64-64, MSE)          0.0735 (202289)           1.589341   0.296   4.25/4.16e-6
  (a) NN proj (MSE loss), 64-64 copy    0.1112 (0)                1.667782   0.374   4.15/4.17e-6
  (b) cubic proj, 128-128 copy          0.0735                    1.614318   0.321   4.33/4.29e-6
  (c) round 7: N_start 1.589341, N_out 1.209508, copy 1.530199, eps_7 0.321; proj7 eps2 0.0802 (211795 clipped).
  Kill lines: (a) and (b) both >= 0.28 (not the term as written); (c) PASS: levels off without degrading (eps_7 - eps_6 = 0.025 <= 0.03, N falls 0.059).
  Chain so far: 1.911, 1.689, 1.589, 1.530 (steps 0.222, 0.099, 0.059); eps 0.232, 0.296, 0.321; share round 7 (8.683-1.530)/7.566 = 0.945.
  Diagnosis of why (a), (b) fired (the 'grow the inputs' branch does NOT follow):
   (a) changed two things: no spikes, but a pointwise-MSE loss, and its D-norm eps2 got worse (0.111 vs 0.0735). The 202k cubic spikes did not hurt; the bulk D-norm accuracy did:
       eps_6 rose 0.078 for +0.038 in eps2 (slope ~2). That supports the equation's term (neighbour spread of d) and says the projection must be fit in the D-norm.
   (b) the wider net reached a HIGHER train loss in the same 20k steps: optimisation-limited, not a capacity verdict.

BOND-DIFFERENCE (D-NORM) LOSS FOR THE NETWORK FITS, PAPER CHECK (lead 2026-11-22)
Derivation: [exact] for psi = T f, <psi|H - E_T|psi> = 1/2 sum_xy |H_xy| T_x T_y (f_x - f_y)^2 (ground-state transform on the sign-fixed operator; Dirichlet form).
 [approx, 2nd order in d] with f = e^d, d = u - U: excess energy ~ 1/2 sum |H_xy| T_x T_y (d_x - d_y)^2 = the D-norm of d. A pointwise MSE sum w d_x^2 penalises smooth d (free in energy) and
 sees bond roughness only through the bound (d_x - d_y)^2 <= 2(d_x^2 + d_y^2): it spends capacity on the wrong components. The same holds for the projection h' (its error enters l' through
 the neighbour spread of d, COPY ERROR GROWTH block). Deciding term: D-norm eps2 of each fit.
Checked: constant shift of d -> both energy and D-norm zero (gauge), MSE not; exact eigenstate T -> zero. Cubic fits (capacity_fit) already minimise the D-norm; the NN fits did not. Confidence 75%.
Prior results: TRIED_IDEAS row 1 (edge/pair losses B13 4x4 66% vs 46%, B30 6x6 'does not generalise', B36 weak): sampled from-scratch training with generalisation; here the full table, val = train: does not bind.
Known case + broken: (d) NN projection with D-norm loss on the chi_5 head must reach eps2 <= the cubic's 0.0735; broken = the same net with MSE loss, already measured 0.111 (arm a).
Next rung (target T_6^N, CPU, ~5 h each, parallel): (d) D-norm NN projection -> mapP5 -> 64-64 MSE copy; (e) 64-64 copy with D-norm loss on the NNc inputs (cubic proj).
 PASS eps_6 <= 0.25 in either -> adopt it in the chain (both if both). KILL both >= 0.30 -> the loss is not the lever; then inputs (joint two-exchange tokens). Between: report.
 Prediction: (d) eps2 0.05, eps_6 0.26; (e) eps_6 0.24 (50% each); 60% that (e) beats 0.296.
Status: DONE.

METRIC (PI 2026-11-22): judge the copy by the per-round kept share q_k = (N_start - N(chi_k))/(N_start - N(T_k)), the PI's 75-80% bar applies to it, not to the cumulative share.
  Network chain: q_4 0.58 (cubic chi_3 input), q_5 0.49, q_6 0.25, q_7 0.16. Gains per exact round stay 0.38-0.45e-6; eps grows 0.23 -> 0.32. Below the bar at every round.

WHY THE PER-ROUND SHARE FALLS, EQUATION + ORACLE TEST (lead 2026-11-22)
Equation: target U_k = ln T_k - lB = h_{k-1} + l_k + c f_J (+ const), with h_{k-1} = NN_{k-1}(L2, h_1, l_1, l'_{k-2}) and l_k(x) = ln|1 + a(E_L[chi_{k-1}](x) - E)|.
 Copy inputs at round k: (L2, h_1, l_1, l'_{k-1}). Two missing pieces [exact bookkeeping]:
 (i) l_k - l'_{k-1}: l' is built from the one-hop shadow h'_{k-1} of the copy; its error is the neighbour spread of d = h_{k-1} - h'_{k-1}; shadow eps2 grows 0.062, 0.073, 0.080 (rounds 5-7).
 (ii) the part of h_{k-1} carried by l'_{k-2}, which the round-k inputs no longer contain (l' is replaced, not accumulated).
 (iii) not information: the fit's loss/capacity (the D-norm runs NNe test the loss).
Oracle fits on T_6^N (cost-cheating, diagnostic only; 64-64 MSE copy, same settings):
 O1 exact l_6 (rmap NNc/n5fit --save-parts) in place of l'_5; O2 h_5 = ln chi_5 - lB added as 12th input (l'_5 kept); O3 both.
 O1 eps_6 <= 0.15 -> (i) dominates: richer fixed shadow next (projection on (L2, h_1, l_1), z^3 cost, flat in k), paper check first.
 O2 <= 0.15 -> (ii) dominates: carry a fixed window of past l' (flat in k). O3 >= 0.25 -> information is not the limit: loss/capacity (NNe decides).
 Prediction: O1 0.17, O2 0.22, O3 0.12 (50% each); 55% that (i) is the larger piece.

ORACLE AND D-NORM FITS, RESULT (lead 2026-11-22; ref2hop/NNe/{d,e}, NNo/{O1fit,O2fit,O3fit}; target T_6^N, gain from chi_5^N 0.394927; e-6; one seed, deterministic grades)
  fit                                     copy N     eps_6   q_6     copy D-norm eps2
  NNc (cubic shadow, MSE)                 1.589341   0.296   0.25    0.0773
  (d) D-norm NN shadow (eps2 0.054), MSE  1.549793   0.256   0.35    0.0724
  (e) cubic shadow, D-norm copy           1.564740   0.271   0.31    0.0312
  O1 exact l_6 replaces l'_5              1.704830   0.411  -0.04    0.0727
  O2 + h_5 (l'_5 kept)                    1.604308   0.311   0.21    0.0822
  O3 exact l_6 + h_5                      1.475111   0.181   0.54    0.0713
  Kill lines: (d) 0.256, (e) 0.271: between (not <= 0.25, not >= 0.30). O1, O2 not <= 0.15; O3 0.181 not >= 0.25: between.
  Readings: (1) the shadow lever is real but small: eps falls ~2x the shadow's eps2 drop (0.0735 -> 0.054 gives -0.040, same slope as arm a).
   (2) the copy's bond-difference error fell 2.5x in (e) with almost no gain in N: the bulk bond error does not set N.
   (3) h_5 and l_6 help only together (O3); each alone does not: l'_5 was doing double duty (carrying h_5 and l_6 partly).
   (4) even with both exact, q_6 = 0.54 < 0.75: the limit is not (only) information in the inputs.
  Hypothesis for (2)+(4) [55%]: the target has deep dips (U down to -15.6, from near-zeros of |1 + a(E_L - E)|) that a bounded output (range [-1.2, 1.4]) cannot place;
   rows with U - mean < -1: 1.3M rows, weight 4.8e-7, copy too high by 0.46 on average; < -3: 156k rows, too high by 3.4. A guide too large at a near-node row costs energy
   through e^{d} (the D-norm drops it). Same place as the sign angle's limit (errors near the round's changes).
PATCH TEST, PASS/KILL (existing tables, ~1 h CPU): replace the NNc copy by the exact target (+ the copy's bulk gauge constant) on rows with U - mean < -1 / -2 / -3; the converse (bulk exact, dips kept);
 and O3 patched at < -1. PASS: patch < -1 gives eps_6 <= 0.15 -> dips dominate: next = copy whose output carries the shadow's Lanczos factor unbounded (u = NN + ln|1 + a'(E_L[shadow] - E')|),
 paper check first. KILL: patch < -1 gives eps_6 >= 0.25 -> dips are not it; the error is spread over the bulk. Prediction: patch < -1 eps 0.12 (55%).

PATCH TEST, RESULT (lead 2026-11-22; ref2hop/NNp/V*/grade.json; known cases V0 = exact target 1.293669 (rel 4e-8), V0b = NNc copy 1.589341 (rel 7e-8); bulk gauge c 8.4e-5)
  variant                         rows       weight     N        eps_6   N at alpha=0
  NNc copy                        -          -          1.5893   0.296   2.849
  V1 dips (d<-1) exact            1316328    4.8e-7     1.5800   0.286   2.352
  V2 d<-2 exact / V3 d<-3 exact   446k/156k  1.5e-8/3e-9 1.5848/1.5874  0.291/0.294
  V4 bulk exact, dips = copy      14488628   0.9999995  1.3335   0.040   2.043
  V5 O3 with dips exact           1316328    4.8e-7     1.4501   0.156   1.879
  exact target                    -          -          1.2937   0       1.514
  KILL for the dip hypothesis (patch < -1 gives 0.286 >= 0.25). The error after the alpha step is in the bulk.
  Reading of the alpha=0 column: at alpha = 0 (fixed-sign FN, magnitude only) the copy loses 1.34 (2.849 vs 1.514), the dips 0.50 of it; the D-norm copy (e) wins 0.62 there
   (N_a0 2.232) as the lattice-FN second-order theory says, but after the step it keeps only 0.024 of that. The grade's Lanczos step repairs magnitude errors and dips;
   what it cannot repair it creates: the step's factor 1 + a(E_L[chi] - E) changes sign where E_L[chi] crosses E - 1/a, and E_L[chi](x) depends on chi's bond differences at x.
DIAGNOSIS [derived, sizes not yet measured]: guide after the step = s chi (1 + a(E_L[chi] - E)); a copy error d moves E_L by sum_b H_xb rho_b e^{h_b - h_x}(e^{d_b - d_x} - 1), which flips
 the step's sign wherever |1 + a(E_L - E)| is small. A misplaced sign costs first order, magnitude second order. So the copy must be accurate in its local energy near the next
 step's nodes, a small set of rows and their neighbours. Same limit as the sign angle's STEP 8 (false flips near the round's changes; <= 1-2% neighbour error needed).
NODE PATCH TEST, PASS/KILL (existing tables, ~1 h CPU): l7 = ln|1 + a(E_L[T_6] - E)| from rmap of the exact target (V0 table) --save-parts. Near-node set Nn(tau) = {l7 < ln tau},
 closure C = Nn plus all its J1/J2 neighbours. Patch the NNc copy with the exact target on: W1 C(0.3), W2 C(0.1), W3 Nn(0.3) only (no neighbours), W4 everything except C(0.3).
 PASS: W1 eps_6 <= 0.10 with weight(C) << 1 and W4 >= 0.2 -> the error sits at the next step's nodes: the copy's loss must weight local-energy error by node proximity
 (train on what the stepped grade sees), paper check next. KILL: W1 >= 0.2 -> not node-localised. Prediction: W1 0.08, W3 0.2, W4 0.22 (50%).

NODE PATCH TEST, RESULT (lead 2026-11-22; ref2hop/NNp/W*/grade.json, map7T: alpha_start -0.040789, N_out 1.098598e-6)
  W1 C(0.3) exact (7.9M rows, weight 0.025): eps_6 0.255; W2 C(0.1) (weight 0.0058): 0.283; W3 Nn(0.3) only: 0.293; W4 all but C(0.3) exact (weight 0.975): 0.079.
  KILL for the node hypothesis (W1 >= 0.2). The error is spread over the high-weight bulk (14x denser per unit weight near nodes, but 3/4 of it in the bulk).

STEP BACK (lead 2026-11-22; second failed diagnosis in a row: dips, then nodes)
1. Verdict: switch from diagnosing the inputs/loss/locations to converging the fit itself.
2. Thought: every NN fit is constant-lr Adam (1e-3, 20k steps) ending on a noisy iterate: val loss jumps between checkpoints (NNc 5.5e-6 at 18k, 4.2e-6 at 20k;
   O2 3.7e-6 at 19k, 6.7e-6 at 20k) and still falls at the end (NNc 7.3e-6 at 10k). Across arms eps_6 tracks the final loss: eps ~ 0.07 x loss(e-6)
   (NNc 4.2 -> 0.296, (d) 3.8 -> 0.256, O3 1.9 -> 0.181). The patch tests fit this too: the error sits where the MSE residual sits, in the weight-dominated bulk.
   Differences of 0.02-0.05 between my arms (width, O2, d) are inside this optimiser noise; they were compared as if they were signal. Constant-step SGD/Adam converges only
   to a neighbourhood of size ~ step size (Bottou, Curtis, Nocedal arXiv:1606.04838, Thm 4.6, fixed-stepsize SGD, from memory; conditions: smooth objective, bounded variance).
3. Cheapest test: the NNc round-6 copy (same inputs, MSE) with cosine lr decay 1e-3 -> 1e-6: f1 20k steps (~1.5 h), f2 80k steps (~6 h), CPU, in parallel.
   PASS: f1 or f2 eps_6 <= 0.15 (q_6 >= 0.62) -> the optimiser was the limit: rerun the chain with the converged fit. KILL: final loss falls >= 2x but eps_6 >= 0.25 -> loss is not
   the measure that sets N. Prediction: f1 loss 2.0e-6, eps 0.16; f2 loss 1.2e-6, eps 0.10 (50%).
  f1 RESULT (lead 2026-11-22; ref2hop/NNf/f1/f1fit/{grade.json,nn.json}): cosine 20k, final val 5.21e-6 (train 5.18e-6), N 1.618356e-6, eps_6 0.325, q_6 0.18, N_a0 3.246e-6,
   alpha -0.0674; val at 5k/10k/15k/20k 17.3/8.04/5.79/5.21 e-6. Neither pass nor kill (loss rose vs NNc 4.16e-6, it did not fall). My f1 loss prediction (2.0e-6) was 2.6x
   too low: at 20k steps the fit is step-limited (curve still falling as the lr vanishes), not only jitter-limited. eps/loss = 0.062 (NNc 0.071): one more point on the eps-loss line.
   f2 decides: at eps ~ 0.065 x loss, PASS (eps <= 0.15) needs final val <= 2.3e-6. f2 val 3.72e-6 at 23.5k of 80k (already below NNc's final).
  f2 RESULT (lead 2026-11-23; ref2hop/NNf/f2/f2fit/{grade.json,nn.json}): cosine 80k, final val 1.79e-6 (train 1.75e-6; val 3.60/2.60/1.97/1.79 e-6 at 20/40/60/80k),
   N 1.506093e-6, eps_6 0.212, q_6 0.46 (NNc 0.25), N_a0 2.164e-6, alpha -0.0586, D-norm eps2 0.0375, grad check 4.4e-7, no out-of-range rows.
   BETWEEN (eps 0.212: not <= 0.15, not >= 0.25). Converging the fit nearly doubles the kept share (0.25 -> 0.46) but misses the 0.62 bar; my prediction (eps 0.10) was 2x optimistic.
   The eps-loss proportionality broke: loss fell 2.3x (4.16 -> 1.79), eps only 1.4x (0.296 -> 0.212). The three cosine/constant points on the same inputs lie on
   eps ~ 0.149 + 0.035 x loss(e-6) (f1 predicted 0.333, measured 0.325); it must bend to 0 at zero loss, so below ~2e-6 the MSE loss stops being the measure that sets N.

FIRST-ORDER COPY COST, DERIVATION + LINE TEST (lead 2026-11-23)
Data (round-6 copies, existing nn.json/grade.json): eps_6 / sqrt(val MSE, e-6): f1 0.142, NNc 0.145, f2 0.158, d 0.132, O3 0.131, O1 0.179, O2 0.120; eps_6 / MSE spreads
 0.062-0.119. So eps scales like |d|, not |d|^2, over a 3x range of loss. A variational (second-order) cost would scale like |d|^2 (the D-norm at alpha=0 is that cost, and the
 D-norm arm e, eps2 0.031 lowest, has eps 0.271, so the D-norm does not set eps either).
Derivation [exact algebra]: with fixed exact sign, the stepped guide is linear in the copy: g = P chi, P = 1 + alpha(H - E). With chi = T + dchi and sign(g) = sign(g_T),
 N[g] = <g|H_FN|g>/<g|g> is a ratio of quadratics, so N[P(T+dchi)] - N[PT] = 2<P dchi | r> + O(dchi^2), r = (H_FN - N_T) PT / |PT|^2  (alpha re-minimised: same order).
 r = 0 only if PT is an eigenvector of H_FN; it is not (PT is one round, not the FN ground state), so the linear term is generic [exact]. The fit errors make it positive in
 every arm [measured, 7 arms; why the MSE error aligns with r is not derived: conj, smoothing of structure the step created raises the energy].
Consequence [derived from the above]: the MSE copy loses at first order, so halving the error only halves eps; and the reverse direction should LOWER N below the exact round.
 The copy loss should be MSE + 2<P dchi|r> (fit to T shifted against the gradient of N at T), with r computed once per round (z^3 per row, not per stored amplitude: flat cost kept).
LINE TEST (existing tables, ~40 min CPU, rung 2): tables lT + t (l_f2 - lT), t in {0.5, -0.5, 0.25}; grade R. f(t) = N(t) - 1.293669e-6, f(1) = 0.212.
 PASS (first order): f(0.5) in [0.08, 0.14] and f(-0.5) < 0 -> design the gradient-shifted copy loss (paper check next). KILL (second order): f(0.5) <= 0.065 and f(-0.5) >= 0.03
 -> the sqrt scaling is an across-fit coincidence; the cost is quadratic in a norm I have not identified. Prediction: f(0.5) 0.11, f(0.25) 0.055, f(-0.5) -0.06 (60%).
LINE TEST, RESULT (lead 2026-11-23; ref2hop/NNl/{L0,L1,Lp25,Lp5,Lm5}/grade.json; known cases L0 rel 4.4e-8, L1 rel 1.4e-7)
  t:      1.0     0.5     0.25    0      -0.5
  f(e-6): 0.2124  0.1043  0.0504  0      -0.0598     (f/t 0.212, 0.209, 0.202, -, 0.120)
  PASS (f(0.5) 0.104 in [0.08,0.14], f(-0.5) < 0); predictions 0.11/0.055/-0.06 met. The copy's energy cost is first order in its error, and the exact round is NOT a local
  minimum of the grade: Lm5 (T - 0.5 x copy error) reaches N 1.2339e-6, 0.060 below the exact round (115% of round 6's exact gain from chi_5 at 1.6886). Curvature shows on
  the negative side (f/t 0.120), so the descent along this line saturates near t ~ -1.
  Meaning: the MSE copy errs systematically along the grade's gradient (why: not derived). Next (paper first): write the gradient of N at T (one pass, z^3 per row, once per round),
  the cosine of the f2 error with it, and the copy loss MSE + 2<P dchi|r>; then the fit. Not covered: one copy (f2), one round, no other direction tested.

WHERE THE GRADE SEES THE COPY, DERIVATION (lead 2026-11-23; code: word_moments.py grade -> delta0.efn_vec, bond_split.build)
Read from the code [exact]: N = min_alpha (lambda_min(A[a]) - E0)/NS, A[a] = diag(D0 + V[a]) - K, V_x[a] = sum_y Vm_xy a_y / a_x, a = guide magnitude of the stepped table
 (la + ln|1 + alpha(E_L[la] - E)|, sign s fixed). K (kept bonds of s) and D0 do not depend on the copy. So N depends on the copy ONLY through the violation field V_x:
 the ratios a_y/a_x across the sign-violating bonds of s (and through E_L inside a, one hop further). Errors anywhere else cost nothing, at any order.
Bound [exact, Rayleigh-Ritz with the trial vector fixed]: for v_T the normalised FN ground state of A[a_T] at alpha_T,
   N(copy) <= N(T) + sum_x v_T,x^2 (V_x[a_copy] - V_x[a_T]) / NS   (alpha fixed at alpha_T; re-minimising alpha only lowers N).
 First order (Hellmann-Feynman) equals the same sum, so the bound is tight to first order, and the measured linear cost (LINE TEST) is this sum.
Consequence: the copy loss should be L[a] = sum_x v_x^2 V_x[a] (a variational upper bound, v from the exact round, once per round), not MSE on log chi.
 Per pair of violating rows the minimum of v_x^2 a_y/a_x + v_y^2 a_x/a_y is at a_y/a_x = v_y/v_x: the guide should take the FN ground state's ratios across violating bonds.
 Why the MSE copy errs uphill [conj, 60%]: smoothing flattens the steep ratios across violating bonds (they sit next to sign changes), raising V where v^2 is large.
CHECK ON EXISTING DATA (rung 2, CPU ~1 h): v_T from efn_vec at alpha_T -0.040789 on the exact table; B(table) = sum v_T^2 (V[a_table] - V[a_T]) / NS for the 5 line tables and the
 7 round-6 copies (NNc, f1, f2, d, e, O1-O3).
 PASS: B >= f (= N - N_T at the table's own best alpha) for every table (theorem: a failure is a code error), and dB/dt at t=0 within 15% of the line's slope (cubic fit to
 f(-0.5, 0, 0.25, 0.5, 1)). Then: paper check of the bound as a copy loss. KILL: slope off by > 30% -> the cost is not in V (re-read the grade). Prediction: B(f2) 0.25,
 B(Lm5) -0.05, slope match within 10% (70%).

GRADIENT OF THE GRADE AT THE EXACT ROUND, AND A SHIFTED COPY TARGET (lead 2026-11-23, in parallel with the bound check)
Derivation [exact, chain rule through the grade's code; alpha fixed at alpha_T, envelope theorem]: b = e^u (table magnitude), M = Hmag (b -> D0 b + Vm b - K(sqrt(n) b)/sqrt(n)),
 E_L = Mb/b, delta = E_L - E, l = u + ln|1 + alpha delta|, a = guide(l), V = (Vm a)/a, v = unit FN ground state at alpha_T.
 gamma = d lambda/d l = a (Vm^T (v^2/a)) - v^2 V;   c = alpha gamma/(1 + alpha delta);
 d lambda/d u = gamma + b (M^T (c/b)) - c E_L - (sum c) dE/du,  dE/du_z = 2 n_z b_z ((Mb)_z - E b_z)/sum(n b^2);   g = (d lambda/d u)/NS.
 Checks: sum_z g_z = 0 (N is blind to a constant shift of u) [exact]; <g, d_f2> = slope of the LINE TEST at t = 0 (~0.2e-6) [first order].
Shifted target: the MSE copy loss plus the linear term, sum w (u - U)^2 + kappa <g, u - U>, is minimised by u = U - kappa g/(2w): the copy should be fitted to
 U*(eta) = U - eta g/w (w = n p^2 the fit weights), clipped at |shift| <= c. Line search: N(U*(eta)) at fixed alpha_T for a grid of eta, c in {0.5, 2}; full grade on the best two.
 PASS (worth a fit): best N(U*) <= 1.23e-6 (below Lm5, the reverse-error line), then the f2-type fit (cosine 80k) on U*. KILL: no eta gives N below 1.27e-6 -> the gradient
 step is too local; use the reverse line (Lm5 target) instead. Prediction: best N(U*) 1.18e-6 (55%).
BOUND CHECK, RESULT (lead 2026-11-23; ref2hop/NNv/bound.json; known case rel 4.4e-8; B = Rayleigh R to 1.4e-7)
  B >= f on all 13 tables (theorem holds), but B is 8-60x loose: L1(f2) B 12.45 vs f 0.21; Lp5 1.89 vs 0.104; Lp25 0.44 vs 0.050; Lm5 +0.41 vs -0.060; copies B 5.5-9.9 vs f 0.18-0.41.
  B(t) is strongly convex (odd part from t=+-0.5: 1.48/unit t; even part 1.15 at t=0.5) - the expansion is not linear even at t=0.25. Reading: V contains e^{d_y - d_x} on violating
  bonds, and the copy error d reaches several units on low-weight rows (target range -15.6..7.6, net output -2.2..2.7); with v frozen those rows blow V up, while the true
  ground state moves its weight off them (lambda adapts, the bound cannot). Pass line (slope within 15%) not decidable from these points; the finite-difference test at
  eps 0.02 in NNg decides first-order agreement. Consequence: the frozen-v bound is NOT a usable copy loss (it would chase rows the true grade ignores).
GRADIENT, RESULT (lead 2026-11-23; ref2hop/NNg/{grad.json,line.json,G1,G2})
  g exact: sum g / sum|g| 2e-15; FD along the f2 error 1.36932e-7 vs <g,d> 1.36944e-7 (ratio 1.0001); broken (no E_L chain) 1.85, (no M^T) 0.245. Along the f2 error the
  first-order slope is 0.137e-6 per unit t (65% of f2's 0.212 cost). Shifted target U - eta g/w: g^2/w sits on 0.2-0.4% of rows with weight <= 2e-11, the clip removes 98% of it,
  best N 1.2814 (fixed alpha), full grade G1 1.2861. Grid too short (gain still linear in eta at the largest eta), so neither pass nor kill. Insight: the grade's steepest
  direction in the fit's own metric lives on rows the MSE fit gives no weight; at equal MSE size it offers ~40x the first-order gain of the f2 error direction (3.96e-3 x |step|_w).
NEXT, TWO IN PARALLEL (lead 2026-11-23):
 (A) extended line, existing tables: eta up to 1 at c = 2 and c = 5, fixed alpha_T, full grade on the best. PASS N <= 1.23e-6. Prediction: best 1.22e-6 near eta 0.3 (50%).
 (B) pre-corrected refit: f2-type fit (cosine 80k, same inputs NNc/mapP5 l_part) on the target 2 lT - l_f2 (the exact round minus the f2 error). If the network repeats its
     error (d' = d), the copy lands on T (q 1). PASS q_6 >= 0.75 (N <= 1.392e-6); KILL q_6 <= 0.46 (no better than f2: the error is not reproducible). Prediction q_6 0.78 (50%).
EXTENDED LINE (A), RESULT (lead 2026-11-23; ref2hop/NNg2): best N 1.2637e-6 at c 2, eta 0.1 (gain 0.030); c 5 diverges (2.00 at eta 0.18). KILL for (A): the steepest direction lives on
  rows where a large shift leaves the linear regime at once.
INPUT FLOOR AT THE ENERGY-SENSITIVE ROWS (lead 2026-11-23; existing data, CPU ~1 h)
Mechanism [conj, 60%]: any copy u = F(z) of the 11 inputs z is constant on rows with equal z. If the rows that carry g are not resolved by z, the best F still pays
 the first-order cost <g, U - E[U|z]>, whatever the loss or optimiser. Estimate E[U|z] by k-nearest neighbours in standardised z (k = 16, reference = all rows, weights w).
 Floor = <g, U - Ubar_kNN> over the rows carrying 95% of sum|g|; also the same with z + 2 candidate inputs (V_x of B, and the count of violating bonds at x) to see if they close it.
 PASS (inputs are the limit): floor >= 0.08e-6 (>= 60% of f2's 0.137 first-order cost) and the extra inputs cut it by >= 2x -> build the copy on them. KILL: floor <= 0.03 ->
 the inputs resolve the rows; the limit is the fit's weighting, not the inputs. Prediction: floor 0.10, with extras 0.04 (55%).
CORRECTION + WEIGHT-FLOORED STEP (lead 2026-11-23; quick look at NNg/g.npy, f2 error d)
  g lives on HIGH-weight rows: 2.5e4 rows (1.6e-3) carry 50% of sum|g| and 66% of the fit weight w; 8.9e5 rows carry 95% of |g| and 97% of w. The f2 first-order cost <g,d> 0.137
  (+0.227, -0.090) sits on 1.1e4 rows (50%) / 6.2e4 rows (90%). So my 'gradient lives on rare rows' (GRADIENT RESULT, EXTENDED LINE) was wrong: g/w put the STEP on rare rows,
  g itself is in the bulk. The MSE fit already weights these rows most, yet errs along g there: the grade is far more sensitive per row than w says.
  Test (existing tables, ~1 h): shift = -eta g / max(w, w_f), w_f = the w of the 1e5-th and 1e6-th heaviest row, clip |shift| <= 2; eta grid until N rises twice; full grade on the best.
  PASS N <= 1.23e-6 (beats Lm5) -> fit f2-type copy on that target; KILL best >= 1.27. Prediction: best 1.21e-6 (50%).
  (A) full grade G1 (c 2, eta 0.1): N 1.2803e-6 at alpha -0.0478, ABOVE its fixed-alpha_T value 1.2637 (NNg G1 the same: 1.2861 vs 1.2814). So the grade's alpha search
  (coarse grid 0,-0.1,... plus a 3-point parabola) can miss the minimum by up to 0.017e-6. CHECK (rung 2, ~30 min): fine alpha scan -0.030..-0.070 step 0.005 at fixed table for
  f2 and NNg2/G1; if the true minimum of f2 lies >= 0.01 below its graded N, every copy's eps is overstated and all grades need the fine scan. Prediction: f2 overstated by < 0.005 (60%).
INPUT FLOOR, RESULT (lead 2026-11-23; ref2hop/NNk/knn.json; known case <g,d_f2> exact; shuffled-g control ~0, 15-25x below)
  'Floor' sum_S g (Ubar_kNN - U): k 4/16/64 = 0.291/0.305/0.335 (z11), 0.285/0.301/0.334 (z13); extras cut it 1.5%. NOT A FLOOR: f2, itself a function of z, pays only 0.137.
  My spec was wrong: E[U|z] minimises the MSE, not the first-order cost <g, F(z) - U>; what a function of z cannot follow is the part of g invisible in z, i.e. the cost
  responds to G(z) = E_w[g|z] only. Also the standardised z spans -1.1e5..4.4e4, so Euclidean kNN is set by a few heavy-tailed columns. Inconclusive; inputs question open.
VISIBLE GRADIENT (rung 2, ~40 min): rank-transform each column of z (and z13 with log(1+V)), kNN k = 16/64 over all rows, Gbar(x) = w-weighted mean of g/w over neighbours;
  visible fraction r = <g, Gbar>/<g, g/w> over S. PASS (inputs are the limit): r(z11) <= 0.3 and r(z13) >= 2 r(z11). KILL (inputs see the gradient): r(z11) >= 0.6.
  Prediction: r(z11) 0.4, r(z13) 0.5 (40%; I now lean to 'inputs are not the main limit').
ALPHA SCAN, RESULT (lead 2026-11-23; ref2hop/NNa/scan.json; known cases rel 1e-8): the grade's alpha search overstates N: f2 min 1.486821 at alpha -0.050 vs graded 1.506093
  (-0.0193); G1 min 1.263554 at -0.040 vs graded 1.280277 (-0.0167). Cause [read from word_moments.py grade]: the 3-point parabola loop stops once the vertex is within 0.01
  of a cached alpha, and N moves 0.02e-6 over 0.009 in alpha. Every graded N in this file may be high by up to ~0.02e-6, so every eps/q is uncertain at that level (f2: q_6
  0.46 -> 0.51 if the exact round is graded right). FIX: refine to |d alpha| <= 5e-4 (golden section on the bracket), regrade the exact round and the copies.
VISIBLE GRADIENT, RESULT (lead 2026-11-23; ref2hop/NNk2/knn2.json; known case r(g' smooth) 1.0000; shuffled ~0.02-0.04)
  r(z11) 0.434 (k16) / 0.397 (k64); r(z13) 0.477 / 0.420. BETWEEN (not <= 0.3, not >= 0.6); prediction 0.4/0.5 met. About 40% of the grade's gradient (in the fit
  metric) is a function of the 11 inputs; the extras (log V of B, violating count) add 6-10%, not 2x. So the inputs are a partial limit, and not via B's violation field.
  Open, for after NNr/NNq: which storable quantity carries the other 60% of g/w (candidates: the copy's own step factor at the neighbours; per-bond rather than pooled V).
FINE REGRADE, RESULT (lead 2026-11-23; ref2hop/NNq/<name>/grade_fine.json, golden section to 3.25e-4 in alpha; known cases f2 <= 1.486821, G1 <= 1.263554)
  All 12 tables drop 0.010-0.027e-6 (exact round L0 1.293669 -> 1.273672 at alpha -0.0298; chi_5 1.688596 -> 1.665985). The bias is nearly common, so the per-round
  shares barely move: q_6 n6fit 0.254, f1 0.185, f2 0.458, d 0.347, e 0.288, O3 0.542; line Lp25 0.887, Lp5 0.753, Lm5 1.127; G1 1.026. All conclusions above stand.
  From now on, grades use grade_fine.py (NNq); old grade.json N are 0.01-0.03e-6 high.
PRE-CORRECTED REFIT (B), RESULT (lead 2026-11-23; ref2hop/NNr/r1/r1fit, one seed, gradcheck 3.5e-7; numbers re-read from grade.json/nn.json)
  N 1.4972e-6 (old grade) -> q_6 0.485 vs f2 0.462 on the same scale: BETWEEN by the lines, but the mechanism is settled. The refit repeats its error: cos(d_r1, d_f2) 0.871,
  projection coefficient 0.88; MSE_T 1.797e-6 vs f2 1.757e-6. Diagnosis [exact for a linear fit]: if the fit is a projection P onto what the inputs/class can reach and
  d = PT - T, then P(2T - PT) = PT, so d' = d. 88% of the f2 error is outside the class's reach; only ~12% was optimiser noise. I predicted q 0.78 (50%) without writing
  this one line; it would have predicted the miss. Closes target-shifting of any kind (pre-correction, g/w shift): the copy error is a reach limit, not a target or loss limit.
WHERE THE UNREACHABLE PART LIVES (lead 2026-11-23; rung 2, existing data, CPU ~1-2 h; ref2hop/NNw)
  Derivation [exact]: chi_5 = B e^{u5(z)} is a function of z, so the copy residual d_f2 is the part of U6 = c1 + c2 not resolved by z, with c1 = ln|1+alpha(E_L[chi_5]-E)|
  (the round's step; E_L needs u5 at the neighbours: up to three hops of B) and c2 = 1.307 f_J[G]. Measure, by w-weighted kNN (rank-transformed, k 16/64, as NNk2/knn2.py),
  how much of d_f2 becomes predictable when ONE column is added to z11: R2_w(d | z11 + c), and the first-order cost left, <g, d - dhat>/<g, d> over S.
  Columns: c1, c2 (both non-storable; reference only: they say WHICH term carries the residual). Known case: c = d_f2 itself -> R2 >= 0.9; broken: c shuffled -> R2 as z11 alone.
  Reading: c1 R2 >= 0.5 -> the residual is the round's own local energy; next on paper a flat-cost estimate of E_L[chi_k]. c2 >= 0.5 -> derive f_J's structure.
  Both <= 0.2 -> the residual is spread over many hops; next a network with c1 as input as an upper reference. Prediction: c1 0.55, c2 0.15 (45%).
WHERE THE UNREACHABLE PART LIVES, RESULT (lead 2026-11-23; ref2hop/NNw/nnw.json md5 3d59ddb4; numbers re-read; known cases: c2 = 1.307 f_J to 3.6e-15, c = d gives R2 0.95)
  R2_w(d | z11 + col) on S, k16/k64: z11 alone -0.03/0.03; +c1 0.600/0.597; +c2 0.151/0.188; +c1 shuffled -0.06/0.00. First-order cost left: +c1 0.44/0.46, +c2 0.79/0.82.
  The shuffled control misses my 0.03 tolerance by 0.004 at k16 S, in the harmless direction (a noise column blurs kNN, lowers R2); accepted. Prediction (c1 0.55, c2 0.15) met.
  So the copy's unreachable part is the round's own step, i.e. E_L[chi_5] at x, which needs u5 at the neighbours; f_J is minor. The 11 inputs see B's ratios pooled with
  Delta u = 0 and miss the e^{u5(y)-u5(x)} weighting. In round 6, c1 is storable at flat cost (chi_5 is a z-only network: E_L[chi_5] costs z x (z-cost), fixed).
  Open on paper: rounds >= 7, where chi_6 would itself carry c1, so E_L[chi_6] at neighbours nests one level deeper unless the input is E_L of a z-only shadow of chi_k.
NEXT (rung 3, PI 11-22 run permission): f2-type fit (cosine 80k, same z11) plus c1 as a 12th input (NN_EXTRA2 = NNw/c1.npy), target T6; grade_fine.
  PASS q_6 >= 0.75 -> derive and test the round-7 shadow design. KILL q_6 <= 0.55 -> the fit cannot use c1; diagnose. Prediction from cost_left 0.45: eps 0.21 -> ~0.10, q_6 0.74 (40%).
ROUND >= 7 AT FLAT COST, PAPER (lead 2026-11-23)
  Exact rounds spread one hop per round: E_L[chi_k](x) = D_x + sum_b h_b rho^B_b(x) e^{u_k(y_b)-u_k(x)} needs u_k at the neighbours, so a copy that takes the round's step as
  an input reaches one hop further each round unless the input is computed from something of fixed reach. Shadow design [conj, 50%]: keep sigma_k = a z-only fit of chi_k
  (fixed reach R) and feed s_k = ln|1+alpha(E_L[sigma_k]-E)| (reach R+1, cost z x cost(z), flat). Its error vs the true step: E_L[chi_k]-E_L[sigma_k] =
  sum_b h rho e^{Delta g_k}(e^{Delta v_k}-1) with v_k = u_k - g_k the non-z part of chi_k: FIRST order in v_k, so the shadow step is good only if v_k varies slowly
  across a bond. Check on existing tables (rung 2): treat the exact T6 as chi_6 (not z-only) and f2 as its shadow. Compute U7 = log M[T6] - lB, c1_true = step from T6,
  c1_sh = step from f2; residual of U7 after kNN on z11; R2 gain from z11 + c1_true vs z11 + c1_sh. PASS shadow keeps >= 80% of the true step's R2 gain; KILL <= 50%.
  Prediction: 70% kept (40%).

SOFT (ANNEALED) SHADOW FOR THE CHAINED SIGN COPY, PAPER CHECK (sign angle lead, 2026-11-23)
Derivation: round map y_{k+1}(x) = sgn h(x), h = sum_b w_b(x) S_k(z_b) [exact, STEP 8]. The copy supplies c(z) for S_k(z) from features phi(z);
 dh = sum_b w_b (c_b - S_b). Let m(phi) = E[S_k | phi] (calibrated mean).
 HARD c = sgn m: E[c - S | phi] = sgn m - m (bias, |.| = 1 - |m|, sign = the copy's guess, i.e. toward S_{k-1} on missed flips); SOFT c = m: bias 0.
 Var(c - S | phi) = Var(S | phi) = 1 - m^2 for both, since c is fixed given phi [exact]. So E[dh^2] (hard) = E[dh^2] (soft) + (sum_b w_b (sgn m_b - m_b))^2.
 Deciding term: the bias field B(x) = sum_b w_b (sgn m_b - m_b). On a wall it is coherent (every missed flip pushes x toward the old sign), so it
 freezes walls; the soft shadow removes it and leaves zero-mean noise. Walker cost unchanged: same network, tanh(logit/2) instead of sgn.
 The annealing schedule beta_k of the review's form has no role: the bias-free shadow is E[S|phi], not tanh(beta h) (the network cannot see h);
 hard targets + a proper loss (BCE) already estimate m. The idea reduces to: calibrated soft shadow; walker sign stays sgn h-hat.
Checked: second route: c_hard - S = (sgn m - m) + (m - S), cross term E[(sgn m - m)(m - S)|phi] = 0, same decomposition. Limits: m = +-1 -> equal;
 m = 0 -> hard error 1 per neighbour coherent vs soft 0 mean. Theorems: conditional mean minimises mean square error (textbook L2 projection);
 BCE is strictly proper, its minimiser is the true conditional probability (Gneiting-Raftery, JASA 102, 359 (2007), from memory, not re-read);
 networks are often miscalibrated (Guo et al. arXiv:1706.04599, from memory) -> Platt/temperature scaling needed. Cross-field: soft- vs
 hard-decision decoding in iterative decoders (soft messages beat hard ones), same structure. [cond] unbiasedness holds per neighbour given ITS
 feature; given all neighbours' features the sum can keep a bias if their errors are shared (T-CERT: 8x8 nets' errors shared). Label error
 depends on the full distribution, not only on E[dh^2]. Confidence the derivation is right: 90%; that it moves q by a lot: low (below).
Prior results: recall.sh/grep 'soft|calibrat|tanh|mean field': T-SOFT1 (PLAN_sign 11-08) used |u| to mark costly bonds, not soft shadows;
 T-CERT V2 used calibrated p_wrong(|m|) for margins (worked, 0.2-0.8% errors) and found 8x8 net errors SHARED (g05/g10 agree on 94%, wrong
 together). Nothing found on soft shadows in the round field. Literature search not run (arxiv/paper-search MCP down this session).
Known case: runs/j1j2/sign_bad/step9/toy_soft_shadow.py (sha 04bba136) 2D 256^2 zero-T synchronous majority, J1 ferro, J2 0/0.1 antiferro, lognormal
 bond factors, domain start (flips 5-7%/round, walls only); copy features (S_{k-1}, h_{k-1} + noise), m calibrated on half A, graded on B, 2 seeds.
 Identity: conditional bias soft 0.01-0.03 (= binning noise) vs hard 0.06-0.14; conditional variance identical to 3 digits. noise 0: q 0.98-0.99 both.
 q_{k+1} hard -> soft: noise 0.5: 0.60->0.68, 0.63->0.71, 0.70->0.78, 0.72->0.79; noise 1: 0.07-0.17 -> 0.22-0.29; noise >= 2: 0 -> 0.01-0.06.
 BROKEN (overconfident soft, tanh(3 atanh m)): bias returns (0.04-0.13) and q falls back toward hard (e.g. 0.643 vs 0.677/0.601). Check can fail: it did.
Not covered: real features (CF sees spins), shared errors across neighbours, chaining > 1 round, FN beyond first order.
Next rung (existing data, CPU minutes): Platt-calibrate CF logits per old sign on train-root neighbours; paired dq = q4(soft) - q4(hard) on the same
 roots, chain bootstrap. Pass dq >= +0.10 AND q4(soft, held-out) >= 0.75; kill dq < +0.03; between: real but small add-on. Prediction: dq +0.05
 to +0.12 [60%], q4(soft) reaches 0.75 on held-out [20%]: the toy says soft adds ~0.07-0.12, not the 0.25+ the bar needs from today's ~0.5.
Status: DONE. Verdict on paper: the soft shadow is right and free but is an add-on worth ~+0.1 in q; alone it does not reach 75-80%.
SOFT SHADOW, RUNG 2 RESULT (2026-11-23; runs/j1j2/sign_bad/step9/{soft_q4.py,soft_q4.json}; CF logits sha 5dcca951, Platt per chain-half)
 CF is near-calibrated already (Platt slope 1.02 held-out / 1.38 train, offset +0.15..+0.39 toward keep), so m ~ the hard copy almost everywhere.
 Paired dq = q4(soft) - q4(hard), chain bootstrap: train (74 flips) -0.005(46); held-out (12 flips) 2-fold Platt +0.76(51) [q4 0.43, one heavy
 flip], train-fit Platt -0.02(8). Pass line (dq >= 0.10 AND held-out q4 >= 0.75) NOT met; train sits in the kill zone, held-out unresolved.
 Predictions: dq +0.05..0.12 MISSED low on train; q4 >= 0.75 [20%] did not happen. Reading: the bias term the soft shadow removes is small for
 CF because CF's misses are confident (|m| ~ 1 on missed flips), not uncertain: the copy's error is in m itself, not in rounding m.
 Verdict: soft shadow is a free, harmless add-on; it is not the mechanism that reaches 75-80%. Goal rule: step back before any variant.

NORMALIZED-FIELD COPY (store v = h/Lambda, deploy sgn v-hat), PAPER CHECK (sign angle lead, 2026-11-23; idea: Astra cj-acff665c36ee route 2)
Derivation: Lambda(x) = sum_b |w_b(x)| > 0, v = h/Lambda in [-1,1], label y = sgn v [exact]. Regret R = (4/N) E_P[Lambda |v| 1{sgn v-hat != y}],
 q = 1 - R/G [exact, Astra eq 1, re-derived]. A sign error needs |v| <= |v-hat - v| [exact], so a uniform error eps gives R <= B(eps) =
 (4/N) E_P[Lambda|v| 1{|v| <= eps}] [exact, tight: attained by the adversarial error]. Deciding term: the error of v-hat at rows with |v| below
 ~0.05 (flips: median |v| 0.057). Why a field target could beat BCE on hard labels [conj]: it supervises distance from the wall on every row,
 is bounded (no log of a vanishing amplitude), and needs no new inference-time information. Walker cost: one net, as CF.
Checked: second route: worst case = flipping every row with |v| <= eps, gives q = 1 - B(eps)/G, matches (step10 'q_adversarial' = 'cert' at
 every eps). Limits: eps -> 0 gives q -> 1; eps >= max|v| gives no certificate. Signs: v and y agree on all 60k panel rows.
 Confidence the derivation is right: 95%; that a net reaches the needed error: 30% (below).
Prior results: TRIED_IDEAS C16/E9 (bond-field compression failed, other setting: amplitude/FN, not a sign label); P14/P16 tangent-kernel
 learnability (held-out estimates need exact calibration); STEP 7e linear surrogate 0.40; CF's best monotone map onto v (step10, binned
 E[v|l_CF], held half of the panel): rms 0.090, median |err| 0.051; on round-3 flip rows rms 0.146, median 0.111 vs their median |v| 0.057.
Known case: runs/j1j2/sign_bad/step10/cert_check.py (sha 8dd2f266), Astra's leak-free 60k panel, exact arithmetic, no network.
 Certificate vs adversarial error: equal at eps 0.01-0.05 (0.970/0.896/0.804/0.765/0.516). BROKEN (true error 2 eps, certificate claimed at eps):
 q 0.672 < 0.896 (eps 0.02), 0.217 < 0.804 (0.03): the check fails as it must. Unstructured Gaussian error, rms -> q: 0.02 0.95, 0.03 0.90,
 0.05 0.71, 0.07 0.41, 0.09 -0.15. CF's actual q 0.457 at monotone-map rms 0.09 -> CF's errors are better than random at its rms.
Not covered: whether a spin-input net reaches held-out rms <= 0.045 on v; later rounds; FN beyond first order (Astra eq 13).
Next rung: a training run (worker; same CF class and data, regression on v instead of BCE on y). Measured: held-out panel rms(v-hat - v) and q.
 Pass: held-out q >= 0.75 on the leak-free panel. Kill: q <= 0.457 (no better than CF) at converged training. Between: field target helps but
 short of the bar -> learning-curve diagnosis before any variant. Prediction: rms 0.05-0.07, q 0.45-0.72 [65%]; q >= 0.75 [25%].
Status: DONE (rung-1). The run needs the PI's go under point 9; the goal (PI 2026-11-23) allows a test after rung 1.
ROUND >= 7 AT FLAT COST, RESULT (lead 2026-11-23; ref2hop/NNy/nnw7.json md5 420336eb, re-read; known cases: chi5 -> NNw c1 and lT exact; +d7 R2 0.95-0.96)
  On M[T6] (alpha -0.040789), k64, S: R2 z11 0.150, +c1_true 0.516, +c1_sh 0.488 -> R2 gain kept 0.92 (PASS by the written line). First-order cost left: z11 0.756,
  +c1_true 0.371, +c1_sh 0.495 -> cost reduction kept 0.68 (k16: 0.77). The grade responds to the cost, so the honest reading is BETWEEN: the shadow step keeps
  about 70% of what the true step buys (my prediction 70%). c1_sh and c1_true correlate only 0.50-0.60 yet carry nearly the same kNN information about d7.
  k16 shuffle control off by 0.10 in the harmless direction (negative baseline); k64 read used. Viable as the round >= 7 input; decided after NNx (round 6 with c1).
WEIGHT-FLOORED STEP, RESULT (lead 2026-11-23; ref2hop/NNg3, G1/G2 grade.json re-read; known case w_f 0 reproduces NNg2 G1 to rel 0)
  shift = -eta g/max(w, w_f) from the exact round T6, clip 2, alpha_T fixed, then full (old) grade:
  F6 (w_f 9.5e-9) eta 13.2: N 0.8164e-6 (alpha -0.0581), eta 7.34: 0.9476. F5 (w_f 7.0e-7) best 1.0895 at eta 14. Exact round T6: 1.2937 (same grade).
  PASS (<= 1.23) by far; prediction 1.21 missed by 0.39: the line was still descending at eta 13 and the clip touched only 1e-7 of rows.
  Meaning: one exact-gradient step from T6 lowers the FN error 37% below the exact round and below the exact chain's fitted fixed point N* 1.05 [approx, arm R].
  So the round map is a slow optimiser of the grade; q > 1 per step is possible, and the copy's target need not be T. The step lives where w is small (w_f 9.5e-9 beats 7e-7).
NEXT [parallel]: second step from G1 (g at G1 with its alpha, FD check, same line). Reading: gain >= 0.1e-6 -> descent not saturated, iterate to the fixed-sign floor;
  gain <= 0.03e-6 -> G1 is near that floor. Prediction: N 0.70 (40%). Also fine-grade G1.
  COST CORRECTION (lead 2026-11-24, after the sign angle's G(-5) timing, PLAN_sign.md STEP 8h): the shadow step needs z(y) at all d ~ 142 neighbours, so a copy with it costs
  ~d x (about 100x) per amplitude more than the z-only copy, and E_L of that copy another factor d. Flat in the round (the PI's criterion), not cheap; the prefactor is a PI call.
ROUND 6 WITH THE STEP AS INPUT, RESULT (lead 2026-11-24; ref2hop/NNx/x1, fine grade x1fine/grade_fine.json md5 38abe606, re-read; known cases: EXTRA2 unset reproduces f2 step 1)
  N 1.349949e-6 (alpha -0.0382) -> q_6 = (1.665985-1.349949)/(1.665985-1.273672) = 0.806. PASS (>= 0.75); prediction 0.74 (40%) slightly low. One seed, not yet quotable.
  val 6.1e-7 vs f2 1.79e-6; MSE_T 6.12e-7 vs 1.76e-6; cos(d_x, d_f2) 0.49: the f2 error is largely removed, not repeated (r1: 0.87).
  Round 6 used the exact step c1 (chi_5 z-only). Cost ~d x per amplitude (COST CORRECTION above).
NEXT [parallel] (a) round 7 with the SHADOW step (the constant-in-round design): chi_6 = NNx/x1 table; exact map M[chi_6]; sigma_6 = z-only fit of chi_6; c1_sh from sigma_6
  (alpha, E of the round-7 map); fit z11 + c1_sh -> T7; q_7 = (N(chi_6) - N(copy))/(N(chi_6) - N(T7)), all fine-graded. PASS q_7 >= 0.75; KILL <= 0.55; between -> improve the shadow.
  Prediction: q_7 0.70 (40%; NNy kept ~70% of the cost cut). (b) second seed of NNx for a quotable q_6.
WHICH ROUTE (lead 2026-11-24, PI unsure): round copy with step input passes (q_6 0.806) at ~d x per amplitude (and E_L ~d^2); one exact-gradient step on the table beats
  the exact round 37%. Deciding question: can the CHEAP z-only network hold the descended table? NNh: f2 setup fitted to NNg3/G1. N <= 1.30e-6 -> descent route wins on cost;
  N >= 1.45 -> it does not (the gain sits on low-w rows the MSE ignores). Prediction: 1.38 (35%); I lean 55/45 to descent as the better route if it can be stored.
  SEED 2 (NNx/x2, NN_SEED=1, grade_fine e6410fb1, re-read): N 1.364918e-6 -> q_6 0.767. Two seeds: 0.806, 0.767 (mean 0.786, spread 0.038). Both >= 0.75. Two fits, not an error bar.

SECOND WEIGHT-FLOORED STEP, RESULT (lead 2026-11-24; ref2hop/NNg4, fine_G1ref/fine_G1/fine_G2 grade_fine.json re-read; off the main line)
From NNg3/G1 (fine N 0.8144e-6, alpha -0.0546), the same step with p recomputed at the new base (known case: reproduces NNg/g.npy exactly, p_out to 1e-7).
Line at fixed alpha: best eta 0.214 (N 0.7773e-6 at alpha_G1); N rose for eta >= 0.385 (0.778, 0.820).
Fine: G2 (eta 0.385) N 0.7476e-6 at alpha -0.0462; G1 (eta 0.214) N 0.7672e-6 at alpha -0.0501. Gain 0.067e-6, between the written lines
(>= 0.1 not saturated, <= 0.03 near floor); step 1 gave 0.48e-6 (old grade). Steps shrink about sixfold: the descent is slowing, not floored.
Tables md5 G1 024dc321, G2 ca5a7be5. Not checked: a third step; eta between 0.21 and 0.39. No further steps queued (off the main line, PI 11-24).

WHICH ROUTE, RESULT (lead 2026-11-24; ref2hop/NNh, hfine+Gfine grade_fine.json re-read, h1fit.log GRADCHECK 4.4e-7)
Cheap z-only copy (f2 setup, rng 0, 80k steps, val still falling 8.76e-6) of the descended table NNg3/G1: N 1.0980e-6 (alpha -0.0529); G1 itself 0.8144e-6.
PASS of the written line (N <= 1.30). Prediction 1.38: missed by 0.28 in the good direction. One seed, not quotable.
Same start chi5 (1.666), same pool, round 6:
  route                          stored N     cost per amplitude   kept share of its own correction
  exact round, cheap copy (f2)    1.486          z-only              0.458
  exact round, step input (NNx)   1.350/1.365    ~d x z-only         0.806/0.767
  round + one step, cheap copy    1.098          z-only              0.667  (= (1.666-1.098)/(1.666-0.814))
Diagnosis of the miss: the copy loses about the same absolute amount from either target (T6 1.274 -> f2 1.486: 0.21; G1 0.814 -> h1 1.098: 0.28).
I had assumed the step's gain sits on low-weight rows the copy ignores; the floor shift = -eta g/max(w,w_f) caps the shift where w < w_f, so the
step moves mostly the rows the copy's weight n p^2 already sees. The descent route keeps a smaller share (0.67 < 0.75 bar) of a gain 1.8x larger, cheaply.
NEXT (descent route, round 7; [parallel] with NNz): from h1, exact round T7h = M[h1], one weight-floored step G7 (eta line as NNg4), cheap copy c7 (NNh setup).
  q7 = (1.0980 - N(c7))/(1.0980 - N(G7)). PASS: N(c7) <= 0.95e-6 and q7 >= 0.60 (round 6: 0.667). KILL: N(c7) >= 1.05 (gain washed out).
  Prediction: N(T7h) 1.00, N(G7) 0.72, N(c7) 0.96 +- 0.08 (50%); q7 0.36-0.5, i.e. I expect degradation of the share; the run tells whether N still falls.
  Plus seed 2 of h1 (NN_SEED=1) for the round-6 number.

ROUND >= 7 AT FLAT COST, RESULT (lead 2026-11-24; ref2hop/NNz, post7.json + four grade_fine.json re-read; GRADCHECK <= 3.7e-7; known case exact, broken alpha+1e-3 fails by 10.7)
Start chi_6 = NNx/x1 (1.349949e-6). T7 exact round 1.0961; shadow sigma_6 1.4791; R7 plain cheap copy 1.3477 (q 0.009: washed out);
X7 copy with the shadow step as input 1.1795 -> q_7 0.672. BETWEEN (pass 0.75, kill 0.55); prediction 0.70. One seed.
Diagnosis: the shadow step s = ln|1+alpha(E_L[sigma_6]-E)| differs from the true step c1 by alpha(E_L[sigma_6]-E_L[chi_6])/(1+...),
and sigma_6 is a worse copy (1.479 vs 1.350), so the input carries less of the step: 0.81 (true step, round 6) -> 0.67 (shadow, round 7).
Cheapest variant: a better shadow (sigma_6 val still falling at 80k). Deferred: the descent route (round 6 stored 1.098 at z-only cost, below X7's
round-7 1.180 at ~d x cost) is ahead; NNd7 (its round 7) decides which route gets the next spend.

PLANTED COPY (lead 2026-11-24; idea from BootLoops 'planted-truth' protocol, github BootLoops-ai/skills, CC BY 4.0): does the copy pipeline recover a
target the cheap network holds exactly? Plant = NNf/f2/f2fit/tables/t_R.npy (a z-only network's own output; fine N 1.486447e-6, alpha -0.0487).
Fit a fresh z-only network (NNh setup, NN_SEED=1, weights from NNc/map6/p_out.npy as f2) to the plant; fine-grade.
Clean fitting: |N - 1.4864| <= 0.03e-6 and weighted MSE to plant <= 1e-6. Fitting problem: N - 1.4864 >= 0.08e-6 (>= 30% of the 0.21-0.28 copy loss
would then be the optimiser, not the reach). Between: part. Prediction: MSE 1-2e-6, |dN| <= 0.03 (60%).
Broken version: plant with +0.5 added to log-amplitude on the 10% heaviest rows (by n p^2), graded directly; must move N by >> 0.03, or the grade cannot see the test's scale.

DESCENT ROUTE ROUND 7, RESULT (lead 2026-11-24; ref2hop/NNd7, four grade_fine.json re-read; known cases K1-K3 exact; GRADCHECK <= 3.1e-7)
From h1 (1.0980): exact round T7h 0.7705; + one step G7 0.6408; cheap copy c7 1.0707 -> q7 0.060 (round 6: 0.667). KILL (N(c7) >= 1.05).
h1 seed 2: 1.1043 (seed 1 1.0980; two fits, spread 0.006).
Prediction c7 0.96 +- 0.08: missed by 0.11. The corrections beat prediction (T7h 0.77 vs 1.00, G7 0.64 vs 0.72); the copy missed.
Diagnosis: stored N = N(target) + L. L = 0.21 (T6), 0.28 (G1), 0.43 (G7), while the weighted MSE of the cheap fit stays 8.5-8.9e-6 for every target:
the cheap network's residual in the n p^2 norm is fixed by what it can hold, so its stored N floors near 1.07 whatever the target.
The step-input network fits its target 16x closer (X7 val 0.54e-6 vs c7 0.89e-5); the inputs, not the correction, set the floor.
Verdict: the cheap route is closed as a chain (it stores one good round, 1.098, then floors). Corrections are not the bottleneck: from h1 they reach 0.64.
NEXT: [wait] NNp1 (planted copy) says whether the cheap floor is the optimiser or the reach. If clean fit: the reach -> the copy needs the
step inputs (X route; the only network that kept >= 0.67 at round 7); next spend there. If the fit is the problem: fix the optimiser first.

PLANTED COPY, RESULT (lead 2026-11-25; ref2hop/NNp1, pfine/bfine grade_fine.json + post.json re-read; GRADCHECK 1.8e-7)
Fit to the plant (f2 table, N 1.486447): N 1.495086, dN +0.009e-6 (<= 0.03); weighted MSE 3.0e-7 (<= 1e-6; val still falling). CLEAN FIT, one seed.
Broken plant (+0.5 on rows with 99.77% of the weight): N 7.755e-6, dN 6.27e-6 = 200x the test scale: the grade sees it.
Prediction (MSE 1-2e-6, |dN| <= 0.03): N held, MSE 3-7x better than predicted.
Reading: on real targets the same fit leaves MSE 8.5-8.9e-6 (h1, c7), 29x the in-class residual. The cheap network's floor (~1.07e-6) is what it
can represent, not the optimiser. Consequence for the shadow: sigma_k is a cheap copy, so it carries the same floor (sigma_6 1.479 vs chi_6 1.350);
a longer shadow fit is predicted not to fix q_7 0.67. Past the floor needs inputs the cheap network lacks, and their own copies must not be cheap ones.
NEXT (paper): which input carries the round's step at round >= 7 without a cheap shadow, and at what cost per amplitude.

WHICH INPUT AT ROUND >= 7, PAPER, step 1 (lead 2026-11-25)
Notation: frozen B; rho_b(x) = B(y_b)/B(x); joint rho_bc(x) = B(x^{bc})/B(x); Phi(x) = {rho_b(x)} (one hop). A reach-R network depends on x only via ratios up to R exchanges.
(1) [exact] Phi(y_b) = {rho_bc(x)/rho_b(x)}_c: the neighbours' one-hop summaries are functions of the two-hop ratios at x.
    Hence for a reach-1 network u = f(Phi), the round's step c1[u](x) = ln|1+alpha(D_x + sum_b h_b rho_b e^{f(Phi(y_b)) - f(Phi(x))} - E)| is an exact function of reach-2 data.
(2) [exact] By (1) the step of a reach-R network has reach R+1, so no fixed input set is closed under the round: a reach-R copy is exact for one round only, from a reach-(R-1) start.
    This is the one-hop-per-round reach limit (NNr) restated as a property of the input set, not of the fit; NNp1 (clean in-class fit) confirms the fit is not the limit.
(3) [exact, cost] The shadow scheme is the fixed-depth version of (2): stored X_k at reach 2 takes s_k = c1[sigma_k], sigma_k a reach-1 fit of X_k. Cost per amplitude ~ z x cheap,
    flat in the round. Its error enters only through ds = alpha sum_b h_b rho_b e^{u(y_b)-u(x)} (delta(y_b) - delta(x)) / (1 + alpha(E_L - E)), delta = ln sigma_k - ln X_k:
    the shadow's error differenced across neighbours. delta is bounded below by the reach-1 floor (NNd7/NNp1: weighted MSE ~8.6e-6), so q_7 0.67 < q_6 0.81 (true step) is that floor.
(4) [conj, p 0.55] Nesting one level deeper (reach-2 shadow = an X-type fit, its own step from a reach-1 shadow) keeps cost flat in the round at ~z^2 x cheap
    and cuts delta by the reach-2/reach-1 residual ratio (X7 val 0.54e-6 vs c7 0.89e-5: ~16x in MSE, ~4x in delta) -> q_7 ~0.78.
Deciding term: how fast the floor falls with reach R (16x from R=1 to 2 at round 7, one point). Open, on paper next: the upper bound for any shadow at round 7
(X7 with the true step c1[chi_6]) and whether its cost (chi_6 at two-hop configurations, ~z^2 B evaluations per row) is affordable on the pool; queued joint
two-exchange tokens (lines 1049, 1247, 1284, 1310, never run) are the same reach-2 information as (1).

WHICH INPUT AT ROUND >= 7, PAPER, step 2 (lead 2026-11-26)
(5) [exact, cost; corrects (4)] Count B evaluations per amplitude. Reach-1 copy: Phi(x) needs z. Shadow (depth 1): s = c1[sigma](x) needs sigma(y_b), each Phi(y_b): z^2.
    Depth-2 nest: sigma2(y_b) needs s1(y_b) = c1[sigma1](y_b), i.e. Phi at x^{bc}: z^3. In general z^{depth+1}; each level x z (~72 at 6x6).
    (4)'s "~z^2 x cheap" was wrong: depth 2 is z^3 ~ 4e5 B evaluations per amplitude, ~72x the shadow route. Flat in the round, not affordable for sampling.
    Same count for the upper bound "X7 with the true step c1[chi_6]": chi_6 at y_b needs c1[chi_5](y_b), i.e. Phi at x^{bc}: z^3 on 15.8M rows, infeasible. The round-6 value
    (0.81/0.77, true step affordable there because chi_5 is reach 1) stands as the bound, assuming the per-round content is stationary.
(6) [exact] So at affordable flat cost the scheme is depth 1, and q_7 is set by delta = ln sigma - ln X, sigma a reach-1 fit. NNp1 shows the optimiser is not the limit,
    but an in-class plant does not separate the two remaining causes of the reach-1 residual (MSE ~8.6e-6 on real targets): information beyond Phi (reach, (2))
    or capacity of the network on Phi (width/depth). Only the second is cheap to fix (cost stays z per amplitude).
Deciding quantity (rung 2/3): weighted MSE of reach-1 fits of the same real target (G7, NNd7) at width x1 (8.65e-6, measured), x4. Capacity-limited if x4 MSE <= 0.5 x;
reach-limited if >= 0.8 x. Prediction: x4 MSE 0.85 x (reach-limited, 60%), since R7 (reach-1 copy of a reach-2 state) kept q 0.009, as (2) predicts for missing information.
[wait] width test: prediction must stand one turn before a run (j1j2 CLAUDE.md point 9).

WALL OR REPRESENTATION? IS ROUND >= 7 NEEDED AT ALL (PI 2026-11-26, after Kairon-Clark arXiv:2609.16308; lead account, rung 1 + records)
Measured now (CPU, existing tables): the loop's fixed sign s2 (compress6/runs/refresh_17409419/s2.npy) vs sign(psi0) (stall6/data/psi0_6x6_table.npz):
  wrong on 2.68% of orbits, |psi0|^2 weight 1.23e-6, sign overlap 0.9999975 (global sign is gauge). So the amplitude loop does run in a wrong chamber, next to the right one.
Kairon-Clark (abstract only, lead 10-26): guide <- FN state converges to the GS only inside the GS chamber; elsewhere it collapses onto chamber boundaries.
Derivation [exact, fixed sign]: V_sf(x) = sum_{y viol x} |H_xy| psi_y/psi_x; on a row whose sign is wrong against its heavy neighbours all bonds violate, so V_sf ~ 1/psi_x and the
  FN step pushes psi_x down further (support collapse). Our map is not pure FN: G = A|1+alpha(E_L-E)| with alpha<0 and E_L(x) ~ +1/A_x on such a row RAISES |G|
  there (the Lanczos step wants to flip the sign; |.| keeps the magnitude, drops the flip). Net direction on wrong rows = f_J term vs Lanczos term: not derived. [open]
Energy value of a wrong row [exact, first order]: flipping row x changes E by -4 a_x sum_y H_xy s_y a_y / |a|^2, linear in a_x; a collapsing row costs ever less energy,
  so the PI's trade-off (easier exact flip, rarer for a sampler) mostly cancels in energy: rows that collapse matter little.
Against the wall as the round-7 limiter (records): the exact map still gains 19-30% per round at round 7 (chi_6 1.350 -> T7 1.096; h1 1.098 -> T7h 0.771; + one step 0.641),
  so the chain is not at a chamber floor; the q fall is arithmetic: copy error eps roughly constant (cubic 0.49-0.54e-6; cheap net floor 1.07e-6) while the round gain shrinks.
For the PI's division of labour (records): the sign the sign angle delivers from Marshall leaves 6.1(1.6)e-5/site (PLAN_sign STEP 17), ~60x the cheap copy floor and ~10x
  N(A_0) 8.68e-6. With that sign, amplitude rounds past 1-2 sit below the sign error, and the copy keeps q >= 0.74 through round 6. Round >= 7 accuracy binds only once the
  sign error is below ~1e-6/site. Coupling not yet derived: the Lanczos flip at x uses amplitude ratios (E_L), so amplitude error can mis-flip; the amplitude accuracy the sign
  step needs is the real requirement. [open, derivable]
RUNG-2 CHECKS (existing tables, CPU, no network; predictions written 11-26, run next turn per point 9):
 (a) chamber floor: E_FN[s2 |psi0|] (and E_var) with the fine grade. Prediction 0.1e-6/site (range 0.03-0.3; 60% below 0.3). Below 0.3: the s2 chamber has room under round 7's
     0.64, the wall does not limit rounds <= 7. Above 0.64: we are near the chamber floor and the PI's reading holds.
 (b) collapse: mean of ln(A_k/|psi0|) on the wrong rows minus on the rest, for every stored round table k = 0..7. Wall: falls >= 0.3 per round. Prediction: changes < 0.1 per round after
     round 2 (55%), since the Lanczos factor feeds amplitude into those rows.
 (c) where round 7's lost gain sits: grade c7 with G7 pasted on (wrong rows + one-hop neighbours) only; share of the lost gain recovered. Wall: >= 50%. Prediction 20-40% (50%).
 (d) the question that decides the full algorithm: one Lanczos sign step from the sign angle's sign, with amplitudes |psi0|, chi_2 and chi_6; post-flip E_FN. If chi_2 and chi_6 agree
     within << 6e-5/site, round >= 3 amplitude accuracy does not matter for the sign step. Prediction: differences <= 1e-6/site (70%).

CORRECTION (lead 2026-11-26): the 6.1(1.6)e-5/site I compared with the 6x6 copy floor is the 8x8 sign angle's gap to sgn(M1), not a 6x6 number. 6x6 equivalent:
  one signed-Lanczos step from the learned guide B2, then an exact FN refresh, leaves F(S_1) 3.13e-5/site (PLAN_sign STEP 0' RESULT). The conclusion (sign error >> copy floor
  1.07e-6) holds at 6x6 with this number. Also prior: C1 (exact K1 + FN refresh loop from Marshall converges at 6x6, 1e-6 rel at iteration 100); A4 (second K step without a
  refresh fails); STEP 0'/T-SPLIT (from learned guides the second sign step gains < 0.5e-6/site at 6x6).
SIGN-STEP DEPTH TEST (PI go 2026-11-26: 'run the round-2 versus round-6 comparison'; rung 2, existing tables, CPU, no network)
Question: does a sign step computed from 6 amplitude rounds beat one computed from 2, once both signs are graded at the SAME amplitude?
Step [exact def, q0r.py lanczos_step]: T[S,A] = sgn(G + alpha (H - E) G), G = S A, alpha = closed-form E_var-optimal. Depends on A only through local ratios (E_L).
Arms: start S in {s2, Marshall M}; A in {A_0 (pfaffian_c c1_cplx_s1 la_table), A_2, A_6 (EXACT rounds of the round map under S from A_0), |psi0| (oracle step T*)};
  s2 extra: copies chi_6 = NNx/x1, h1 = NNh. Grade: E_FN[T A_c] - E0 per site at common A_c in {|psi0|, A_2^S}, and E_FN[S A_c] as the zero.
Per-flip: flip sets F_A = {T != S}; vs F*: precision/recall (count, |psi0|^2 weight, first-order energy weight e_x = -4 S_x a_x sum_{y!=x} H_xy S_y a_y / |a|^2, a = |psi0|);
  F_2 vs F_6 Jaccard in the same three measures; per-flip correctness vs sgn(psi0). Ratio accuracy: on F_2 u F_6 u F* rows, E_L[S A_k] - E_L[S |psi0|] and the step margin
  1 + alpha(E_L - E); share of mis-flips (vs F*) whose ratio error exceeds their margin. Walls: mean ln(A_k/|psi0|) on S-wrong rows minus the rest, k = 0..6; E_FN[S |psi0|].
Predictions (lead, before any number; (d) of the block above, 70%, is the registered one):
  s2: |dE(T_2) - dE(T_6)| at |psi0| <= 0.05e-6/site; energy-weighted Jaccard(F_2, F_6) >= 0.8 (60%).
  M: step gain at |psi0| close to the Marshall FN gap; T_2 vs T_6 within 5% of the step gain (60%); if they differ, T_6 is WORSE (55%): rounds under M push A toward M's walls.
  Mis-flips: >= 70% of their energy weight has ratio error > margin (60%).
Decision (PI's rule): T_2 within 10% of T_6 in step gain AND energy-weighted flip agreement >= 0.9 at both starts -> park round-7 compression; next = the coupled loop
  (shallow amplitude rounds + sign step), judged by cumulative gain per cost against the baseline. Otherwise amplitude depth matters for the sign; compression stays.

SIGN-STEP DEPTH TEST, RESULT (amplitude lead 2026-11-26; job 18227574, runs/j1j2/consol_6x6/signdepth/result.json md5 c5cda3af; grades re-read from the file)
Known cases: K1 round map reproduces rounds23/lA_1 to 2e-16; K2 E_FN[s2 A_0] 2.092e-5, chi_6/h1 re-grade to 1e-9 rel; K3 wrong weight M 1.956e-2, s2 1.227e-6;
  K4 1% random flips of s2 -> 8.11e-3 (fails as it must); e_x = exact single-flip change to 1e-15, without self-loop term off 9%. All 26 grades gate_ok.
Step gain E_FN[S A_c] - E_FN[T A_c] per site (residual E_FN[T A_c] in brackets):
  | start | arm    | at |psi0|            | at A_2^S             | J_E(F_k,F_6) |
  | s2    | A_2    | 1.123e-6 (3.70e-7)   | 5.15e-7 (2.805e-6)   | 0.823 |
  | s2    | A_6    | 1.407e-6 (8.65e-8)   | 4.93e-7 (2.827e-6)   | 1     |
  | s2    | chi_6  | 1.342e-6 (1.52e-7)   | 5.14e-7 (2.806e-6)   | 0.953 |
  | s2    | h1     | 1.376e-6 (1.18e-7)   | 4.24e-7 (2.896e-6)   | 0.955 |
  | s2    | |psi0| | 1.490e-6 (3.6e-9)    | 1.45e-7 (3.175e-6)   | 0.953 |
  | M     | A_2    | 4.177e-3 (5.74e-4)   | 2.302e-4 (1.312e-3)  | 0.983 |
  | M     | A_6    | 4.160e-3 (5.91e-4)   | 2.421e-4 (1.300e-3)  | 1     |
Decision rule: M passes (A_2 = A_6 to 0.4% / 5%, J_E 0.983); s2 FAILS (A_2 gains 20% less than A_6 at |psi0|, residual 4.3x larger; J_E 0.823 < 0.9).
  So amplitude depth matters for the sign step near the exact signs, not from Marshall. Compression is not parked by the rule.
Predictions: s2 |dE_2 - dE_6| <= 0.05e-6 MISSED (0.284e-6); J_E >= 0.8 hit (0.823); M within 5% hit (0.4%; T_6 marginally worse, as predicted);
  mis-flips with ratio error > margin >= 70% hit at own alpha (0.96-1.00 energy share; h1 0.67).
What it says about round >= 7: the round-6 COPIES already give 95-98% of T_6's gain (chi_6 1.342, h1 1.376 vs 1.407e-6) and J_E 0.95. The most a perfect amplitude can
  add over A_6 is 0.083e-6/site (|psi0| arm). So depth from 2 to 6 rounds matters and the stored round-6 copy captures it; compression past round 6 buys <= 0.08e-6/site
  of sign quality at s2. Its value would have to come from the amplitude itself.
At the loop's own amplitudes (A_2^S) every sign arm gives the same 0.42-0.52e-6 and the residual 2.8e-6 is amplitude error: at s2 the amplitude, not the sign, limits E_FN.
Wall (mean ln(A_k/|psi0|) on S-wrong rows minus the rest, |psi0|^2-weighted): s2 -0.37 (k0), -1.10 (k2), -1.55 (k4), -4.22 (k6); M -0.05, -3.17, -4.31, -5.53.
  The exact rounds DO suppress wrong-sign rows, monotonically (the Kairon-Clark picture holds in our chain). This suppression is what the step reads: s2 flip precision
  (energy) 0.91 (A_2) -> 0.99 (A_6). The copies keep only part of it (chi_6 -1.10, h1 -0.79, about the A_2 level) yet reach precision 0.98-1.00: the copy's sign
  information is in the ratios, not the depth of the dip. Median |E_L error| on the flip union grows to 283 at s2 k=6 (vs 8-25 for k<=5): the round-6 ratios on those rows are badly off
  while the flip decision stays right.

FIRST COMPLETE CYCLE, RESULT (amplitude lead 2026-11-27; FN-lead's arms A-D; jobs 18231287-90; runs/j1j2/consol_6x6/cycle1/result.json md5 4af5e600, arm files re-read)
Known cases: B round 1 = NNz/map7 table to 9.4e-11 rel, N_out 1.0976940e-6 both; start grades chi_6 1.7609602 (alpha 0) / 1.3499 (fine), A_6 1.1173492; T files recompute
  with 0 mismatches; C (same flip count per |psi0|^2 decade as T, 3 isolated rows redrawn) worse than B every round by 0.68-0.79e-6.
E_FN alpha 0 / N best alpha, 1e-6 per site, rounds 0-3 of the round map at the arm's sign:
  | arm (start, sign, wrong weight)          | r0          | r1          | r2          | r3          |
  | A chi_6, T[s2,chi_6], 1.16e-7            | 1.421/0.951 | 0.807/0.603 | 0.529/0.452 | 0.405/0.373 |
  | B chi_6, s2, 1.23e-6 (control)           | 1.761/1.375 | 1.262/1.098 | 1.021/1.021 | 0.958/0.958 |
  | C chi_6, s2 + random flips, 2.57e-6      | 3.176/2.161 | 1.860/1.860 | 1.729/1.729 | 1.640/1.640 |
  | D A_6, T[s2,A_6], 6.6e-8                 | 1.105/0.690 | 0.603/0.452 | 0.409/0.372 | 0.333/0.311 |
Verdict: FN-lead's line PASSES (N <= 0.8e-6 within two rounds: 0.603 at round 1); kill (>= 1.0 after 3) far off. One sign step opens a chamber whose rest lies below
  the fixed-s2 rest: A at round 3 (0.373) is 2.6x below B (0.958) and below the s2 chain's Aitken floor 0.80; A is still falling (Aitken on A's N: ~0.29).
Sub-predictions: 'A round 1 worse than B at alpha 0' MISSED (0.807 vs 1.262; A is below B already at round 0). Amplitude lead's [parallel] prediction
  E_FN[T chi_6] alpha 0 = 1.26 +- 0.2: measured 1.421, inside the band at its edge.
Copy vs exact: A lags D by one round almost exactly (A r2 0.452 = D r1 0.452; A r3 0.373 = D r2 0.372): starting from the round-6 copy instead of A_6 costs one round.
Under s2 (B, C) the best alpha goes to 0 from round 2 on; under T it stays -0.04 to -0.12: the Lanczos factor keeps working after a sign step.
Not covered: T is an exact full-space sign step; the stored sign (sign lead) and stored amplitudes inside the loop are the next cycle; no baseline comparison yet.

GAMMA SCAN OF THE SIGN-STEP INPUT, RESULT (amplitude lead 2026-11-27; FN-lead's request; job 18231823; runs/j1j2/consol_6x6/gammascan/result.json md5 762926d4, re-read)
Known cases: K1 A_k rebuilt at gamma 1.307 to 3.6e-10 (k=2) / 2.7e-9 (k=6), broken E' fails at 1.3e-5 / 5.9e-7; K2 R(1.307) = 0.3703 / 0.0865 = saved, T bit-identical;
  K3 gamma -2 worse than 0 at both k. A_gamma = G_k e^{gamma f_J[G_k]}, R = E_FN[T[s2,A_gamma] |psi0|] - E0 per site, 1e-6:
  | k | gamma 0 | 1.307 | 2     | 3     | 4     | -2    |  E_FN[T A_gamma] at 0 / 1.307 / 2 / 3 / 4      |
  | 2 | 0.418   | 0.370 | 0.342 | 0.298 | 0.262 | 0.469 |  3.340 / 2.805 / 2.582 / 2.333 / 2.172          |
  | 6 | 0.091   | 0.0865| 0.0860| 0.0863| 0.0892| 0.128 |  1.188 / 1.105 / 1.064 / 1.010 / 0.960          |
Verdict: kill does NOT fire (k=2: gamma 3 and 4 beat 1.307 by 20% and 29%). Prediction 'min of R at gamma 2-3' MISSED at k=2 (still falling at 4, edge of grid),
  met at k=6 (min at 2, only 0.6% below 1.307: flat). 'R(A_2) <= 0.25e-6' MISSED (0.262 at gamma 4). At k=2 larger gamma raises flip precision
  (count 0.647 -> 0.781, weight 0.919 -> 0.980) at some recall cost (count 0.856 -> 0.759); at k=6 the sign step does not care about gamma.
Add-on (r = 0.03 sign step at |psi0|): from s2 E_FN 4.6e-10 (414060 flips); from Marshall 2.76e-4 (8728839 flips); r = 1 reproduces saved T at both.
Not covered: gamma > 4 at k=2; deterministic single run (no chain statistics apply).
STORED SIGN IN THE CYCLE (S-sign), RESULT (amplitude lead 2026-11-27; jobs 18237505 (S), 18237506 (K); runs/j1j2/consol_6x6/cycle2_ssign/result.json md5 e11fb334, re-read)
Sign: sign lead's STEP 21 avg2 (two-seed XL-CNN copy of T[s2,chi_6]), gauge-fixed S_avg2.npy md5 dbe7160e; w 5.742e-7 (lead re-grade; T 1.159e-7, s2 1.2265e-6).
Known case: arm K (chi_6 at T) reproduces cycle1 arm A rounds 0-1 exactly (rel 0, round-1 table md5 equal); broken (vs arm B) differs 19-45%.
  | round | E_FN alpha0 / N (S) | N (A, exact T) | N (B, s2) | wall unw / psi0^2 (S) |
  | 0     | 1.895 / 1.366       | 0.951          | 1.375     | +0.24 / -0.36         |
  | 1     | 1.221 / 1.056       | 0.603          | 1.098     | -0.75 / -1.31         |
  | 2     | 0.971 / 0.971       | 0.452          | 1.021     | -1.88 / -2.31         |
  | 3     | 0.901 / 0.901       | 0.373          | 0.958     | -2.15 / -2.44         |
Verdict: passes the bar (0.901 < 0.958) by 6% only; FN-lead's eq (10) prediction 0.61 (band 0.55-0.70) MISSED by +0.29e-6. S tracks B (N_S/N_B 0.99, 0.96, 0.95, 0.94), not A;
  best alpha goes to 0 from round 2 as under s2. The copy removes half of s2's wrong weight but almost none of the cycle gain: w is not the variable that sets N_3.
Registered before the next check (amplitude lead): if the sign's FN energy at the exact amplitude, x = E_FN[S|psi0|] - E0, sets N_3, the A-B line
  N_3 = 0.373 + 0.436 (x - 0.152) (x: T 0.1521, s2 1.4935, e-6) predicts x(avg2) = 1.36e-6 (band 1.2-1.5) and x(C) = 3.06e-6.
Not covered: seed0/seed1 single copies in the cycle; deterministic single runs.
Why w is the wrong weight (amplitude lead, paper step 2026-11-27, done while the x run waits).
  H_FN for sign s: each violating bond (s_i H_ij s_j > 0) loses its off-diagonal and adds |H_ij| to both diagonals. For psi = s a, a >= 0:
  (a) <H_FN>[s a] = <H>[s a] + sum_viol |H_ij| (a_i - a_j)^2                                       [exact]
  (b) one isolated wrong row i, a = |psi0|, all other signs exact: <H>[s a] - E0 = 4 sum_j |H_ij| a_i a_j, so the
      variational FN cost of the row is  c_i = sum_j |H_ij| (a_i + a_j)^2                          [exact, to first order in the single flip]
  (c) relaxing the amplitude (the FN solve) drives a_i -> 0; the |H_ij| a_j^2 added to the neighbours' diagonals stays:
      c_i(relaxed) ~ sum_j |H_ij| a_j^2 = the NEIGHBOURS' weight, not a_i^2                        [cond: isolated flip, first order in a_i/a_j]
  So w = sum_wrong a_i^2 undercounts a false flip on a small row beside large ones by ~ sum_j |H_ij| a_j^2 / a_i^2. avg2's error is mostly
  false flips (4.2e-7 of 5.7e-7) in |psi0|^2 decades <= -11: (c) predicts they cost far more than their w. Adjacent wrong rows share bonds
  that are then not violating, so clustered flips cost less than (c) sums: the cost is a bond count on the boundary of the wrong set.
  Consequence if it holds: the copy's target is the bond-boundary weight B(S) = sum_{viol ij} |H_ij| (a_i^2 + a_j^2), not w. The x run tests
  the scalar version (x = E_FN[S|psi0|]); B(S) is the per-row loss it would justify.
FN COLLAPSE MARKS THE WRONG SIGNS (Kairon & Clark check), RESULT (amplitude lead 2026-11-27; FN-lead's registered check; job 18238556; runs/j1j2/consol_6x6/collapse/result.json md5 8bf06323, re-read)
Known cases: f_J[G_2], f_J[G_6] rebuilt to the saved A_3, A_7 (broken E' fails); e* over oracle flips = signdepth flip_energy_signed; score = -label AUC 1; random 0.46-0.60.
Weighted AUC, label = s2 wrong vs sgn psi0 (low score = collapsed):
  | score          | psi0^2 | D3 (neighbour weight) | e* (clipped >= 0) |
  | ln A_1/A_0     | 0.738  | 0.819 | 0.934 |
  | ln A_2/A_0     | 0.785  | 0.864 | 0.963 |
  | ln A_6/A_0     | 0.9996 | 0.990 | 0.984 |
  | f_J[G_2] (one round) | 0.992 | 0.951 | 0.877 |
  | f_J[G_6] (one round) | 0.99993 | 0.994 | 0.9995 |
  | ln|psi0| (reference, not available in the loop) | 0.993 | 0.985 | 0.9998 |
Label shuffle within |psi0|^2 decades: collapse scores fall to 0.42-0.55 (psi0^2, e*), D3 stays 0.71-0.80; ln|psi0| stays 0.99. So the collapse carries
  information inside a decade (not just "small |psi0|"); the D3 residue is decade confounding of that weight.
Gate from one round, {f_J[G_6] < -c}, psi0^2 weight: cut 0.5x wrong weight -> precision 0.9997 / recall 0.50; 1x -> 0.893 / 0.893; 2x -> 0.465 / 0.932.
  D3 weight at 1x: 0.645 / 0.645.
Caveat on the registered numbers: e* with the graded sign is NEGATIVE on wrong rows (401,993 orbits, 388,511 of them wrong), so clipping to >= 0 zeroes
  the wrong class (W_wrong 6e-8 vs W_right 48): the e*-weighted AUC 0.984 and the e*-cut mass ratio 88 are not meaningful. Mass ratio s2/T on the
  psi0^2 cut: 10.5 (s2 chain k=6 vs arm A r3); matched depth (arm B vs arm A, both 3 rounds from chi_6): 8.7 (psi0^2), 1.34 (e*), 1.40 (D3).
Verdict: FN-lead's line passes on the meaningful weights (psi0^2, D3 AUC > 0.99 at k=6; kill 0.7 far); the mass-ratio claim holds on psi0^2 (8.7-10.5),
  not on D3 at matched depth (1.4). A single round's f_J under the wrong sign finds ~89% of the wrong weight at 89% precision, with no network.
SIGN FN ENERGY AT THE EXACT AMPLITUDE (x) AND ROW COSTS D1-D3, RESULT (amplitude lead 2026-11-27; jobs 18238171 + 18239675; runs/j1j2/consol_6x6/xpsi0/result.json md5 b0f87777, re-read)
Known: x(s2) 1.4935371e-6, x(T) 1.5207752e-7 to rel 9e-9; broken (s2 + 1% random flips) 8.11e-3 = signdepth K4. Per site, 1e-6:
  | sign  | x     | x_var | w     | D1 (e*, rows != T) | D3 (nbr weight, 1e-3) | N_3 in cycle |
  | T     | 0.152 | 0.163 | 0.116 | -                  | -                     | 0.373 |
  | avg2  | 0.772 | 0.878 | 0.574 | 0.743              | 0.752                 | 0.901 |
  | seed0 | 0.883 | 1.007 | 0.661 | 0.878              | 0.838                 | -     |
  | seed1 | 0.938 | 1.074 | 0.700 | 0.947              | 0.968                 | -     |
  | s2    | 1.494 | 1.653 | 1.227 | 1.527              | 1.829                 | 0.958 |
  | C     | 2.915 | 3.966 | 2.567 | 3.841              | 1.868                 | 1.640 |
Verdicts: my registered x(avg2) = 1.36 MISSED (0.772); the sign lead's 1.1-1.6 MISSED (its kill <= 0.8 fires); x(C) 3.06 vs 2.915 met.
  (x - x_T)/D1 = 0.83, 0.83, 0.83, 0.88, 0.72 (avg2, seed0, seed1, s2, C): the first-order e* cost D1 predicts the static sign energy to ~10%.
  D3/w is 1470 (avg2) vs 1600 (s2): the copy's scattered errors are NOT costlier per unit w at |psi0|; my eq (c) argument does not show here.
  No static scalar sets N_3: on the A-B line x(avg2) = 0.772 predicts N_3 0.64, measured 0.901. avg2 is half-way between s2 and T at |psi0|
  but behaves like s2 in three rounds from chi_6.
STEP BACK (amplitude lead, two failed variables w and x for one question):
  1. Verdict: continue, but change the observable: N_3 is a dynamic number (rest + speed of the round map from chi_6), and every static sign scalar
     (w, x, D1-D3) has missed it. Under avg2 the best alpha falls to 0 from round 2, as under s2; under T it stays -0.04..-0.12.
  2. Thought: a fixed-point iteration is judged by its contraction, not by the value at its fixed point; the stored sign may have a good rest
     (x 0.77) and a stalled map (alpha -> 0). Aitken on S's rounds 1-3 gives ~0.58 (B's gives 0.67, against its long-chain floor 0.80, so Aitken
     on 3 rounds is optimistic by ~0.13).
  3. Cheapest test: rounds 4-6 for S and B from their round-3 tables (CPU ~2 h each). Prediction: S N_6 0.72-0.80, B N_6 0.88-0.92; if S keeps
     falling at ratio >= 0.8 while B flattens, the stored sign's cost is speed, and the fix is in the map (alpha), not in the copy.
ROUNDS 4-6 AT FIXED SIGN (S = S_avg2, B = s2), RESULT (amplitude lead 2026-11-27; jobs 18241619, 18241620; runs/j1j2/consol_6x6/cycle4_extend/result.json md5 f75221c9, re-read)
  N per round (1e-6/site; best alpha = 0 at every round, E_FN alpha0 = N):
    round   3       4       5       6       drop ratio d5/d4, d6/d5
    S     0.9006  0.8412  0.7899  0.7451   0.864, 0.876
    B     0.9580  0.9042  0.8576  0.8167   0.866, 0.877
    B-S   0.0574  0.0630  0.0677  0.0716
  Known: round 0 = stored round 3 to double precision (both arms). Broken: start table graded at the other arm's sign 1.374e-6 (rel 0.53, 0.43).
  Prediction: S N_6 0.72-0.80 hit; B 0.88-0.92 missed (B fell to 0.817). The 'speed, not rest' reading (S falls, B flattens) is refuted: both
  contract at the same ratio 0.87 and the stored sign buys a level offset that grows ~0.005/round. B also passes the 'long-chain floor 0.80' cited
  in the step back by round 7 if the ratio holds; that floor came from a different chain and does not bind here.
  Reading (equation level): with the sign fixed, alpha = 0 is best from round 2 on, so the round map is A -> A e^{1.307 f_J[A]} alone; its
  contraction (0.87/round) does not depend on which fixed sign it sits under.
  CORRECTION (same day, after reading cycle3_gate): I first wrote here that one fixed sign cannot reach the margin and a per-round sign is needed.
  Wrong: G6 is one fixed sign for all three rounds and reaches N_3 0.534 with alpha -0.07..-0.035 kept nonzero. alpha dies under S (w 0.574) and
  B (w 1.227), not under G6 (w 0.263) or T (w 0.116): the Lanczos step survives only below some wrong weight between 0.26 and 0.57.
  Not checked: deterministic single runs, no error bars (no sampling); no rounds past 6.
GATE SIGN IN THE CYCLE, REGISTERED (amplitude lead 2026-11-27, before the run; owner of the stored loop per PI 11-27)
Sign S_g = s2 * (1 - 2 g), g = [f_J < -c]: flip s2 where the fixed-node round collapses (Kairon & Clark map; collapse block above). No network.
  G6:   f_J[G_6] of the exact s2 chain, c = 0.008762 (1x psi0^2 cut). Upper bound for the gate idea.
  Gchi: f_J of the round chi_6 -> (s2), i.e. cycle1 arm B round 1's G; same numeric c (no psi0 used to set it). This is what the stored loop has.
  Each: w, x = E_FN[S_g|psi0|], then 3 exact rounds from chi_6 at S_g (as cycle1/cycle2).
Mechanism: the gate's errors sit at the wrong domains' boundary (it reads the wall), so they come in domains, not scattered like the CNN's; by the
  isolated-row cost (c) above, domain errors cost their boundary only.
Prediction: G6 w ~ 0.26e-6, N_3 0.45-0.65 (50%); Gchi N_3 0.55-0.80 (45%). Objective margin N_3 <= 0.665 (half of exact gain). Kill of the gate route
  at 6x6: Gchi N_3 >= 0.90 (no better than the CNN copy avg2).
GATE SIGN IN THE CYCLE, RESULT (amplitude lead 2026-11-27; runs/j1j2/consol_6x6/cycle3_gate/result.json md5 75d0899d, build/result_build.json, re-read)
  Fixed sign, start chi_6, three exact rounds (1e-6/site):
    arm    w      P/R (psi0^2)   N_0     N_1     N_2     N_3     alpha_3
    G6   0.263   0.893/0.893   1.274   0.901   0.673   0.534   -0.035    margin MET, band 0.45-0.65 hit (exact-chain input: f_J of A_6)
    Gchi 2.198   0.347/0.894   2.281   2.009   1.851   1.736    0        KILL fired (>= 0.90): f_J gate from the stored amplitude closed
  CAVEAT on G6 (amplitude lead, same day): G6's f_J is taken on A_6 of the exact chain, which was built under the moving exact sign T, so
  its wall shape already carries T's information; G6 passing is an upper reference, not evidence that a gate on a stored amplitude can pass.
  Only stored-input arms (Gchi, R1, R1r) count for the objective.
  Diagnosis of the kill: the cut c = 0.0088 was set on the exact chain's f_J, whose right-row spread is 0.0038 (unweighted std); chi_6's round has
  0.0158, so the cut sits inside the stored amplitude's own copy noise and flips 1.43M rows (precision 0.35).
  Same stored round, rule R1 (flip iff VA > SK): 359,588 flips, P/R 0.991/0.927 (psi0^2), so remaining wrong weight ~0.10 [derived from P/R, to be
  measured]; R1 flips lie 94% inside the gate's, the gate's flips only 34% (weight) inside R1. R1 does not use the f_J magnitude, so the copy
  noise that killed Gchi does not enter it.
  N_3 against w over all fixed-sign arms from chi_6: T 0.116 -> 0.373, G6 0.263 -> 0.534, S 0.574 -> 0.901, s2 1.227 -> 0.958, Gchi 2.198 -> 1.736.

DOES THE COPY LOSS MEASURED UNDER s2 TRANSFER TO S_R1? (amplitude lead 2026-11-27, paper, while arm C runs)
  [exact, ten Haaf] At fixed sign S the guide amplitude enters E_FN only through V_sf(x) = sum_{y: S_x H_xy S_y > 0} |H_xy| a_y/a_x (violated bonds of S).
  [approx, 2nd order in d, l.1202] chi = T e^d: L = N(chi) - N(T) ~ sum_{viol(S)} |H_xy| phi_x phi_y 4 sinh^2((d_x - d_y)/2) / sum phi^2 (phi the FN state).
  So L depends on the sign only through which bonds are violated. S_R1 differs from s2 on 359,588 of ~1.4e7 orbit rows (2.6%), so the violated set changes only on
  bonds touching those rows; on the rest L is the s2 value. [cond: d not concentrated on the flipped rows] -> the s2-measured L 0.076-0.091 transfers to arm C
  up to the share of sum |H| phi phi (Dd)^2 carried by bonds touching flipped rows (not computed; the flipped rows carry psi0^2 weight 1.15e-6 of 1).
  What does not transfer: L at lower targets. Records show L grows at fixed fit MSE (cheap net 0.21 -> 0.43) because the MSE norm (n T^2) and the cost norm
  (|H| phi_x phi_y on violated bonds) weight rows differently; the ratio of the two norms on the fit's residual is the term arm C measures through L_1, L_2.
  Consequence for the prediction: unchanged (0.52-0.66); a miss with L_k rising round by round would point at the norm mismatch, the fix being a loss weighted
  by the violated-bond cost instead of n T^2.

RIGHT CHAMBER OR STUCK?, RESULT, AND THE SANDWICH CHECK (amplitude lead 2026-11-27; job 18244623; runs/j1j2/consol_6x6/chamber_check/result.json md5 aa840e7e, round 0 re-read)
  Prediction FAILED (50%): over rounds 0 -> 3 variance and flag weight fall in every arm, so neither separates the near-right signs (A, G6) from the fixed wrong ones.
  var3/var0, flag3/flag0: A 2.71, 0.128 | G6 0.225, 0.052 | B 0.339, 0.015 | S 0.235, 0.013 | Gchi 0.193, 0.006. (Predicted A, G6 var <= 0.5; B, S, Gchi var >= 0.7.)
  Round 3 (var 1e-4, flag 1e-9, dE_var 1e-6/site): A 5.61, 1.5, 0.843 | G6 1.11, 24.8, 0.682 | B 1.16, 3.1, 1.013 | S 1.18, 5.1, 0.959 | Gchi 5.14, 18.7, 1.819.
  B and S rounds 4-6: var flat at 1.16-1.17e-4 (B) and 1.16-1.13e-4 (S), flag flat 3.5-4.9e-9, while dE_var still falls (B 0.95 -> 0.85).
  Diagnosis [cond]: the first rounds are dominated by AMPLITUDE error, which falls in every arm whatever the sign; the sign error shows only once the amplitude
  has converged, as the plateau B and S reach by round 3-6. Arm A's rising variance with falling E_FN (E_var flat 0.843 from round 2 to 3) says the
  amplitude moves toward the FN state of its sign, whose variance under full H is not small; so var(H) of the guide is not a chamber indicator in 3 rounds.
  Usable form [conj, 55%]: a plateau of var and flag over >= 3 rounds at fixed sign marks the floor of that sign; a sign change that lowers the plateau
  is progress. No right-sign arm has been run to round 6 to show its plateau is lower (that is the missing control).
  Known cases: exact |psi0| with exact sign var 4e-28, flag 0; |psi0| with s2 var 8.1e-4, flag 1.2e-6 (> 0 as required; the worker's own 1e-3 threshold missed).
  SANDWICH (registered 70%, 'WHY R1 SHOULD LOWER THE ENERGY'), reference B: g_B = E_var - E_FN = 1.995 - 1.761 = 0.234 (1e-6/site, round 0).
    -dE_var vs B: A +0.330, R1 +0.331 (> g: provably below B) | S -0.210, G6 -0.228, Gchi -4.02 (not provable).
    Round-0 E_FN: A 1.421, R1 1.419 below B 1.761 as provable; S 1.895, G6 1.847, Gchi 4.310 above B. Round 3: A 0.373, R1 0.368 below B 0.958. PASS.
    But S (0.901) and G6 (0.534) also end below B at round 3 although above it at round 0: the bound is sufficient, not necessary, and round-0 order
    does not fix round-3 order.

R1 SIGN FROM THE STORED AMPLITUDE IN THE CYCLE, REGISTERED (amplitude lead 2026-11-27, before the run; objective from FN-lead 11-27)
  Sign S_R1 = s2 (1 - 2 [VA/SK > 1]), VA, SK from round_map's fJ_and_VsK on lG(chi_6, alpha -0.0517, E' of cycle1 armB round 1) under s2: the
  same chain as Gchi (build_grade.py chain('Gchi')), stored pieces only (chi_6 + s2), SK = 0 repair as in build_grade.py.
  Arm R1: fixed S_R1, start chi_6, three rounds (cycle3.py). Arm R1r: at each round k, S_k = s2 flipped by R1 evaluated on that round's own
  start table A_k under s2 (its own line search for alpha and E'), then the round under S_k.
  Paper: on the w -> N_3 points above, w 0.10 sits next to T (0.116); the deciding term is whether the Lanczos step survives (alpha_3 != 0).
  Data: R1 P/R 0.991/0.927 on chi_6's round (build/result_build.json).
  Predictions (mine): R1 N_3 0.38-0.52 (55%); w(S_R1) 0.08-0.13; alpha_3 nonzero. R1r N_3 <= R1's (60%). FN-lead's: 0.42-0.55 (50%).
  Margin <= 0.665; kill >= 0.90. Between: report w and alpha per round, diagnose which term carries the miss.

R1 SIGN FROM THE STORED AMPLITUDE IN THE CYCLE, RESULT (amplitude lead 2026-11-27; arm R1, job 18244986, build 18244985)
  N_3 = 0.368e-6 per site [exact FN, deterministic]: margin <= 0.665 MET, kill not fired, below both predicted bands (mine 0.38-0.52, FN-lead 0.42-0.55).
  Per round N (1e-6/site): 0.946 / 0.594 / 0.448 / 0.368; alpha -0.122 / -0.125 / -0.053 / -0.042 (Lanczos step survives, alpha_3 != 0).
  w(S_R1) 0.0998e-6 (s2 1.227e-6; T 0.116e-6), 359,588 flips = build_grade's Gchi R1 record; P/R (psi0^2) 0.991/0.927.
  Reference arms: exact-T arm A 0.373, G6 0.534, s2 (B) 0.958. R1 sits at T's value, consistent with the w -> N_3 points (R1 w below T's).
  Leak check (code read, build_r1.py l.90-108): mask VA/SK from lanc(chi_6, alpha_start, E') under s2 with alpha, E' from cycle1 armB map1;
  psi0 enters only the grades (w, P/R). Known cases pass (R0 under s2 reproduces arm B round 0, rel 6e-10; broken versions fail).
  Round 0 sandwich numbers (1e-6/site): E_var[S_R1 chi_6] 1.664, E_var[s2 chi_6] 1.995 (Delta E_var = -0.331), E_FN[S_R1, alpha 0] 1.419.
  Scope: the sign is fixed (flip once) and computed by a local rule from the stored amplitude; amplitude rounds 1-3 are exact round_map
  tables, as in every cycle3 arm the margin was set on. Not yet covered: amplitudes copied into the network each round; 8x8.
  Source: /project/theorie/a/A.Otaifi/runs/j1j2/consol_6x6/cycle5_r1/armR1/result_arm.json (md5 55732b2f). R1r (per-round regate) pending.

R1r (RE-FLIP EACH ROUND), RESULT (amplitude lead 2026-11-27; job 18244987; cycle5_r1/armR1r/result_arm.json md5 37915cb4, re-read; result.json md5 df8e0ab4)
  N per round (1e-6/site): 0.946 / 0.594 / 0.441 / 0.359 (R1 fixed: 0.946 / 0.594 / 0.448 / 0.368). Margin met; predicted <= R1's (60%): HIT.
  The sign each round used: wrong weight 9.98e-8 / 9.98e-8 / 6.01e-8 / 2.95e-8; P/R 0.991/0.927 -> 0.994/0.957 -> 0.997/0.979; rows changed 44,844 then 83,155.
  Delta E_var of the flip against s2 on the same amplitude: -0.331, -0.660, -0.870 (1e-6/site): the flip gains more as the amplitude improves.
  Reading: re-flipping from the improved amplitude keeps finding wrong rows (wrong weight falls 3.4x in two rounds), and the energy gain is small (0.009 at round 3)
  because the energy is by then set by the amplitude, not the sign. Supports regating in the stored loop (arm Cr), with exact amplitudes; under copy noise not shown.

THE LOOP WITH NOTHING EXACT INSIDE, REGISTERED AND PAPER CHECK (amplitude lead 2026-11-27, before any run; objective from FN-lead 11-27)
  Objective (FN-lead): 6x6, R1 sign from the stored amplitude, each round's amplitude copied into the network before the next round; N_3 <= 0.665e-6
  (graded exactly); kill >= 0.90e-6; log per round the R1-flag count and A_k^2 weight. FN-lead's prediction 0.50-0.70 (45% inside).
  Arm C (now): fixed S_R1 (cycle5_r1 build, md5 ecb4b8b5), start chi_6 = NNx/x1 t_R. Round k: exact round_map under S_R1 on the STORED chi_k -> target T_{k+1};
  copy = the step-input network of NNx (nn_fit.py, width 64, 80k steps, cosine; inputs z + c1[chi_k] with c1 = ln|1+alpha(E_L-E)| recomputed on chi_k
  UNDER S_R1); chi_{k+1} = the network table; N_k graded exactly under S_R1 (cycle5 grader). Arm Cr (registered after R1r, 2026-11-27): branches from arm C after fit0 (S_0 = S_R1 bit for bit, so round 0 is shared); for k = 1, 2: S_k = s2 flipped by R1 on the STORED chi_k under s2 (cycle5r.py rule), then the round and the copy as in C. Prediction: N_3(Cr) = N_3(C) - 0.01 +- 0.02 (Cr <= C, 60%); risk: copy error misflips (A27).
  Derivation [approx, model fitted to the records]: stored N_{k+1} = M(N_k) + L_k, with M the exact round and L_k the copy loss (STORAGE_PROBLEM 'DESCENT ROUTE
  ROUND 7': stored N = N(target) + L). M from arm R1's exact chain 0.946/0.594/0.448/0.368: N_{k+1} - N* = c (N_k - N*), c = 0.548, N* = 0.271
  (rounds 1-3; round 0->1 contracts faster, c 0.41, used for round 1: M(0.946) = 0.594 and M(N) = 0.594 + 0.41 (N - 0.946) near it).
  Measured L for the step-input copy (under s2): round 6 0.076 (x1) / 0.091 (x2), round 7 0.083 (X7, shadow input). Cheap z-only copy: 0.21-0.43 (floors near 1.07).
  N_3(L constant): L 0.08 -> 0.674 / 0.572 / 0.516; L 0.12 -> 0.714 / 0.634 / 0.590; L 0.16 -> 0.754 / 0.696 / 0.664; L 0.35 (cheap copy) -> 0.944 / 0.990 / 1.015 (kill).
  Deciding term: L at targets 0.37-0.6e-6, never measured (lowest copied target so far 1.096e-6). Margin is crossed at L ~ 0.16.
  Checked: L = 0 returns the exact chain; c, N* reproduce rounds 2-3 exactly by construction (two equations, two unknowns), so the model's only test is round 1.
  Prior results: copy error does not compound when it is constant (STORAGE_PROBLEM l.949, 1261); L grows as the target improves for a fixed-capacity copy (cheap
  net 0.21 -> 0.43); A27 (greedy flips fail under sigma 0.3 amplitude noise) bears on Cr, not on C (sign fixed).
  Known case for the run (worker builds): round 1 with the copy replaced by the exact target reproduces arm R1 round 1 N 0.5943e-6; broken: c1 computed under s2
  instead of S_R1 must change the input and the fit; S_R1 file md5 and w as in cycle5.
  Next rung: pass N_3 <= 0.665; kill >= 0.90; between: report L_k per round and whether L grows as the target falls (the representation limit) or stays flat.
  Prediction (mine): N_3 0.52-0.66, 60% inside the margin; L_1 0.08-0.12. Risk as FN-lead: the copy, not the sign. Cost ~3 x 6 GPU-h serial + grades.
  Status: DONE for the registered prediction; the model is [approx] and its one untested input is L at low targets, which is what the run measures.

RIGHT CHAMBER OR STUCK? A TEST THAT NEEDS NO EXACT STATE (amplitude lead 2026-11-27, idea from discussion with the PI; [conj] until checked)
  [exact] The FN ground state of sign S equals the true ground state iff S is the ground state's sign; then sigma^2_H = <H^2> - <H>^2 = 0
  (full H, not H_FN). In any wrong chamber the FN state is not an eigenstate of H, so sigma^2_H > 0.
  [exact] At the true ground state B(x) = E0 - H_xx < 0 for all x (6x6: min H_xx ~ -9 from Neel/stripe, E0 = -18.14), so R1 flips nothing.
  [conj, 60%] Along the loop: right chamber -> sigma^2_H falls toward 0 and f_J -> 0 on every row; stuck in a wrong chamber -> sigma^2_H stalls
  > 0 and a fixed set of rows keeps f_J << 0 round after round (Kairon-Clark collapse onto the wall). Both are measurable at 8x8 by sampling.
  Not in the records before today (grep PLAN.md/STORAGE_PROBLEM.md 'variance', 'chamber'). Check on 6x6 from data on disk: sigma^2_H of the
  FN state per round for arms T (right-ish), S and B (fixed wrong signs); prediction: T's falls, S and B stall.
  FN-lead amendments (11-27, message): (1) variance -> 0 proves an eigenstate, not the ground state; the certificate is variance -> 0 with the
  energy monotone and at the lowest value seen. (2) At 8x8 the measurable quantity is the VMC energy variance of the stored guide S_k A_k
  (E vs variance extrapolation axis, as in the NQS papers); a stuck loop shows as points stopping at finite variance. (3) Second indicator:
  the guide-weighted share of R1-flagged rows (VA > SK) per round: -> 0 in the right chamber, stalls on the same rows when stuck (necessary,
  not sufficient). FN-lead: no proof of reaching the right chamber; 65% at 6x6 from s2.
  CHECK (rung 2, existing tables, registered before computing): per round, VMC variance of H and R1-flag weight (both A_k^2-weighted, psi0-free)
  for arms A (sign T), G6, B (s2), S (S_avg2, rounds 0-6), Gchi. Prediction (mine, 50%): A and G6 variance falls >= 2x from round 0 to 3 and
  flag weight falls >= 5x; B, S, Gchi variance falls < 30% over the same rounds and their flag weight stalls (< 2x). Between: report per arm.

WHY R1 SHOULD LOWER THE ENERGY, PAPER CHECK (amplitude lead 2026-11-27, written after cycle5_r1 was briefed; the brief's 'Paper check DONE'
line overstated it: the registered block held only the w -> N_3 trend, no derivation)
Derivation [exact]: psi = S a, h_xy = H_xy S_x S_y a_x a_y, B(x) = sum_y h_xy / a_x^2 (so E_L(x) = H_xx + B(x)). Flip a set F at fixed a:
a bond changes sign iff it is cut by F. Hence
    Delta E_var * sum a^2 = -4 [ sum_{x in F} a_x^2 B(x) - 2 sum_{bonds xy inside F} h_xy ]                      (R1a)
  One flip: Delta E_var = -4 a_x^2 B(x) / sum a^2 < 0 iff B(x) > 0. So R1 = every single flip that lowers the variational energy at
  fixed amplitude, applied all at once; the second term is the error of applying them at once (bonds between two flipped rows).
Checked: limits: F empty -> 0; F = all rows -> both terms cancel (global sign is gauge) [checked: sum_x a^2 B = 2 sum_bonds h]. 
Not covered [the deciding gap]: E_FN, not E_var, is the grade. <H_FN>[S a] = <H>[S a] + sum_viol |H_ij| (a_i - a_j)^2, and the penalty
  changes with F too (bonds cut by F switch between violated and kept). R1 guarantees only the first term. The penalty change is
  sum over cut bonds of +-|H|(a_i - a_j)^2: small where a_i ~ a_j, large across steep amplitude walls. No bound derived yet.
Prior results: the Kairon-Clark picture (arXiv:2609.16308, abstract only): the FN guide collapses on chamber walls outside the right
  chamber; R1 reads the same wall locally. Name collision: PLAN.md's 'R1' (11-09, FN-refit complex PP at fixed sgn M1) is a different
  object. TRIED_IDEAS A27 (10-18, 20 sites, results/scalable_sign_2026-10-18/README.md): greedy orbit sign flips recover 94.4% of the
  gain with exact amplitudes, but with log-ratio noise sigma 0.3 11/12 trials got worse (needs ~2-7% log-ratio precision). Conditions
  differ (20 sites, synthetic noise); on chi_6 R1's measured precision is 0.991, so the copy is inside A27's working range on the
  rows R1 reads. A27 is the closest record and predicts the risk: R1r, which re-reads each round's noisier copy, is the arm it bears on.
Penalty term, STEP 2 (same day, while cycle5_r1 runs) [cond: the account's identity <H_FN>[S a] = <H>[S a] + sum_viol |H_ij|(a_i - a_j)^2, per
  sum a^2, holds for the FN form the grader uses]: a bond cut by F switches violated <-> kept. Per bond, the <H> change (-+4|H| a_i a_j) and the
  penalty change (-+|H|(a_i - a_j)^2) add to -+|H|(a_i + a_j)^2. Hence, exactly under that identity,
    Delta <H_FN> * sum a^2 = - sum_{cut bonds} s_b |H_b| (a_i + a_j)^2,   s_b = +1 if the bond was violated, -1 if it was kept        (R1b)
  Single flip of x: <H_FN> falls iff sum_{viol} |H_xy|(a_x + a_y)^2 > sum_{kept} |H_xy|(a_x + a_y)^2  (call it R1_FN; the D2 weight).
  R1 compares sum |H_xy| a_x a_y instead. They agree where a_y ~ a_x and differ where x sits on a steep wall: with a_x << a_y both weight
  by a_y but R1_FN by a_y^2, so R1_FN leans harder on the large neighbours. Bound on the disagreement: not derived.
  CONFLICT TO SETTLE: W1 (alpha slope) assumed the ten Haaf form with guide ratios |H| a_j/a_i on the diagonal; under that form, with state =
  guide, <H_FN> = <H> exactly and the (a_i - a_j)^2 identity does not hold. The identity holds for the ratio-free sign-flip term |H_xy|, or for
  ten Haaf with state != guide. Which form round_map.py / the FN grader uses decides whether R1 or R1_FN is the right single-flip rule.
  SETTLED by code read (ansatz/code/delta0.py efn_vec l.21: Dg = D0 + (Vm @ a)/a): the grader is ten Haaf with guide ratios. So W1's form
  is right and R1b is VOID for our grade (it needs the ratio-free form). What holds instead [exact]: E_FN[S', a] <= <H_FN[S',a]>_{S'a} =
  E_var[S' a] = E_var[S a] + Delta E_var (R1a). R1 lowers the variational upper bound on the new FN energy, not E_FN itself.
FN-lead's SANDWICH (11-27, message; checked by me, each step [exact] for ten Haaf with guide S a): E_FN[S,a] <= <a|H_FN|a> = E_var[S a];
  slack g = E_var[S a] - E_FN[S,a] >= 0 (= 0 at the amplitude fixed point a = phi). For a flip set F:
    E_FN[S',a] <= E_var[S'a] = E_FN[S,a] + g + Delta E_var   =>   E_FN falls provably if  -Delta E_var > g                 (SW)
  At the fixed point (g = 0) single-row R1 is exactly the provable rule. Off it, flips are certified only if their E_var drop beats g.
  Consequences (FN-lead): loop "relax to f_J ~ 0, flip a set with Delta E_var < 0" is monotone in E_FN; the bound fixes direction, not
  size (a fully collapsed row has a_x^2 B -> 0, gain comes in later rounds), so flip after partial collapse. Candidate reading of Gchi:
  its flips had Delta E_var > 0, so nothing protected it.
  CHECK (rung 2, existing data, registered): round-0 Delta E_var (vs s2, on chi_6) and g for arms A, B, S, G6, Gchi from the chamber-check
  E_var values. Prediction (FN-lead, I agree 70%): Gchi has Delta E_var > 0; every arm with -Delta E_var > g ended below B at round 3.
Known case + broken version: NOT DONE (a 4x4 exact check: R1 from a wrong sign, Delta E_var from (R1a) vs direct, and the same with B's
  sign reversed; CPU minutes).
Status: NOT DONE: known case, penalty bound.

R1 KNOWN CASE, 4x4 (amplitude lead 2026-11-27; ansatz/code/r1_4x4.py sha256 7768c4c25f3e..., ED, CPU 1 min; fills the known-case block of 'WHY R1 SHOULD LOWER THE ENERGY')
  4x4 J1-J2, J2 = 0.5, PBC, Sz = 0 (D 12870), E0/N -0.52862. Start: Marshall sign (wrong weight 0.0127), amplitude |psi0| e^(sigma xi). One R1 pass (flip iff B > 0),
  ten Haaf FN graded exactly. E_FN - E0 per site:
    sigma 0.0: Marshall 7.28e-3 -> R1 2.06e-3 (3.5x lower); broken (same number of flips on B < 0 rows) 1.34e-1.
    sigma 0.1: 7.80e-3 -> 3.25e-3; broken 1.44e-1.   sigma 0.3: 1.14e-2 -> 7.49e-3; broken 1.28e-1.
  Flip precision 0.86; wrong weight 0.0127 -> 0.0029 / 0.0037 / 0.0044. At exact psi0 with exact sign R1 flags 0 rows (fixed point, as derived).
  Reading: one R1 pass lowers E_FN at every noise level tried, and the broken rule raises it 18x; the gain shrinks with amplitude noise (3.5x -> 1.5x at sigma 0.3).
  Against A27 (greedy flips fail under sigma 0.3): A27 was iterated greedy flips; this is one pass, which still gains at 0.3. Iterated R1 under noise is not covered.
  Status of the R1 paper check: known case DONE; penalty bound still open.

WHY ALPHA DIES UNDER A FIXED SIGN, STEP 1 (amplitude lead 2026-11-27, paper, while cycle5_r1 runs)
  Lattice FN (ten Haaf et al. 1995): H_FN[S,a] keeps sign-consistent hops and moves each violated bond (ij) into the diagonal as
  |H_ij| a_j/a_i on row i. E_FN = lowest eigenvalue, ground state phi (normalised). The guide a enters only through those ratios, so by
  Hellmann-Feynman, for G = a|1 + alpha(E_L - E)| (rows where 1 + alpha(E_L - E) > 0):
    dE_FN/dalpha |_0 = sum_viol |H_ij| (E_L,j - E_L,i) [ phi_i^2 a_j/a_i - phi_j^2 a_i/a_j ]          (W1)
  [cond: nondegenerate FN ground state; E_FN in round_map.py is this ten Haaf form - to check against the code]
  Consequences: (i) the slope is a sum over violated bonds only, so with fewer wrong rows (T, G6) there are fewer terms but also less
  cancellation; (ii) at a guide whose violated-bond ratios equal phi's (a = phi there), every bracket vanishes and alpha* = 0 exactly:
  the alpha step has nothing left to do once the amplitude has relaxed to its sign's FN state on the wall. Under S and B, alpha* = 0 from
  round 2 says their walls relaxed within two rounds; the later 0.87/round fall is the f_J dressing, not the Lanczos step.
  Not yet derived: why the bracket stays nonzero under G6/T for three rounds (alpha_3 -0.035), i.e. how the slope scales with w.
  Next step on paper: evaluate W1's bracket on a two-row toy (one violated bond, phi from the 2x2 FN problem) as a function of the bond's
  share of psi0^2; the R1 arms' alpha_k (cycle5_r1) then test W1, they do not set it.

RATIO-INPUT SIGN COPY, PAPER CHECK (sign lead, 2026-11-26; FN-lead's sigma bar, docs/theory/2026-11-26_resting_energy.md)
Derivation:
 (1) [exact] The C-free label is y(x) = sgn sum_b rho_b(x) c_b s_old(z_b), rho_b = |M1(z_b)/M1(x)| (RBM+PP, pp2_lemma), c_b = +-J_b/2 (audit_existing.py lines 38-49 assert it on every root).
 (2) [exact] A spins-only net must rebuild rho_b; with ln rho_b as inputs, y is a sign of a bond sum of known weights times s_old(z_b).
 (3) [approx, deciding term for learnability] s_old(z_b) = sgn(l_old(x) + dl_b(x)); dl_b is a local change within the CNN's receptive field, l_old(x) one scalar. A per-bond MLP on
     (ln rho_b, c_b, local spins, l_old(x)) with log-domain pooling contains y up to the error in dl_b.
 (4) [exact, deciding term for cost] FN needs s(z_a) at every neighbour z_a of the walker x, so the inputs are rho_b(z_a) = B(z_ab)/B(z_a) = [B(z_ab)/B(x)] / [B(z_a)/B(x)]:
     all two-swap ratios from x, pp2_lemma.two_swap_ratios, instead of one_swap_ratios. The cost is C/C0 = 1 + (t2 - t1)/t_total, unmeasured; z^2 = 128^2 lemma terms per image vs 128.
Checked: (1) reproduces labels exactly (audit_existing assert); (4) the ratio identity is exact for any nonzero B. The same input gap was measured on the amplitude side: spins-only ViT retention
 0.056 vs ratio-input ratnet 0.73-0.89 (ARM H, FIX D). Theorem: none needed for (1), (2), (4); (3) is a modelling claim. Confidence the derivation is right: 85%; that the class can reach q >= 0.82: 40%.
Prior results: T-GROW (width, head, weights: < 2% cut, train err = held-out err ~0.10), T-CAP8 (depth: 2.2(14)e-5), A48 (12 trained classes killed), tfield (regression q 0.27), STEP 7/18
 (exact neighbour signs via gating: 2.3x for 63% of the stake). None gave the net M1's ratios. sigma_audit/train_vs_held.json: CF misses 45% of its own training flips.
Known case: y == sgn sum_b rho_b c_b s_old(z_b) on all 10,240 roots (audit_existing assert). Broken version: rho from a same-decade partner row (ratnet control, retention 0.008 on 6x6).
 The sign-side broken run is part of the data check.
Not covered: the cost (4) is not timed; whether dl_b is local enough at 8x8; FN realization of a lower panel R.
Next rung (cheapest first, it decides the route): time two_swap_ratios against one_swap_ratios per walker state at 8x8 on an a40-24gb slice (~0.1 A40-h).
 Pass: projected C/C0 <= 2 -> data check (rho inputs for the panel + a training subsample, fit, panel R). Kill: C/C0 > 4 -> ratio inputs are not a constant-factor route at 8x8; the copy
 then needs a cheaper input (ln|B(z)| itself is free) or the gate. Prediction: C/C0 3-10 (60%), because the lemma's per-pair work scales with z^2.
Status: DONE (derivation, check, prior results, known case written; the broken case is run with the data check).
RATIO-INPUT SIGN COPY, COST DERIVATION (sign lead, 2026-11-26, after FN-lead's challenge: "what does a ratio-input copy buy over the exact step at the same inputs?")
 Measured anchors (STEP 18 probe job 18213822 and count job 18214912, same node class): cf baseline t0 = 14.26 s/block over ~703 walker states/block, i.e. 20 ms per state;
 split amp 3.2 s, CF net 5.7 s, rest 5.4 s per block. Exact gate, per gated neighbour z: 0.91 t0-units = amp call at z ~0.20 + CF at z's 128 neighbours ~0.38 + overhead ~0.33
 (amp 4.2 s, net 8.0 s extra per block at n_g 1.51). One amp call at z costs about one walker-state amp call (t1 ~4 ms).
 Ungated ratio-input sign: rho_c(z_a) at all ~144 neighbours. Through one_swap at each z_a: C/C0 ~ 1 + 144 x 0.20 ~ 30. Through two_swap from x: C/C0 ~ 1 + 0.22 (t2/t1 - 1);
  <= 2 needs t2/t1 <= 5.5 (timing job running; z^2 lemma terms predict far more). [derived]
 Gated ratio-input sign (share n_g): C/C0 ~ 1 + (0.20 + head + part of overhead) n_g ~ 1 + 0.3-0.5 n_g, against the exact gate's 1 + 0.91 n_g. At (2,-5): 1.4-1.7x lossy vs 2.31x exact;
  at (2,-6): 1.6-2.0x vs 2.87x. [derived from the split; head cost unmeasured]
 ANSWER to FN-lead (1): for one step from a known sign the net is a lossy copy of an exact formula at the same ratio cost. It buys only the CF calls at two-hop states, ~40-55% of the
  gate's extra cost, so it is worth building only gated, and only if its gated q stays >= ~0.8 (otherwise the exact gate at a little more cost wins). Rounds >= 2 are the other case.
 (3) The case for it: if the stored amplitude becomes a ratio-input head, rho is paid at every requested state, the guide's own signed ratios give the round-1 exact sign at zero marginal
  cost, and no sign copy is needed for round 1. That decision sits with the amplitude side's input set (RATIO-INPUT add-on, LOG-DOMAIN POOLING); the two-swap timing is the shared cost number.
 Plan change: the near-term route to sigma/3 is the gate family; STEP 18b (exact gate, FN realization) decides it. The ratio-input sign is held until (a) the two-swap timing and (b) the
  amplitude side's choice of input; it is built only as a gated head or as part of a joint signed ratio-input network.

PER-BOND TARGETS FOR THE SIGN COPY, PAPER CHECK (sign lead, 2026-11-26; FN-lead's finding, arXiv:1703.07950 Sec. 2-3 read by the FN-lead's scout, not by me)
Derivation:
 (1) [exact] y(x) = sgn sum_b t_b(x), t_b = c_b |rho_b| s_old(z_b) (audit_existing.py lines 38-49), every t_b known per row where |rho_b| and s_old(z_b) are stored (roots: rabsr, nrel).
 (2) [exact] Class: the CF trunk (per-site 64-channel features, 16-image average) with a bond head tau_b = g(f_i + f_j, |f_i - f_j|) on active bonds, logit l = sum_b tau_b. Forward cost:
     one head per bond (128) instead of per site (64), same trunk, so ~1x CF [derived].
 (3) [conjectured, the deciding term] Loss L = BCE(y, l) + lambda sum_b (tau_b - K t_b/Lambda)^2. Each tau_b gets its own gradient. The cited claim: for sgn of a composite, the end-to-end gradient
     carries vanishing information about the parts; part-wise losses restore it. Our signature fits it: training flips missed 45% (sigma_audit), width/depth/weights null (T-GROW, T-CAP8),
     and CF's features lack the parts exactly on flip rows (STEP 19: R^2 <= 0.17, sign factor negative).
 (4) [risk] |rho_b| is an amplitude ratio; spins-only nets stored ratios badly on the amplitude side (retention 0.056). Here CF features already carry |W_b|/Lambda at R^2 0.37 linearly, and only the
     sign of the sum must be right.
Checked: (1) exact (audit assert on all roots); (2) cost by construction; (3) is the paper's claim for products of sign parts (k >= 3 fail end-to-end), our parts are real-valued and summed, so it is an
 analogy, not a theorem for our case. Confidence the derivation is right: 70%; that it cuts train flip miss below 0.25: 40%.
Prior results: B23 (bond features as INPUTS, 6x6, fails); T-BOND (per-bond surrogate of eps^2, rejected on paper, different target); STEP 10 tfield (total field, q 0.27); none fits the parts as targets.
Known case: the parts sum to the label: sgn sum_b t_b == y on all 10,240 roots (audit_existing assert). Broken version for the run: per-bond targets taken from a same-decade partner root; it must
 not beat the BCE-only arm.
Not covered: neighbour rows lack stored |rho_b| and s_old(z_b) (rabsr is roots only); the full refit needs them built (one_swap at each row + n_2 at its neighbours, ~6 A40-h for all 1.47M rows).
Next rung (existing data, roots only): fine-tune from CF on the 8192 training roots, arm P (per-bond loss + BCE) vs arm B (BCE only, same rows, steps, seed) vs arm X (P with partner-root targets).
 Measure the TRAIN-root flip miss (the underfit), held-out-root parts R^2 on flip rows, and held-out root keeps. Pass: P's train flip miss <= 0.25 with B >= 0.40 and keeps >= 0.98; kill: P within
 0.05 of B. Between: report. Prediction: 40% pass. Cost <= 1 A40-h.
Status: DONE.

GATE WITH sgn(M1) FALLBACK, STEP FROM THE ADAPTED AMPLITUDE, PAPER CHECK (sign lead, 2026-11-27; FN-lead hypothesis, docs/theory/2026-11-26_resting_energy.md step back)
Derivation: guide s_B(x) = y_A(x) on gated rows, sgn M1(x) elsewhere, y_A(x) = sgn sum_b c_b |A(z_b)/A(x)| sgn M1(z_b) (one exact power step from sgn M1 with amplitude A) [exact
 definition]. FN energy with |M1| amplitude: by eq (9) (exact at 6x6, uses H psi0 = E0 psi0), the bar to fix a wrong row is B_x > theta r/(1+r) and to break a correct row C_x > theta/(1+r):
 an amplitude suppressed on wrong rows (r < 1) fixes more and breaks fewer. Transfer to 8x8 with A = R1 (FN-refit complex PP at fixed sgn M1) as the suppressed amplitude [conjectured:
 R1's suppression on wrong rows is unmeasured]. Deciding term: sign of sum over gated rows of (fixed - broken) energy mass, which depends on A only through rho_b.
Checked: limits: gate empty -> s_B = sgn M1, Delta = 0 exactly; A = psi0 -> exact step. 6x6 FN r-scan (gammascan/result.json md5 762926d4, FN-lead): T(r=0.03) from Marshall 2.76e-4 vs
 5.74e-4 (A_2 arm): amplitude dependence of one step under FN is a factor ~2 [cited, not re-run by me]. Confidence the design isolates the amplitude term: 80%; that R1 is wall-suppressed: 50%.
Prior results: T-NODE (S_-7: exact step at decade <= -7, fallback sgn M1, FN on |M1|, step from |M1|): +8.0(4.1)e-6/site. G(-5) (fallback s306, decades <= -5, |M1|): +6.4(3.5)e-6.
 Both: step from |M1|, amplitude-decade gate (not CF-confidence). No record with the step from an FN-adapted amplitude at 8x8 (recall j1j2 G(-5) T-NODE; PLAN_sign grep R1/S_-7).
Known case: 6x6 r-scan above. Broken version for the run: same count of flips on random gated rows (must raise or not lower E_FN).
Next rung: branched FN pairs vs sgn(M1) (18b grader), gate (2,-5) rows. Arms B_R1 (step from R1), B_M1 (step from |M1|), B_R1 broken. Pass: B_R1 Delta <= -1.0e-5/site; kill: B_R1 Delta >= 0
 within its error (then send flip precision on gated rows vs the first-order sign). Between: report. Prediction (mine): B_R1 pass 30%, B_M1 Delta in [-0.5, +1.0]e-5 65%. Cost ~7 A40-h per arm.
Status: DONE (transfer of eq 9 to R1 conjectured, labelled).

WHY HALVING w KEPT ONLY 10% OF THE CYCLE GAIN: AN ERROR'S COST IS ITS WALL, NOT ITS psi0^2 (sign lead, 2026-11-27, written before the amp lead's x(avg2))
Data: STEP 21 avg2 has w 5.74e-7 against s2's 1.23e-6, yet N_3 is 0.901 against 0.958 (exact T 0.373). It keeps (0.958-0.901)/(0.958-0.373) = 10% of the gain (cycle2_ssign/result.json, re-read).
Derivation (a = |psi0|, s0 = sgn psi0, S a stored sign, sigma_xy = S_x S_y s0_x s0_y, N = sum a^2):
 (1) [exact] E_var[S a] - E0 = -(2/N) sum_{x<y, sigma_xy = -1} 2 H_xy psi0_x psi0_y. Only bonds whose relative sign is wrong cost energy (the wall), and a coherently flipped domain costs only its boundary.
 (2) [exact, from H psi0 = E0 psi0] An isolated wrong row x costs (4/N) a_x^2 (H_xx - E0). This is its full psi0^2 times a large diagonal factor.
     A domain D costs -(4/N) sum_{x in D, y not in D} H_xy psi0_x psi0_y, its wall only. Per unit psi0^2 that is smaller by about the wall/volume ratio.
 (3) [cond: the FN cost per wrong row follows its var cost up to an O(1) factor; for an isolated row FN drives psi(x) to ~0, so its cost is ~ a_x^2 (H_xx - E0)]
     The w metric weights every wrong row by a_x^2 and so ignores the wall/volume factor.
     s2 and T are smooth functions of the configuration, so their errors are domains. A per-row CNN copy adds scattered isolated errors: 667k false flips and 279k missed flips, cut out of the T flip regions.
 (4) [conj, 70%] Per unit psi0^2, a CNN copy's errors cost more than s2's or T's. Hence x/w(avg2) > x/w(s2) = 1.49/1.23 = 1.21.
Prediction (recorded before the amp lead's x(avg2) arrives):
 - x(avg2) lies in 1.1-1.6e-6, against the 0.70e-6 that w-proportional cost would give (1.49 x 0.468). That is x/w about 2-2.8.
 - The x ordering of seed0, seed1 and avg2 follows their wall weight, not w.
 - The between reading, x(avg2) 0.8-1.1e-6: walls matter, but part of the 10% retention sits elsewhere (FN versus var, item 3).
 - Kill: x(avg2) <= 0.8e-6. Then w-proportional cost holds and the miss is not in the sign's energy.
Consequence for the copy, if (4) holds [derived from (1)]:
 - The training loss is a bond loss against the target, L = sum_bonds |H_xy| a_x a_y [S_x S_y != T_x T_y]. It is gauge-free, charges the wall, and makes a copy prefer coherent errors to scattered ones.
 - Its row form for a single error is the 8x8 regret weight 4p|h| (audit_existing.py grade()), which is exact only for isolated errors.
 - At 8x8 the guide's |M1| supplies a.
 - Free check from existing code: efn_vec already returns E_guide_var. So x_var(S) = E_guide_var - E0 for T, s2, avg2, seed0 and seed1 at the exact amplitude tests (1)-(4) on paper data, at no training cost.

 RESULT (sign lead, 2026-11-27; xpsi0 job 18239675, runs/j1j2/consol_6x6/xpsi0/result.json, re-read): KILL FIRES. x(avg2) = 7.72e-7 (band 1.1-1.6, kill <= 0.8).
  x/w: avg2 1.345, s2 1.218, so avg2 is 10% more costly per unit w, not the 1.5x my band needed. Diagnosis: the bond identity stands, but the premise that the CNN copy scatters its errors fails.
  D3/w is 1470 (avg2) vs 1600 (s2), so avg2's wrong rows have as much neighbour weight per unit w as s2's domains.
  The cycle gap (N_3 0.901 against 0.64 predicted from x on the T-s2 line) is not a static sign cost. The amp lead is testing whether it is the round map's speed (cycle4_extend).
  Consequence: no bond/wall loss. The copy loss is weighted by e*_x (FN-lead's D1 predicts x - x_T to within 10%, ratio 0.83 for avg2/seed0/seed1, 0.88 for s2).
WALKER-LAW KEEPS PLUS FLIP ROWS (the PI's "rows from where the walkers are, plus the flip rows"; FN-lead STEP 24), PAPER CHECK (sign lead, 2026-11-27)
Derivation:
 (1) [exact] The deployed error of a flip classifier F_hat on s2 is w = sum_x p_x [s2 (1-2 F_hat) != s0]. Write it as w(T) + missed_p + false_p, as in grade.py's split, which is exact for a table that agrees with T or s2 on each row.
 (2) [exact] An unweighted BCE on rows drawn from a law q minimises E_q[loss]. With q = p (Born keeps), a keep row's share of the loss equals its share of w's false term, so the training law matches the deployment weight.
     STEP 22's law (decade-capped keeps, none in -9..-4) has q/p -> 0 on the heaviest rows. That is the measured failure: deployed w 0.02-0.4 while held false p was ~0 at tau 2.
 (3) [approx] Flip rows drawn uniformly from the pool over-weight the light decades relative to p. The threshold is therefore fixed on the training rows by the p-weighted error, sum p[miss] + sum p[false], which corrects the class-prior shift (logit offset ln(q_flip/p_flip)).
Checked: (1) reproduces grade.py on STEP 21 avg2 (missed 8.8e-8 + false 4.23e-7 + w(T)-overlap = 5.74e-7, result.json). (2) is the definition of an expected loss. STEP 21 is the known case: T labels on teacher-law rows (beta 0.5, which includes heavy rows) gave false p 4.2e-7, against STEP 22's 0.02-0.4 on capped rows. Confidence 85%.
Prior results: STEP 21 (signcopy/result.json); STEP 22 (flipcurve/curve.md, thr_sweep.py); tfield/T-GROW (8x8, CF misses 45% of its own training flips: a different class and law).
Known case plus broken version: the oracle F_hat = (T != s2) gives w = w(T); F_hat = 0 gives w(s2); shuffled flip labels are the broken arm and must give w >= w(s2) or q_held <= 0.
Not covered: w is not the energy measure (cycle2_ssign: avg2 rested at 0.901 against s2's 0.958). This step tests learnability under the right law, and the energy weight comes later from the amplitude lead's xpsi0.
Next rung: arms n_flip 10k and 100k, each real plus shuffled. Keeps are drawn from p over all decades (T == s2, held rows excluded), 20 per flip row; 40k steps; same held set.
 Pass: deployed w < 1.227e-6 at the training-fixed threshold. Kill: w >= w(s2) at both n with held-by-weight miss > 0.5. Between: anything else.
 Prediction (sign lead): pass at 100k 65%, at 10k 45%. The false flips set the outcome, and STEP 21's 4.2e-7 says the heavy rows can be held.
Status: DONE

WHERE sgn(M1)'S ERROR ENERGY SITS BY DECADE: WHY THREE DECADE-GATED ONE-STEP SIGNS GAINED NOTHING AT 8x8, PAPER CHECK (sign lead, 2026-11-27)
Derivation:
 (1) [exact at 6x6, measured law] The static FN cost of a sign is set by its wrong rows' e*: x - x_T ~ 0.83 D1, with D1 = sum of e*_x over the wrong rows (xpsi0, amp lead).
     So a relabel S -> S' gains ~ 0.83 [sum_fixed e*_x - sum_broken e*_x]. A gate G restricts both sums to G.
 (2) [exact] With G = {decade <= d}, gain(G) = sum over decades <= d of net(decade). If sgn(M1)'s wrong e*-mass sits at decades > d, then gain(G) ~ 0 whatever the step's precision.
 (3) [measured, 8x8] STEP 23: B_R1 flips 5.7% of gated neighbour states and gains +1.06(0.55)e-6. The same count of random flips loses 2.23(1.25)e-5.
     So R1's gated flips are not random but net-neutral, i.e. fixed ~ broken mass inside G. 18b gained 2.57e-5 in the same G against CF, so G can hold gain when the base sign errs there.
 (4) [conj, 65%] sgn(M1)'s error energy sits above decade -5. Deciding term: the per-decade net in (2).
Checked: G = all rows recovers the 6x6 ungated R1 result (N_3 0.368 = exact 0.373, cycle5_r1). G = empty gives 0 exactly. e* sign convention as xpsi0 (negative on wrong rows).
 Second route: the broken arm in (3) bounds the precision independently of the decade argument. Confidence that (1)-(3) are right: 85%; the decade claim (4) is what the data check tests.
Prior results: T-NODE (A49) +8.0(4.1)e-6, G(-5) (A52) +6.4(3.5)e-6, STEP 23 B_R1 +1.06(0.55)e-6 / B_M1 +0.59(0.84)e-6. All are decade-gated <= -5..-7 over sgn(M1) or s306.
 No record splits sign error energy by decade (searched TRIED_IDEAS, PLAN_sign for 'decade split', 'by decade', 'e*').
Known case: the job must reproduce xpsi0's e_x_signed_sum_wrong (s2, T) to 1e-10 and R1's w = 9.98e-8. Broken version: MREF + 10 decades must move the cumulative shares.
Not covered: 6x6 s2 stands in for 8x8 sgn(M1). The decade convention transfers by the psi^2-weighted median (gated_sign.py l.6), not by an identity.
Next rung (data, CPU, existing tables): runs/j1j2/consol_6x6/decade_split.
 Pass (conj holds): >= 80% of R1's net gain and of s2's wrong e*-mass at decades > -5. Then the 8x8 gate moves up (d = -3, -2, cost 1 + 0.91 n_g(d)).
 Kill: >= 50% at <= -5. Then the 8x8 null is R1's amplitude quality, and the question goes to the amp lead.
 Between: report the split. Cost < 1 CPU-h. Prediction: pass, 65%.
Status: DONE.
 RESULT (2026-11-27, job 18249001, runs/j1j2/consol_6x6/decade_split/result.json md5 a1b8a05d, re-read). KILL FIRES on the decade conjecture (4).
  - At decades <= -5 sit 0.738 of s2's wrong e*-mass and 0.720 of R1's net gain (-3.82e-5 of -5.31e-5), but only 0.0033 of the psi0^2 mass. At <= -6: 0.392 / 0.364.
  - Known cases are bit-identical, and the broken MREF+10 sends every share to 1.0.
  - So at 6x6 a (<= -5) gate keeps ~72% of R1's gain. The gate's location does not explain the 8x8 null.
 ADDENDUM: HOW MANY ADAPTED ROUNDS DOES THE ONE-STEP SIGN NEED? (sign lead, same day; this extends the paper check above)
  Derivation: y_A(x) = sgn sum_b c_b |A(z_b)/A(x)| S(z_b) [exact definition].
   - With A the amplitude that S came from, y_A ~ S: a near fixed point. Measured at 8x8: B_M1 flips 1.7% and gains +0.59(0.84)e-6.
   - The information that S is wrong at x enters only through A's collapse at x relative to its neighbours. The collapse score ln A_k/A_0 grows with the round count k: AUC 0.738 (k=1), 0.785 (k=2), 0.9996 (k=6) (collapse/result.json).
   - [conj, 60%] R1's gain needs several adapted rounds. The 8x8 R1 is ONE FN refit at fixed sgn(M1), which is k ~ 1, while the 6x6 success used chi_6, k = 6.
  Next rung (data, CPU, existing tables A_k of the s2 chain, as in collapse.py): the one-step sign from A_k for k = 0, 1, 2, 3, 6 and from chi_6. For each: P/R (psi0^2), net e* gain against s2 (all rows and <= -5).
   - Pass (conj holds): net(k=1) <= 30% of net(chi_6), rising with k. Then the 8x8 route is k adapted amplitude rounds at fixed sgn(M1) before the relabel, which is the amp lead's loop, and the question goes to FN_lead as a direction decision.
   - Kill: net(k=1) >= 70%. Then 8x8 sgn(M1) itself differs from s2, and the one-step route at 8x8 is closed.
   - Cost < 1 CPU-h. Prediction: pass, 60%.
 ADDENDUM RESULT (2026-11-27, job 18249705, runs/j1j2/consol_6x6/rounds_split/result.json md5 413d7139, re-read): conj (rounds needed) mostly DEAD.
  The one-step sign from A_k keeps this share of chi_6's net e* gain (all rows / decades <= -5):
  | k | 0 | 1 | 2 | 3 | 6 |
  |---|---|---|---|---|---|
  | share | 0.24 / 0.05 | 0.74 / 0.67 | 0.89 / 0.86 | 0.96 / 0.95 | 1.03 / 1.04 |
  - The same picture at alpha 0. The shuffled broken version gives +5.6e-4. Known cases pass (K1 rebuilds, S_R1 reproduced exactly).
  - One exact adapted round already gives 2/3-3/4 of the gain. The 8x8 null (one FN refit R1, gain 0) is therefore not the round count.
  - What is left: the 8x8 refit amplitude R1 does not carry the collapse an exact round carries, or sgn(M1)'s errors are not one-step-correctable at 8x8.

WHY THE 8x8 ONE-STEP SIGN GAINED NOTHING: COPY (a), SELF-CONSISTENCY (b) OR LEVER (c), PAPER CHECK (sign lead, 2026-11-27; FN-lead's three hypotheses, decision: sign folded into the stored loop)
Derivation:
 (a) [exact, f_J definition] f_J[A,S](z) = ln SK(z) - ln(D(z) + VA(z)) is the one-step estimate of ln(phi/A) at z.
     At 6x6 an exact round obeys ln(A_1/A_0) ~ l + 1.31 f_J[G] (eps^2 0.0033, 'u ~ l + 1.31 f_J[G]').
     So a refit that carries the round satisfies, on the gated rows, u_R1(z) := ln|A_R1(z)/A_M1(z)| ~ a l(z) + c f_J[M1, sgn M1](z), with R^2 near 1 after a gauge constant.
     [conj] If R^2 < 0.3, the refit does not carry the collapse where R1 reads it. That is (a).
 (b) [exact] VMC minimises E_var[S|A|]. Flipping row x changes E_var at first order by -(4/N) A_x sum_b H_xb S_x S_b A_b; a VMC optimum has no flip with a negative first-order change, so it is an R1 fixed point to first order.
     6x6 s2 is not one: R1(A_0) flips 417k rows at P 0.63 (rounds_split). [conj] A self-consistent start leaves one adapted round little to fix. That is (b).
 (c) [exact] At fixed |M1|, Delta E_var(S) = <E_L^S - E_L^{sgn M1}>_{|M1|^2}, with E_L^S(x) = H_xx + sum_z H_xz S(z)|M1(z)|/(S(x)|M1(x)|) and S changed only on gated rows by the STEP 23 rule.
     If -Delta E_var(B_R1) is below ~1.5e-6/site (3 sigma of B_R1's FN error), the FN arm could not show a gain whatever the rule's quality. That is (c).
     The broken arm's Delta E_var must come out positive and large, matching its FN +2.2e-5.
Checked: (a) at 6x6, A_1 against A_0 must reproduce R^2 ~ 1. That is a known case, and gives the bar. (c) G = empty gives Delta E_var = 0 exactly. (b) at k = infinity the fixed point is the FN-consistent sign.
 Confidence the three tests separate the hypotheses: 75%.
Prior results: STEP 23 (B_R1 +1.06(0.55)e-6, B_M1 +0.59(0.84)e-6, broken +22(12)e-6); rounds_split (k=0 0.24, k=1 0.74); cycle3_gate (Gchi f_J spread 0.016 against cut 0.009); u ~ l + 1.31 f_J[G] (STORAGE_PROBLEM).
Known case: (a) 6x6 exact A_1 vs A_0, R^2 >= 0.9; broken = u shuffled within decades, R^2 ~ 0. (b) S* = R1 iterated to its fixed point on A_0 must flip 0 rows under R1(A_0).
 (c) the broken arm, Delta E_var > 0.
Not covered: (a) uses the one-step f_J as the reference for the true suppression; f_J itself carries M1's copy error.
Next rung (existing tables and stored networks, no training, no FN):
 8x8 job, (a)+(c): on |M1|^2 samples, rows gated by the STEP 23 rule.
  (a) R^2 of u_R1 against (l, f_J[M1]). (a) holds if R^2 < 0.3; it is dead if R^2 >= 0.7.
  (c) Delta E_var for B_R1, B_M1 and broken. (c) holds if -Delta E_var(B_R1) < 1.5e-6.
 6x6 job, (b): S* = R1 fixed point on A_0; one exact round_map at fixed S* gives A_1*; gain share of R1(A_1*) over S* against T, as e* net. (b) holds if the share is < 30%.
 Predictions: (a) holds 50%; (b) 35%; (c) holds 40%. Cost: 8x8 ~2 A40-h, 6x6 ~2 CPU-h.
Status: DONE.
 RESULT (a)+(c) (2026-11-27, job 18255770, runs/j1j2/sign_bad/branched/step23/diag_ac/result.json md5 b56be84d, re-read). Sample: own |M1|^2 Metropolis, 512 chains, n 20480, E_L0 -0.49865/site.
  (a) HOLDS. u = ln|A_R1/M1| on gated rows against (l, f_J[M1]): R^2 0.251 (halves 0.253/0.249, shuffled 1e-4); a 0.10, c 4.1.
   The exact 6x6 round reaches eps^2 0.0033, i.e. R^2 ~ 0.997, with a 1.0, c 1.3. So the 8x8 refit keeps a quarter of the round's structure on the rows R1 reads.
  (c) HOLDS as registered: Delta E_var at fixed |M1| is +0.53(0.46)e-6 (S_R1), +0.04(0.14)e-6 (S_M1), +4.3(3.7)e-6 (broken).
   R1's gated flips are neutral or slightly harmful at |M1|. S_M1 ~ 0 is what (b) predicts: sgn(M1) is first-order optimal for |M1|.
   The lever is not zero: random flips at the same count move E_var by ~4e-6, and the FN arm by ~2e-5.
  Reading: the 8x8 R1 label was read off an amplitude that does not carry the round; R^2 0.25 against ~1 is the failing term.
   Inside the loop, the stored amplitude must reach R^2 against (l, f_J) near the exact round's before R1 on it can gain. Arm C at 6x6 measures this on a copy. (b) is pending (selfcons_b).

RESULT (b), SELF-CONSISTENCY (sign lead, 2026-11-28). (b) DOES NOT HOLD.
Iterating R1 on fixed A_0 converges in 10 steps to S* (flips 417426 -> 0, ~4x per step, no oscillation); w(S*) 1.011e-6 against 0.934e-6 after one step and T 0.116e-6.
One exact round at S* (efn E' -18.136882, res 9.3e-7), then R1(A_1*): share of the attainable gain 0.548 (<=-5 decades: 0.546), P/R vs T 0.992/0.534, w 1.011e-6 -> 0.536e-6. Kill line was share < 0.30.
For comparison, from s2 the same round gives a share of 0.764. Self-consistency costs about 0.22 of share, but R1 still keeps most of the round's gain once an exact round sits under it.
Known cases: round at s2 reproduces stored lA_1 to 4.1e-10 (broken E'+1e-3: 1.1e-4, fails); R1(stored A_1) on s2 = rounds_split k=1 exactly (346546 flips, net -3.940488e-5). Broken version (R1 with A_0 on S*): 0 flips.
Path: runs/j1j2/consol_6x6/selfcons_b/result.json (md5 a24514ee), job 18264587; attempt 1 (18255344, TIMEOUT) in attempt1/.
VERDICT (a)/(b)/(c) at 8x8: (a) holds, (c) holds as registered, (b) does not. The failing term is the amplitude copy:
the refit 8x8 amplitude carries the round with R^2 0.25 (exact 6x6 round ~0.997). R1 is not the problem; the amplitude it reads is.
Consequence for the loop: the R1 evaluation inside the 8x8 loop can gain only as far as the stored amplitude carries each round; arm C's N_1 measures exactly that at 6x6.

ARM C (THE LOOP WITH NOTHING EXACT INSIDE), RESULT (amplitude lead 2026-11-28; runs/j1j2/consol_6x6/cycle6f_store/result.json md5 f5d44545, re-read; jobs 18260460-65)
  N_3 = 0.774e-6 [exact grade, deterministic]: BETWEEN (pass <= 0.665, kill >= 0.90). Mine 0.52-0.66 (60%) and FN-lead 0.50-0.70 (45%): both MISSED.
  Per round (1e-6/site): N(chi_k) 0.946 / 0.820 / 0.801 / 0.774; N(T_{k+1}) 0.594 / 0.546 / 0.539; L_k 0.226 / 0.256 / 0.235 (flat, not rising).
  Flags on lG(chi_k) under S_R1: 75,936 / 93,611 / 87,477 / 130,878 rows, weight 1.5 / 1.0 / 2.1 / 1.1e-8. E_var - E0: 1.66 / 1.96 / 1.82 / 1.91.
  The model with the measured L reproduces the chain (N_1 = M(0.946) + 0.226 = 0.820 exactly; N(T_2) 0.546 vs M(0.820) 0.572): the miss is L alone.
  Deciding term: L 0.24, 3x the s2-measured 0.076-0.091 that the prediction carried. Fit MSE (same net, loss, steps) 2.6e-6 vs x1's 6.1e-7 (4.3x).
  Flat L says no representation floor across these targets (0.59 -> 0.54); the step from s2 to S_R1 itself costs the copy.
  Same verdict as the sign lead's 8x8 (a) (selfcons_b, md5 a24514ee): the amplitude copy is the failing term, not R1.

WHERE DOES ARM C's COPY LOSE?, REGISTERED (amplitude lead 2026-11-28, before the run; rung 2, existing tables, grades only)
  Derivation [approx, l.1897]: L ~ sum_{viol(S)} |H_xy| phi_x phi_y 4 sinh^2((d_x - d_y)/2) / sum phi^2, d = chi_1 - T_1. Two readings:
  (i) sign-local: d is large on or next to the S_R1-flipped rows (the round's step is largest there; the net has no input that knows the flip);
  (ii) diffuse: the S_R1 target is harder everywhere for width 64 (fit MSE 4.3x over the whole table).
  Measurement [exact, grader = round_map start N]: splice tables, constant aligned (weighted mean of d removed): T_1 on rows R, chi_1 elsewhere.
    G2: N(chi_1), N(T_1) under s2 -> L_s2.  G3: R = S_R1-flipped rows (359,588).  G4: R = top 1% and top 10% rows by |d| (by count).
    Plus: rms of the round's step r = T_1 - chi_0 - c1_0 (weighted n e^{2T}) split flipped / not, vs the same on x1's round (if its target is on disk).
  Known case: R = none -> 0.8202, R = all -> 0.5943 (result.json). Broken: splice without aligning the constant must change N.
  Decision: (i) if G3 recovers >= 50% of L_0 -> next is one refit with the flip in the copy (input feature or bond-cost loss weight), ~20 min A40;
    (ii) if G3 < 20% and G4(10%) < 50% -> next is a width-128 refit of round 0; between: report the shares.
  Prediction (mine, 55%): (ii) diffuse; L_s2 0.15-0.25. Cost: ~8 grades, ~20 GB each, cip A40 node as the stages, ~3 h.
  Status: paper check DONE (derivation l.1897, known case + broken above); data check DONE (result.json re-read, x1 nn.json loss 6.1e-7).
  Addendum (same day, before the run): FN-lead registers against mine: flipped rows + their one-exchange neighbours carry >= 50% of L_0 (55%).
    The reason, on paper: six s2 rounds collapsed these rows. After the flip, the round regrows them: a large positive f_J on rows of low weight, which the Born-weighted fit barely sees.
    The run adds splice G3b (R = flipped plus one-exchange neighbours).
    The remedy if it holds, alongside "flip as input": on R1-flagged rows, compute the guide by one local power step, psi_x = -(sum_{y!=x} H_xy psi_y)/(H_xx - E), instead of learning it. Check it against TRIED_IDEAS B18 (net + one hop kept 82%) before staging.

WHERE DOES ARM C's COPY LOSE?, RESULT (amplitude lead 2026-11-28; job 18267822; runs/j1j2/consol_6x6/c6_where/result.json md5 38a54ec1, re-read)
  The loss is spread across the table, reading (ii); my prediction HIT. FN-lead's flipped+ring >= 50% MISSED (43%).
  L_0 shares (exact grades, 1e-7/site):
    G3, flipped rows: 0.4%. G4 top 1% of |d|: 0.2%. G4 top 10%: 2.7%.
    G3b, flipped + one-exchange ring (69% of rows, weight 0.169): 43%. Per unit of weight that is 2.6x the rest, but the ring is not local.
  Known case: chi_1 and T_1 under S_R1 rel 0. Broken version: the unshifted splice differs by only 1e-13 (unusable); a +0.05 shift moves N by 5.6e-10.
  Under s2 the same copy loses L_s2 = 0.187 (N 1.115 vs 0.928), so the sign does not drive it.
    s2's 0.076-0.091 came from another copy (x1); x1's own target was deleted 11-26 and cannot be regraded.
  Residuals: flipped rows miss by rms 0.31 in log, but carry weight 4e-7 and a negligible share of L.
  Learning curves at equal steps (val): fit0 lags x1 from step 5k. At 20k: 6.9 vs 3.1e-6; at 80k: 2.7 vs 0.61e-6. Both fall 10-12% over the last 20k steps.
  Deciding term: the fit residual everywhere (MSE 4.3x x1's).
    Candidates: width, steps, inputs (the z column rebuilt by the NNc/NNd7 recipe, which x1's original inputs cannot be checked against), and the target itself.

COPY CAPACITY OR STEPS?, REGISTERED (amplitude lead 2026-11-28, before the run)
  Two refits of round 0, each on fit0's exact inputs (map0, mapP0, c1_0), graded by the round_map start N under S_R1:
    W128: NN_WIDTH=128, 80k steps.
    S240: width 64, NN_NSTEP=240000.
  Prior: B31 (10-23, PP/RBM refit tax) found capacity and time did not cut the tax. Its class and objective differ, so it does not bind this MLP.
  Pass: L_0 <= 0.12 in either arm. By the model that gives N_3 ~ 0.59 if it holds every round; the next step is that arm's settings through 3 rounds.
  Kill: both arms L_0 >= 0.20. Then width and steps are not the term; the next step is the inputs (z-column rebuild vs the NNx recipe) or the target.
  Between: report both L_0 and their curves.
  Prediction (mine, 50%): W128 L_0 0.13-0.19; S240 0.17-0.21. Cost: 2 x ~1 h on a full A40 + 2 grades.
  Status: paper check DONE (deciding term from the result above, known case = fit0 reproduced by its own settings); data check DONE (c6_where re-read).

COPY CAPACITY OR STEPS?, RESULT (amplitude lead 2026-11-28; jobs 18270434-36; runs/j1j2/consol_6x6/c6_cap/result.json md5 a8030866, re-read)
  KILL fired: neither width nor steps is the term. Both arms have L_0 >= 0.20. My prediction (W128 0.13-0.19) MISSED.
  L_0 (1e-6/site) and final train MSE:
    fit0 (width 64, 80k steps): L_0 0.226, MSE 2.60e-6.
    W128 (width 128, 80k steps): L_0 0.224, MSE 2.61e-6.
    S240 (width 64, 240k steps): L_0 0.229, MSE 2.17e-6.
  W128 under s2: L_s2 0.181. Known case chi_1 rel 0.
  The fact that redirects: S240 cuts the MSE by 17% while L_0 does not move (+0.003). So the copy loss is not set by the size of the fitted residual.
  Diagnosis [equation]: the l.1897 estimate is 2nd order in d because it assumes the guide sits at the FN optimum, where E_FN is stationary.
    T_1 is not at that optimum: N(T_1) = 0.594, while the fixed point is N* = 0.271. So E_FN has a FIRST-order term:
      N(T + d) - N(T) = sum_x g_x d_x + O(d^2),   g = dN/d(log guide) at T.
    An MSE fit leaves a residual whose projection on g is not minimised. If its bias lies along g, it costs L linearly, whatever the width or steps.
    x1's target (round 6, near the s2 fixed point, g small) would explain s2's 0.08. A flat L over rounds 1-3 also fits: all three rounds sit far from N*.

FIRST OR SECOND ORDER?, REGISTERED (amplitude lead 2026-11-28, before the run; grades of existing tables)
  Tables T_1 + lam (chi_1 - T_1 - c), c = 1.0142e-5 as in c6_where. Grade them for lam = 0.5, -0.5, -1 under S_R1.
  Fit N(lam) - N(T_1) = a lam + b lam^2 through the 5 points (lam = 0 and 1 are already known: 0.5943, 0.8202).
  Pure 2nd order: N(0.5) and N(-0.5) both sit +L/4 above T_1. Pure 1st order: they sit at +L/2 and -L/2.
  Decision:
    a/L >= 0.5 (1st order dominates): the copy loss is the MSE fit's bias along the FN gradient.
      Next: a copy loss aware of the FN gradient, or a one-parameter correction along d. Write that on paper before any run.
    a/L <= 0.2 (2nd order): the L-norm residual is not the MSE residual. Next: the violated-bond-weighted loss (l.1904).
    Between: report a, b.
  Known case: lam = 0 and 1 reproduce 0.5943 and 0.8202. Broken: the quadratic through 0, 0.5 and 1 must predict lam = -1 within 5% of L, or the form is wrong.
  Prediction (mine, 55%): a/L >= 0.5, with N(-0.5) below N(T_1). Cost: 3 grades, ~40 min, cip 72G.
  Status: paper check DONE (derivation above); data check DONE (c6_cap re-read).
  Addendum (FN-lead, same day, before the run): a line that uses only stored objects, which therefore exists at 8x8. ln chi(gamma) = ln chi_0 + gamma (ln chi_1 - ln chi_0), with gamma = 1.5, 2, 3.
    If the linear term dominates and the copy shrinks the step, then d is antiparallel to the step s = T_1 - chi_0, and over-relaxing reverses the shrinkage for free.
    gammascan/result.json (762926d4) has gamma 2-4 speeding early exact rounds, so the two effects would add.
    FN-lead's prediction (40%): some gamma in [1.5, 3] gives N(gamma) <= N(chi_1) - 0.11. Kill: no gamma improves N by more than 0.03.
    Predictor computed before the grades: cos(d, s) and beta = <d,s>/<s,s>, weighted by n e^{2T_1} and by psi0^2. Shrinkage shows as a clearly negative value.
    Mine: I agree on the direction. 35% that it reaches >= 0.11, because the splice showed d is diffuse, while the step is concentrated where the round acts.

FIRST OR SECOND ORDER?, RESULT (amplitude lead 2026-11-28; jobs 18272931 (lam), 18272966 (gamma); runs/j1j2/consol_6x6/c6_lam/result.json md5 122082f3, result_gamma.json md5 206fe951, re-read)
  N(lam) - N(T_1), 1e-6/site:
    lam -1: +0.068;  lam -0.5: +0.031;  lam 0: 0;  lam 0.5: +0.110;  lam 1: +0.226.
  Exact 4-term solve: odd part 0.079 lam (cubic term ~0); even part 0.326 lam^2 - 0.180 lam^4.
  The linear share is a/L = 0.35: BETWEEN. N(-0.5) lies ABOVE T_1, so my prediction (55%) MISSED.
  The quadratic through 0, 0.5 and 1 misses lam = -1 by 1.19 L: the expansion fails beyond |lam| ~ 0.5 (even part saturates).
  Reading: T_1 sits near the minimum of N along d. Reversing the error gains nothing, and the copy's error is not mainly a downhill bias.
  Gamma (stored step over-relaxed, FN-lead's line): N 0.820 (gamma 1) / 0.772 (1.5) / 0.817 (2) / 1.742 (3).
    Best gain 0.048: BETWEEN FN-lead's kill (0.03) and pass (0.11).
    Predictor: cos(d, s) = -0.31 and beta = -0.27, weighted n e^{2T_1}. The copy keeps 73% of the step, and 90% of d's variance is orthogonal to it.
    psi0^2 weighting is not computed: no psi0 table on the graders' path.
  RECORDS LAPSE (mine): I registered the order test without running recall. The 11-23 records already held a line test and its sequel, under s2:
    - 'LINE TEST', f2 copy: first order, f(-0.5) < 0.
    - 'PRE-CORRECTED REFIT': the copy repeats its error (cos 0.87). The error is a REACH limit of the inputs; target shifting is closed.
    - 'WHERE THE UNREACHABLE PART LIVES': adding the round's step c1 as an input made d 60% predictable (R2), and that gave x1 (q_6 0.81).
  So today's capacity kill is that reach limit again. The lever is the inputs, not the loss or the target.

WHICH INPUT DOES THE S_R1 COPY MISS?, REGISTERED (amplitude lead 2026-11-28, before the run; existing tables, no training; NNw method)
  Derivation: T_1 = chi_0 + c1_0 + c2_0 exactly, with c2_0 = 1.307 f_J under S_R1 (= r of c6_where). fit0 sees z (10 columns), the shadow step (mapP0 l_part) and c1_0.
    Under s2, f_J was minor (NNw: R2 +0.15). After a sign change the f_J part of the round is where the flip acts, so it is the candidate the inputs cannot see.
    The other candidate: chi_0 itself (x1 = F(z, c1 of round 6)). Its round-6 input is not among fit0's inputs.
  Measurement: R2_w(d | columns), d = chi_1 - T_1, w-weighted kNN on rank-transformed columns, k 16/64, as ref2hop/NNw/nnw.py. Column sets:
    base = fit0's 12 inputs; base + c2_0; base + chi_0; base + c2_0 + chi_0.
    Known case: base + d gives R2 >= 0.9. Broken: base + shuffled c2_0 stays within 0.03 of base.
  Decision:
    +c2_0 raises R2 by >= 0.3: next is one refit with c2_0 as a 13th input. That costs the same as c1, since f_J needs only chi_k at the neighbours.
    +chi_0 raises R2 by >= 0.3 and c2 does not: the loss is nesting (round k's table depends on round k-1's step). Then a paper question for FN-lead (flat-cost design).
    Both < 0.1: the residual is spread over many hops; report.
  Prediction (mine, 50%): c2_0 dominant, +0.3 to +0.5. Cost ~1-2 h CPU.
  Status: paper check DONE (derivation above + the 11-23 records); data check DONE (c6_lam, c6_where re-read).

WHICH INPUT DOES THE S_R1 COPY MISS?, RESULT (amplitude lead 2026-11-28; job 18274515; runs/j1j2/consol_6x6/c6_reach/nnw.json, re-read)
  R2_w(d | columns), sample / S, k16 and k64:
    base (fit0's 12 inputs): -0.01/-0.02 and 0.04/0.03.
    + c2_0: 0.48/0.52 and 0.48/0.53.
    + chi_0: 0.01/-0.02 and 0.06/0.03.
    + c2_0 + chi_0: 0.49/0.53 and 0.49/0.53.
    Known case, + d: 0.93/0.95 and 0.91/0.93.
  Gate: the broken version (+ shuffled c2_0) MISSED its 0.03 tolerance: it fell BELOW base by 0.025-0.039 in all four cells.
    I accept the run, as on 11-23 (NNw, 0.004 over).
    The gate exists to catch a spurious GAIN, and the shuffled column produced a LOSS: a noise column blurs kNN.
    The c2_0 gain (+0.49) is 13x that offset.
  Verdict: c2_0 (= 1.307 f_J under S_R1, the part of the round where the flip acts) makes half of the copy error predictable; chi_0 adds nothing (no nesting loss).
    My prediction (c2 dominant, +0.3-0.5, 50%): HIT at the top of the band.
    S here is the old NNg top-|g| set (no g for this round), used only to pick rows; sample and S agree.

C2 AS A 13TH INPUT, REGISTERED (amplitude lead 2026-11-28, before the run)
  Refit of round 0 as fit0 (width 64, 80k steps, cosine, same 12 inputs), plus column 13 = c2_0 = T_1 - chi_0 - c1_0, graded under S_R1 (start N).
  nn_fit_jax.py takes one extra column only. The copy nn_fit_jax_x3.py adds NN_EXTRA3 and changes nothing else.
    Known case: with NN_EXTRA3 unset, step 1 equals fit0's chkS (1.05356e-1 / 8.04127e-2). GRADCHECK must pass with the 13th column.
  Prior: 11-23, c1 as an input raised R2 to 0.60 and q_6 from 0.46 to 0.81, which is about 65% of the copy cost removed.
  Pass: L_0 <= 0.12. Next: arm C with c2 as an input through 3 rounds (stage saves c2_k).
  Kill: L_0 >= 0.20. The R2 gain then does not reach the grade; diagnose.
  Prediction (mine, 50%): L_0 0.09-0.15.
  Cost: one A40 fit (~20 min) + 1 grade. At 8x8, c2 costs z^2 chi_k evaluations per amplitude (delta_today [parallel] note).
  Status: paper check DONE (result above; cost note); data check DONE (c6_reach re-read).
