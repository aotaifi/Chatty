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
  (With the diagonal term, C = 1 + alpha(H_xx - E) < 0 everywhere (PLAN_sign), C S|B| and Q share the sign -S(x) at such rows, so the full Lanczos
  label does not flip them relative to the bulk: it only raises |psi'(x)| by |alpha| Vs.)
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
