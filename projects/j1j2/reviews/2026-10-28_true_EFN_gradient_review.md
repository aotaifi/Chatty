# Review: "true E_FN gradient" scheme (HF gradient + frozen-chi* inner steps), 2026-10-28

Worker review. I only read and derived: no jobs, no code read, no numbers recomputed. Every statement below is a
derivation. Where I could not see the implementation, I write the statement as a condition. The measured
numbers in claim (4) are the lead's; I did not re-derive them.

## Verdict
**Sound, with conditions.** No term breaks the scheme. The majorise-minimise (MM) argument is stronger than the
Hellmann-Feynman (HF) argument it was built on: monotone decrease survives degeneracy, a loose solver and any
inner step count k. The conditions are C1-C8 at the end. One point changes how the scheme should be read:
the scheme with k -> infinity and enough network capacity **is** the classic iteration "guide <- |FN ground
state|" (section b3). So it brings no new fixed point and no faster per-solve rate unless it is accelerated.
Claim (4) is half right: the pair loss with phi = chi*(theta_current) is the same majoriser and has the same
gradient. It is loose only when phi was solved for a different guide.

Notation. S = diag(s2). s_x = ln|A(x)|. u_xy = |A(y)|/|A(x)| = e^{s_y - s_x}. B = the set of blocked bonds.
chi is normalised, sum_x chi_x^2 = 1, and real (see C3).

## (a) Is Hellmann-Feynman valid?

**a1. The identity in claim (1) is exact.**
<chi|diag V|chi> = sum_x chi_x^2 sum_{y:{x,y} in B} |H_xy| u_xy = sum_{{x,y} in B} |H_xy| (chi_x^2 u_xy + chi_y^2 / u_xy) = L(theta; chi).
Only V depends on theta, so dH_FN/dtheta = diag(dV/dtheta). Therefore ∂_theta L(theta; chi*)|_{chi* fixed} = <chi*|dH_FN/dtheta|chi*> identically.
The formula needs <chi*|chi*> = 1. Otherwise divide by the norm.

**a2. The blocked set is fixed.**
B = {(x,y): s2_x H_xy s2_y > 0} depends only on s2 and H, not on theta. So H_kept is constant, and theta enters
only through the smooth diagonal V. V is smooth wherever A(x) != 0. The only non-smoothness is at zeros of A,
where V_x ~ 1/|A_x| and ∂ s_x ~ 1/A_x diverge.
- Condition: the code must build B from the s2 table, never from sign(Re A) or the phase of A. Otherwise B
  depends on theta and E_FN is piecewise.
- Condition: in the orbit basis, H_xy is a character-weighted sum and can be exactly 0. Such pairs are not edges.

**a3. Degeneracy: Perron-Frobenius settles it.**
S H_FN S has off-diagonals s2_x H_xy s2_y <= 0 on the kept bonds, so H_FN is stoquastic in the s2 gauge.
- If the kept graph restricted to the sector is connected (irreducible), the ground state is non-degenerate and
  s2_x chi*_x > 0 for every x. Then E_FN(theta) is analytic wherever A != 0, and HF holds. A small gap does not
  invalidate HF.
- If the kept graph is disconnected, H_FN is block diagonal and E_FN = min over components. Components can cross
  in energy, and then E_FN has a concave kink (Danskin). HF fails at the crossing, but every branch gradient is
  still a descent direction (see b1).
- Symmetries left unreduced cannot create degeneracy when the graph is irreducible.
- Required check, once only, because s2 is fixed: count the connected components of the kept graph in the sector.

**a4. Near-degenerate gap Delta (E_1 - E_0 of H_FN in the sector).** HF stays exact. Three quantities scale as 1/Delta:
- the curvature, d^2E = -2 sum_n |<n|dV|chi*>|^2 / (E_n - E_0) + <chi*|d^2V|chi*>;
- the eigenvector error, ||delta chi|| <~ r/Delta for residual r;
- the gradient error, which is **first order**: delta g = 2<delta chi|dV|chi*>. The energy error is second order (r^2/Delta).
Delta comes free as the second Ritz value of the solver and should be logged every solve.

**a5. Complex parameters.** E is a real function of (Re theta, Im theta). If A is holomorphic in theta, write O_x = ∂ ln A_x / ∂theta. Then:
- ∂ s_x / ∂ Re theta = Re O_x
- ∂ s_x / ∂ Im theta = -Im O_x
- The steepest-descent step is delta theta = -eta (g_R + i g_I) = -eta sum_x w_x conj(O_x), with w_x = ∂L/∂s_x.

A step of -eta sum_x w_x O_x (no conjugate) descends in Re theta and **ascends** in Im theta. The run still
decreases E slowly and looks fine. Autodiff conventions differ: PyTorch returns the conjugate Wirtinger
direction; JAX `grad` of a real function of complex input returns its conjugate. Any non-holomorphic layer
(|.|, a real activation of a complex value, a normaliser) invalidates the O_x form, and then only the full real
gradient is correct.

Two checks that cost no eigensolve:
1. Finite differences of L(theta; chi) separately along Re and Im of random directions, against the implemented gradient.
2. sum_x w_x = 0 to rounding. This is exact because L(s + c·1) = L(s); it fails on a wrong gradient code.

## (b) Frozen-chi* inner steps

**b1. The frozen L is a majoriser, so the scheme is MM.** Define F(theta, chi) = <chi|H_kept + diag V_theta|chi> with ||chi|| = 1. Rayleigh-Ritz gives
  E_FN(theta) = min_chi F(theta, chi).
So M_k(theta) := <chi_k|H_kept|chi_k> + L(theta; chi_k) >= E_FN(theta) for all theta, with equality at theta_k. If the inner steps end with L(theta_{k+1}; chi_k) < L(theta_k; chi_k), then
  E_FN(theta_{k+1}) <= M_k(theta_{k+1}) < M_k(theta_k) = E_FN(theta_k).

Robustness of the bound:
- **Degeneracy:** any chi in the ground space gives a valid majoriser, so monotone decrease survives.
- **Inexact solve:** the bound holds for any normalised chi~, with touching gap eps = F(theta_k, chi~) - E_FN(theta_k) ~ r^2/Delta >= 0, so the decrease is monotone up to eps. With a warm start from chi_k, Lanczos or LOBPCG Ritz values are <= the Rayleigh quotient of the start, so the sequence of reported Rayleigh quotients is monotone.

Names for this structure:
- majorisation-minimisation (MM): Hunter & Lange, Am. Stat. 58, 30 (2004); Lange, *MM Optimization Algorithms* (SIAM 2016);
- EM read as coordinate descent on a free energy F(q, theta), with chi in the role of q: Dempster-Laird-Rubin 1977; Neal & Hinton 1998;
- the gradient statement is Danskin's / the envelope theorem;
- the physics ancestor of the upper bound is ten Haaf et al., PRB 51, 13039 (1995);
- a continuum analogue of "re-solve, then update the guide" is self-healing DMC (Reboredo, Hood, Kent, PRB 79, 195117 (2009)). Cited from memory: check before quoting.

**b2. What bounds k: nothing, for monotone decrease.** True decrease = (decrease of M_k) + gap(theta_{k+1}), where gap(theta) = M_k(theta) - E_FN(theta) >= 0 and
  gap ≈ sum_{n>0} |<n|Delta V chi_k>|^2 / (E_n - E_0) <= ||(Delta V - <Delta V>) chi_k||^2 / Delta.
The gap is a bonus, not a cost: the majoriser under-reports progress. So k is bounded only by:
- the cost of an inner step versus the cost of a solve;
- the risk of fitting solver noise in chi_k where |chi_k| is near the noise floor (see d2).

The one hard requirement is that the inner loop ends with L lower than it started. Adam does not guarantee
this. Keep the best iterate, or use a monotone method (see c3).

**b3. With k -> infinity and enough capacity, the scheme is the classic iteration.** In s, each bond term
|H_xy| (a e^{s_y - s_x} + b e^{s_x - s_y}), with a = chi_x^2 and b = chi_y^2, is convex. All terms are minimised
together by s = ln|chi_k|. So a complete inner minimisation gives |A_{k+1}| = |chi*(A_k)| on the blocked
endpoints, which is "guide <- |FN state|".

The fixed point |A| = |chi*(A)| is stationary:
  ∂L/∂s_x = sum_y |H_xy| (-chi_x^2 u_xy + chi_y^2 / u_xy) = 0 at u_xy = |chi_y|/|chi_x|.
So the scheme has the same fixed points as that iteration and the MM linear rate. Inference, not checked: if the
cx_s2ceil arm (projects/j1j2/results/compress_6x6/runs/cx_s2ceil.json, step ratio 0.926 per solve) is that
iteration, the "true gradient" scheme will not beat 0.926 per solve without acceleration. Options are SQUAREM
(Varadhan & Roland, Scand. J. Stat. 35, 335 (2008)), Anderson mixing over rounds, or L-BFGS on E_FN with the HF
gradient. The scheme differs from that iteration only when the network cannot represent |chi*|: then it finds
the stationary point of E_FN inside the network manifold, which is the right objective.

## (c) Guards

**c1. The scale fix sum_x phi_x^2 s_x = const.** E is invariant under s -> s + c·1, and also under any phase field on A.
- As a pure gauge it is harmless with any weight, phi^2 or chi^2. It is pure gauge when it is done by an analytic
  rescale or by a dedicated scalar log-scale parameter that the optimiser does not touch.
- It is **wrong** when it is a penalty or a projection on the network parameters and the network has no exact
  scale direction, for example a normalised layer. Its gradient sum_x phi_x^2 ∇s_x then has physical components,
  and it removes or biases real descent.
- Check: Pf(lambda F) = lambda^{N/2} Pf(F) gives an exact scale direction only if the pairing matrix enters linearly.

**c2. The cap on |Delta s_x| per step.** E depends only on s_y - s_x on blocked bonds, so the cap has three problems:
- (i) It is gauge-dependent unless it is measured after the gauge fix. Better: cap |Delta(s_y - s_x)| over (x,y) in B.
- (ii) Configurations that touch no blocked bond do not enter E at all. They should not count.
- (iii) A max-norm cap turned into a global step scale lets the deep region stall the whole step. Deep
  amplitudes carry weight chi^2 < 1e-11, so they are the least constrained and the most volatile under Adam.

The metric that actually controls the majoriser gap is ||Delta V chi||^2 / Delta, which is chi^2-weighted. A
per-x clip (free amplitudes) stalls the deep region itself: it needs |Delta s|/cap steps, so a factor of 100
mismatch is about 4.6/cap rounds. The MM decrease does not need a cap. The only role of the cap is to keep the
step away from new near-zeros of A, where V diverges and the gradient's Lipschitz constant goes as 1/|A_x|^2.

**c3. The optimiser.** The inner problem is deterministic and full-batch. Adam normalises tiny gradients (flat
phase and scale directions, and deep-only directions) to O(lr) steps. This is the likely source of the
deep-region volatility that the cap then fights. L-BFGS or gradient descent with an Armijo backtrack on L gives
a guaranteed monotone M_k and makes the cap unnecessary.

## (d) Failures that look fine

**d1. Wrong eigenvector.**
- Certificate (PF, needs connectivity): s2_x chi_x > 0 for every x. Report the count of sign violations by bin of chi^2. Violations only in the deepest bins mean those values are solver noise.
- Disconnected kept graph plus warm start: the start has zero weight on the other components, so Lanczos never
  finds a component whose energy dropped below. The run is monotone and smooth, and the eigenvalue is wrong.
  Rounding seeds that component only slowly. The connectivity check (a3) excludes this. An occasional cold random start detects it.

**d2. Solver tolerance (residual ~6e-7).**
- Eigenvalue error <= r^2/Delta ~ 4e-13/Delta: negligible against E_FN - E0 ~ 3e-5 total (0.915e-6 per site x 36, from the loop review).
- Eigenvector error ~ r/Delta, and the gradient error is first order. If the error is spread uniformly over N =
  1.58e7 orbits, the per-orbit error is ~ (6e-7/Delta)/sqrt(N) ~ 1.5e-10/Delta. That is below |chi| = 3e-6 at
  chi^2 = 1e-11, and becomes comparable only near chi^2 ~ 1e-17 (Delta ~ 0.1). Inference: the 1e-11 region is
  probably resolved **unless the error concentrates there**.
- Test: per-bin local energy, E_loc(x) = (H_FN chi)_x / chi_x - E, binned by chi^2. Report max and rms per bin, and compare the gradient at two tolerances.
- Ask: is 6e-7 an absolute ||H chi - E chi|| with ||chi|| = 1, or relative to |E| ~ 18? If relative, all of the above is 18x worse.

**d3. Precision.**
- L = sum chi^2 V is O(1). Changes of interest are ~1e-8 to 1e-5, so in float32 a monotone check on L is blind.
- The gradient terms -chi_x^2 u + chi_y^2/u cancel near the fixed point.
- Implement L in the algebraically equivalent sum-of-squares (pair) form
    L = sum_B |H_xy| (|chi_x| sqrt(u) - |chi_y| / sqrt(u))^2 + 2 sum_B |H_xy| |chi_x chi_y|,
  where the second term is constant in theta. Do it in float64.

**d4. Orbit basis.** HF and Rayleigh-Ritz need H_FN symmetric. This means normalised orbit states, or
sqrt(N_y/N_x) factors applied consistently to both H_xy and A. A representative-basis H with H_xy != H_yx makes
V wrong and needs left eigenvectors, and the formulas silently give the wrong gradient. The sector must have
real characters for s2 = ±1 to be meaningful (C3).

**d5. Monotone does not mean correct.** A wrong Im-part sign (a5) or a biased gauge penalty (c1) still gives a
monotone, slower decrease. Compare the realised decrease with the predicted g · delta theta for each round. A
ratio well below 1 at the first inner step flags a gradient bug; above 1 is the gap bonus.

## Claim (4)
Let P(theta; phi) = F(theta, phi) - <phi|H|phi>. With phi = s2|phi| (PF), H_xy phi_x phi_y = |H_xy||phi_x phi_y| on blocked bonds, so
  P = sum_B |H_xy| (|phi_x| sqrt(u) - |phi_y| / sqrt(u))^2 >= E_FN(theta) - <phi|H|phi>.
- The bound is correct.
- P >= 0, while the right-hand side can be negative. P can therefore never reward a guide that improves beyond phi; it pulls |A| toward |phi|.
- But ∇P = ∇L(·; phi). If phi = chi*(theta_current), P is exactly the MM majoriser M_k: its gradient is the true gradient and its value is tight.
- The over-weighting is the majoriser gap. It is second order in the distance from the guide that phi was solved for, and it comes from chi* relaxing away from regions where V is wrong.
- So the contrast with (2) is the frequency of re-solving, not a different objective.

On the evidence: "52-81% cost share below phi^2 1e-11" is an additive share of P. "Splicing removes 35% of the
true loss" is a non-additive response. These are not the same quantity. Compare like with like: Delta P and
Delta E_FN under the same splice. Also, "exact values" must be named: |chi*| of the current H_FN, or |psi_0|?

## Conditions for "sound"
- C1. The kept graph in the sector is connected (one-time check). Otherwise HF holds only off crossings, and the warm-started solver can be trapped (d1).
- C2. B is built from the s2 table, not from A.
- C3. H_FN is real symmetric in a normalised orbit basis of a real-character sector, and chi is normalised.
- C4. The complex gradient is the real gradient in (Re, Im): a finite-difference test on L, and sum_x w_x = 0.
- C5. The scale fix is a pure gauge (analytic rescale), not a penalty on network parameters.
- C6. The cap is on blocked-bond differences, gauge-invariant, or is replaced by a backtracking on L. Each inner loop ends with L below its start.
- C7. Solver accuracy is checked per chi^2 bin (local energy, sign certificate). Log Delta. The units of the residual are stated.
- C8. L is evaluated in the pair (sum-of-squares) form in float64.

## Not checked
The implementation, the network's holomorphy and scale structure, Delta, the kept-graph connectivity, the
meaning of the residual's units, and whether cx_s2ceil is the "guide <- FN state" iteration.
