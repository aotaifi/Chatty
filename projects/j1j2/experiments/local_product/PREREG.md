# Local-product Krylov (LP) test: pre-registration (lead, 2026-10-26 19:55, takeover; before any code or run)

**Question.** Can a product of LOCAL Krylov factors do what one global Krylov step does (fix signs at near-node configurations,
lower E_FN), with a structure whose evaluation can be made O(N) per sweep on a finite-receptive-field network?

**Ansatz.** For a guide psi (network), bond energy density eps_b(x) = (H_b psi)(x)/psi(x) for every J1 and J2 bond b, and
  LP:   psi'(x) = psi(x) * prod_b [1 + alpha_t (eps_b(x) - e_t)],   t = J1 or J2, e_t = <eps_b>_psi, alpha_J1, alpha_J2 variational;
  LPs:  the same with per-site densities eps_i = (1/2) sum_{b contains i} eps_b and one alpha (variant).
  G:    the global step psi(x) [1 + alpha (E_L(x) - E)], alpha by Ritz (the reference; our add-on).
To first order in alpha, LP = G (prod (1 + a eps_b) ~ 1 + a sum eps_b); they differ in how separate local fluctuations combine
(product vs sum) and in where the sign flips (each factor flips on its own bond).

**Prior work this must beat (RESEARCH_MAP 2026-10-13, Chatty/runs/local_krylov, commits 63c1172, 51fcbbf).** A local THRESHOLD
sign rule on 4x4 windows lost to the global rule at 8x8 from Marshall (-2.84(32)e-3 vs -3.96(24)e-3/site) and was null from the
ViT sign (window-energy noise). The global step is not diluted below ~1e3 sites for good guides. So LP cannot win on size
consistency at 8x8-14x14; its possible win is COST (O(N) instead of O(N^2) per sample) at equal gain. This test asks only
whether LP keeps the global step's gain. LP differs from the 10-13 rule: smooth multiplicative factors, alpha optimised, no threshold.

**Stage A (exact, CPU/A40, no Paderborn).** 4x4 restricted classes (f-only, RBM-only, at their VMC optima; l8_loop4x4.py) and
6x6 RBM+PP A (exact 15.8M-orbit tables, G-tax machinery). Compute E_VMC and exact E_FN for G, LP, LPs (alphas optimised for E_VMC).
Recompute G's 6x6 E_FN in the same code (reference 5.57e-6/site above E0, results/rbmpp_6x6/runs/table_A.json).
  PASS:     6x6 E_FN(LP or LPs) - E0 <= 1.0 x [E_FN(G) - E0]  -> Stage B.
  KILL:     6x6 E_FN(best LP) - E0 >  1.5 x [E_FN(G) - E0]   -> stop: locality loses what makes Krylov work.
  MARGINAL: in between -> Stage B only if the LP/G ratio of FN gains improves from 4x4 to 6x6.
Controls: alpha = 0 returns psi exactly (E_FN(psi)); a broken LP with eps_b from the WRONG bond (shifted by one lattice vector)
must lose most of the gain; the first-order identity (LP and G agree to O(alpha^2) at small alpha) checked numerically.

**Stage B (8x8, sampling, Paderborn <= 12 H100-h).** rb8_prod guide; LP alphas by correlated-sampling VMC; then 2 FN chains as M1.3b.
  PASS: E_FN(LP) <= -0.498946 + 2 sigma (G's value, +-4.6e-6) -> Stage C (separate pre-registration: CNN guide, O(N) cost, 10x10).
  KILL: E_FN(LP) > -0.498946 by > 3 sigma.
Lead's prior: Stage A PASS ~45%, Stage B PASS given A ~55%.

## Amendment 1 (lead, 2026-10-26 ~20:20, after Astra review cj-b7c4313e2b88; before any Stage A result)
- Astra [derived, checked on a synthetic 144-site guide]: for a guide with strictly local ratios (log psi = sum of patch
  functions, e.g. a CNN with finite receptive field and sum readout), the GLOBAL step G is also O(1) per proposal / O(N) per
  sweep with cached bond densities: E_L(y) = E_L(x) + sum_{b in A_c} [eps_b(y) - eps_b(x)]. So LP has no cost advantage over G
  on the same guide; the N^2 cost of Krylov is a property of global-ratio guides (RBM+PP, ViT), not of the Krylov step.
- Stage A changes (pass/kill rules unchanged): (i) broken control replaced: densities from an unrelated configuration
  (translating every bond only permutes factors, so it was not broken); (ii) added arm P_plaq = prod_p [1 + alpha sum_b w_pb
  (eps_b - e_b)]; (iii) variant with alpha_J1 = alpha_J2 (where the first-order LP = G identity holds); (iv) crossed
  diagnostics E_FN(|psi_G| sgn psi_LP) and E_FN(|psi_LP| sgn psi_G) to separate sign repair from amplitude damage.
- New Stage A' (the actual scalability question): implement G with cached local energies on a finite-receptive-field CNN
  guide; correctness: cached vs direct psi_G ratios and E_L agree to <= 1e-12 relative (broken: drop one affected bond ->
  disagrees); cost: s/sweep at L = 6, 8, 12, 16, 20 (random weights, fixed receptive field) must scale as N^a with a <= 1.2
  (direct evaluation, the control, must show a ~ 2). Cheap (GPU minutes); theorie A40 or Paderborn <= 0.5 H100-h.
