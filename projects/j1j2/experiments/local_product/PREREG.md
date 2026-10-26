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
