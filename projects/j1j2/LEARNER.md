# Learner profile: j1j2 (FN-Lanczos loop)

Read by the tutor skill and by any lead discussing the loop with the PI. General profile: /project/theorie/a/A.Otaifi/LEARNER.md.
Started 2026-11-17 by the FN-loop lead after a long pedagogical discussion (theorist mode). Owned by the PI; Claude proposes edits.

## Established with the PI (2026-11-17), in the PI's own understanding
- Log loss: d ln B/d theta = (dB/d theta)/B. The failure at 8x8 is the 1/B, not a large slope: B is small while dB/d theta is ordinary.
- B is small at those configurations because the 256 symmetry copies of the pair product cancel (kappa = |sum t_g| / sum |t_g| ~ 0.005).
  The copies cancel, their gradients do not, so |d ln B/d theta| ~ 1/kappa. Smaller steps cannot fix it (the bulk would freeze).
- Our PP is the Gutzwiller-projected singlet pair product det f[U,D], a special case of a Pfaffian; a spin exchange changes a row AND a column.
- Frozen base: model B_0 e^{r}; ln of a product is a sum, ln B_0 is a constant, so the log loss on r has no 1/B in its gradient.
- Plain square loss (T - copy)^2 removes 1/B but under-weights near-node configurations by phi(y)/phi(x); the relative square loss keeps
  the weight and drops the log's curvature.
- The bounded ratio: FN equation at x gives phi(x)/psi_T(x) = K_phi / ((D - E) psi_T(x) + V_s); the tiny psi_T(x) cancels, so the ratio is
  bounded. e^{c f_J} = R_est^c, R_est the one-step estimate of that ratio. The exponential only makes the factor positive.
- Product form is an identity (|phi| = |psi_T| x ratio), not an assumption. Its two gaps: guide zero in the bulk (ratio infinite) and
  isolated configurations (phi = 0, ratio 0). Isolated ones are repaired by the next C-free sign step; near-zero dips are lifted round by round
  (heads accumulate), at most e^{4.88} per round.
- Big picture: psi_k = S_k |B_0| e^{r_0 + ... + r_{k-1}}; the sign from Lanczos steps, the magnitude from accumulated FN corrections.

## Shaky or open for the PI
- Q1 (is B_0 a floor?): not yet understood. Plain version: the heads must learn ln(phi/B_0); dips of B_0 make that a spike.
- Whether the frozen product is "enhancing the network": PI doubts it; lead holds ~60% with the middle way (periodic base refit).
- Why c ~ 4 (over-relaxation) rather than 1.

## How explanations landed here
- Asked for "step by step, derived" and each step's equation; a step skipped ("you stopped a step") lost the PI.
- Caught circular-looking arguments (exponential form, ratio) and wanted the order: identity, derivation, consequence, parametrisation.
- Corrected the lead's notation (K_phi = sum -H_xz phi_z) and wants one fixed notation; mixed phi/varphi confused the PI.
- Asked "why do you call them spurious?": loaded words need a definition at first use (spurious, heads, dips).
- Spotted the lead reaching for a measurement where algebra could answer; wants derivations first.
