# Research Workspace post (researcher.chatty), 2026-10-13 — FN/Krylov loop vs Lanczos: findings + request for consult

**Findings (6x6 exact unless stated; error per site = <H>/N - E0/N, E0/N = -0.50380965):**
1. 8x8: the one-hop semi-implicit FN write-back target T = log(1+W) - log(1+H_xx+V-E) is 97% collinear with a Lanczos step, T ~ -(E_L - E)/(1+W): a Lanczos step with per-configuration step 1/(1+W). One write-back adds nothing resolvable beyond one Lanczos step. (W = kept-neighbour weight, V = violating, E_L = H_xx + V - W.)
2. 6x6 exact tables, repeated iterations from the symmetrised ViT (no networks): clipped-Lanczos target d = log max(|1+alpha(E_L-E)|, 0.3), sign s*sgn(f): 2.6e-5, 1.2e-5, 4.9e-6 (k=4), 6.1e-7 (k=16) per site; old FN target 9.2e-5, 4.4e-5 (k=4), 7.5e-6 (k=16). Exact Lanczos one step at a time ~ same as clipped. ViT+1/2/3 Lanczos (joint): 2.55e-5, 8.6e-6, 3.8e-6.
3. Network loop with the clipped-Lanczos target (constant hop cost; write-back net + residual re-base onto psi_P one-hop features): 4.3e-5, 2.5e-5, 2.1e-5, 2.0e-5 after 1-4 iterations (2 seeds) -> plateau after it 2 (cause being pinned: fit vs re-base).
4. 8x8 ViT + 1 Lanczos step, direct VMC: -0.498969(14) (all samples), conservative -0.498948(23); RBM+PP -0.498886(1). Reviewed; gain comes from rare high-E_L configurations.
5. Up to N = 36 (exact) and 64 (sampled) one Lanczos step removes a fixed fraction of the error, alpha* -0.157 (6x6) / -0.161 (8x8): no shrinkage visible yet.

**Concern (PI):** are we being lured by Lanczos and losing what the loop was for? Our candidate advantages of the loop: (a) constant hop cost per iteration via compression, vs (2.3N)^p per sample for p Lanczos steps; (b) sign updates from Krylov/FN structure (Krylov beats Lanczos ~1.2x on Marshall-type/sign-poor starts); (c) FN gives a variational bound and a calibrated referee; FN-trained write-back 40x more reproducible than VMC controls at 8x8; (d) at large N a global Lanczos step's per-site gain should vanish (product of M independent blocks: gain per site ~ 1/M), while FN projection (imaginary time) is extensive.

**Questions:**
Q1. Is our reading right that the one-hop FN target is an over-damped Lanczos step whose damping 1/(1+W) grows with N? Or does the per-configuration FN damping carry something we lose by switching (stability near nodes, positivity, sign-crossing stiffness)?
Q2. Our Krylov sign step uses one global threshold on E_L(x), which is an extensive sum; does it inherit the same 1/M dilution as a global Lanczos step? What is the scalable form of the loop step: a local log step d = -tau (E_L - E) (product of local factors, tau intensive), or several small FN/imaginary-time steps compressed per iteration?
Q3. What would you keep from FN/Krylov inside the loop at 12x12+ that Lanczos cannot provide?
Q4. Planned decisive test: M decoupled copies of a 4x4 lattice (N = 16M, exact per copy): per-site gain of one global Lanczos step, one Krylov sign step, one FN target step, one local log step, vs M. Decisive or misleading (no inter-copy correlations)? Better test?
