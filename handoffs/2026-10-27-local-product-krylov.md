# Handoff 2026-10-27: local-product Krylov test (takeover), killed

**What was built.** experiments/local_product/: exact 4x4 and 6x6 comparison of one global Krylov step
G = psi (1 + alpha (E_L - E)) with products of local Krylov factors (per bond, per site, per plaquette), and gcache.py, a cached
implementation of G on a finite-receptive-field CNN. Pre-registration: experiments/local_product/PREREG.md (+ Amendment 1).

**Why.** The PI wanted Krylov's sign repair made scalable. A global step costs ~N^2 network evaluations per sample on global
networks; a product of local factors would be size-extensive and linear-cost.

**Result.** KILL: the local products keep ~0 of the global step's fixed-node gain at 6x6 (3.5247e-5 vs 5.5738e-6 per site).
Report: projects/j1j2/docs/reports/2026-10-27_local_product_krylov.md.

**Alternatives rejected.** Local threshold sign rules (tested 2026-10-13, lost at 8x8); per-site and plaquette products (fail
the same way); a learned local function of the densities (no guarantee it reproduces a threshold on a global sum; not run).

**Three things to understand.**
1. The global step adds alpha (H - E) psi, which has no pole where psi -> 0. A product of factors [1 + alpha eps_b] divides by psi
   once per active bond, so at the near-node configurations (exactly where the sign repair matters) it explodes, and the
   variational optimum switches the factors off.
2. On a guide whose ratios are strictly local (CNN, finite receptive field), the global step itself is linear-cost with cached
   bond energies; on RBM+PP or ViT it is not. The constant overhead is ~the number of bonds a move can affect (hundreds).
3. An outside reviewer (Astra) predicted both points before the run; the 4x4 control looked fine and only 6x6 exposed the pole.
