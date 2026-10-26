# 2026-10-27 ansatz angle: the sign-free pre-image tested inside the FN loop (closed)

**What was built.** A loop test of a stored, nearly sign-free state. Undo J2 on the loop's 6x6 target for a short
imaginary time, Phi_t = exp(+t H_J2) phi_0. Store |Phi_t| with the Marshall sign. Rebuild exp(-t H_J2)(stored), then two
Lanczos hops plus fixed-node. Score: refit tax against the G-tax baseline (1.657e-6/site, gate 0.82e-6). Code:
projects/j1j2/ansatz/code/c2le_gtax.py (+ c2le_expm.py, Lanczos exp). Runs: /scratch/a/A.Otaifi/ansatz_runs/c2le_6x6/.
Before it, exact 4x4/4x6 tests R1-R3 (local_undo.py, r2_recon.py, r3_local.py).

**Result.**
- Exact storage passes easily: tax 2.4e-8 at t = 0.20.
- A 2442-parameter positive network storing it fails: 7.4e-6, 4.5x the baseline.
- Supplying the exact residual sign does not help (8.0e-6).
- Closed by the pre-registered rule; an outside reviewer (Codex gpt-6-astra, 4 rounds) agreed.

**Why (hypothesis).** The rebuild is a cancellation where amplitudes are small, so fit errors grow there. The centered
log error at |phi_0|^2 decade -13 goes from 0.14 in the fit to 0.61 after the rebuild. The fixed-node energy is sensitive
exactly there. Not isolated: whether the network's size, or its loss (weighted by Phi_t, not phi_0), is what limits it.

**Alternatives rejected.**
- Sampling the rebuild: cost 3.4 -> 5.0 from 16 to 24 sites; exponential growth would not scale.
- A local product-of-bond rebuild: divides by near-zero amplitudes, 1.3-1.8x worse than the Marshall sign.
- The additive form (1 + a H_J2): its strength must grow with N.
- "J2 is special": a J1 filter is equal to first order and at most 2.2x worse.

**Three things to understand.**
1. H_J2 psi0 = (E0 - H_J1) psi0. Undoing J2 equals evolving under J1 to first order, so any claim of J2-specific physics
   must rest on second-order effects.
2. Storing a node-free object does not remove the need for accuracy near the node. It moves that need into the
   cancellation of the rebuild.
3. At a network's own energy optimum, a refit toward its one-hop state gains nothing at first order (L8, project lead).
   This test used a two-hop FN target, which has a nonzero gradient.

Plan and dead ends: projects/j1j2/PLAN_ansatz.md. Memo: projects/j1j2/ansatz/ANSATZ_MEMO.md (section "Reopened").
Reviews: projects/j1j2/ansatz/REVIEW_astra_2026-10-26.md.
