# Codex (gpt-6-astra, high) consult answer, 2026-10-13
On: results/consult/post_2026-10-13_loop_vs_lanczos.md + DESIGN_scalable_loop_2026-10-13.md (sections 1-4, before the fold-back section). Consultant opinion, not instructions.

Keep FN projection and compression, but the proposed sign rule and residual loss do **not yet establish a scalable loop**. The authors undervalue both optimization conditioning and what ordinary Lanczos can still deliver at accessible sizes. I reviewed the [consult](results/consult/post_2026-10-13_loop_vs_lanczos.md) and [design](DESIGN_scalable_loop_2026-10-13.md); their numerical findings are supplied evidence, not independently reproduced here.

**Q1.** Writing \(\delta=E_L-E\), the FN target is exactly
\[
T=-\log[1+\delta/(1+W)].
\]
Its linearization is the stated diagonally preconditioned imaginary-time step. If \(W\propto N\), the effective step decreases as \(1/N\). Calling it “over-damped” additionally requires evidence that a larger step remains stable.

The nonlinear denominator matters near small guide amplitudes, where ratios become large; the update preserves positivity wherever its denominator is positive. Ordinary Lanczos instead permits sign crossings. Thus 97% collinearity supports redundancy for typical configurations, but does not settle tail behavior—especially when your measured improvement comes from rare configurations. Compare energy contributions and ratio stability in those tails.

**Q2.** Global thresholding couples each block’s decision to fluctuations from all other blocks, so dilution is real. But **\(1/M\) is not the generic optimized-Lanczos scaling**. For identical independent blocks with variance \(v>0\) and third central spectral moment \(\mu_3\), the exact first Krylov matrix gives
\[
\Delta E_{\rm total}
=\frac{c-\sqrt{c^2+4Mv}}2,\qquad c=\mu_3/v.
\]
Thus optimized gain per site scales as \(M^{-1/2}\); fixed nonzero \(\alpha\) gives \(M^{-1}\). Neither exponent automatically describes sign-only thresholding. These are analytical statements about product states.

The update \(a'=a\exp[-\tau(E_L-E)]\) factorizes on independent blocks, with intensive \(\tau\), but preserves signs and is only a first-order imaginary-time approximation on coupled systems. I would favor several controlled projection steps with compression, while treating sign improvement separately. Fixed-order Lanczos is asymptotically inconsistent, yet published calculations retain useful gains through hundreds of sites. [Becca et al., arXiv:1412.2656](https://arxiv.org/pdf/1412.2656)

**Q3.** Keep FN’s positive projection, its upper bound, and its frozen-Hamiltonian benchmark for measuring compression loss. The bound concerns the original ground energy; it does not certify signs, correlations, or distance to the ground state. VMC itself is also variational.

Compression is potentially the central contribution: it prevents repeated projection from multiplying evaluation cost. Keep Lanczos as a control and final polish. Sorella’s generalized approach already combines amplitude and sign improvement with a potentially size-consistent effective dynamics; it deserves a substantive comparison. [Sorella, cond-mat/0009149](https://arxiv.org/pdf/cond-mat/0009149)

**Q4.** Decoupled copies are a necessary algebraic test, not a predictor of the crossover size in the coupled Heisenberg model. Reoptimize each method at every \(M\), and distinguish global sign-only, full Lanczos, and compressed updates. Then introduce inter-block coupling continuously. This tests whether the advantage survives entanglement and boundaries. A product-block \(N_{1/2}>64\) does **not** make an 8×8 null “expected.”

**The local sign rule is a plausible heuristic, not a derived Krylov step.** Partition parity follows exactly from tensor-product factors. Connected-component parity does not: two negative factors require even parity, whereas merging their marked windows produces one component and reverses that answer. Agreement between window sizes cannot establish correctness.

Moreover, \(e_b(x)\) can depend on distant spins through the neural amplitude ratio. Geometrically local windows do not guarantee local information. Nor does one extended component imply one sign error. The decoding analogy needs an actual defect algebra. NLCE achieves extensivity through linked-cluster subtraction, not marked-component parity. [arXiv:1810.06202](https://arxiv.org/pdf/1810.06202)

Test deliberately inserted one- and two-defect patterns, vary their separation through the merging distance, and measure energy-weighted relative-sign errors. Retain fixed tilings as a diagnostic on unreduced configurations; orbit-table convenience should not eliminate the mathematically controlled baseline.

**The FN equation is correct; several claims about its loss are not.** With frozen guide, exact energy, connected support, and a positive solution, solving everywhere identifies the FN ground amplitude. Exact fixed-sign updates obey the proposed energy inequalities. The lattice upper-bound theorem supports this foundation. [ten Haaf et al., cond-mat/9412037](https://arxiv.org/abs/cond-mat/9412037)

However:

- \(D-S_u\) adds across independent blocks; \(1-S_u/D\) and \(\log D-\log S_u\) do not. A defect confined to one block has normalized residual \(O(1/M)\). The proposed loss can hide precisely the local errors the design aims to repair.
- One snapshot supplies one summed equation, not one independent constraint per edge. Four million evaluated edges are approximately thirty thousand residual equations; erroneous ratios can compensate.
- Deterministic residual evaluation does not eliminate sampling, energy-estimation, approximation, or optimization error. Biased walker weights affect the optimum whenever exact fitting is impossible.
- Centered residual minimization under frozen walker weights is not the usual quantum variance under the candidate’s squared amplitude.

**The checks are useful but incomplete.** Keep ORACLE, RES-W versus RES-V, and equal-cost minSR. Add disconnected-copy tests of the *actual fitted loss*, exact versus sampled residual fitting, and sign-only/amplitude-only/full-loop ablations. Validate the learned sign network separately from its labels. Measure structure factors and dimer correlations alongside energy.

“Non-increasing within \(2\sigma\)” permits cumulative deterioration; repeated threshold selection also needs an untouched final evaluation. Calibrate injected energy errors to each harvest’s actual uncertainty.

My recommendation is to establish amplitude capture and loss conditioning first, while running cheap adversarial sign tests. The current evidence supports compression as the opportunity; it does not yet support cluster parity as the missing principle.
