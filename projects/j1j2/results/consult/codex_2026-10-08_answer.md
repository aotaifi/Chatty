**My assessment: structured but difficult to compress remains plausible; “essentially noise” is not established.** Capacity dependence suggests approximation limits, while proposal dependence establishes a sampling bottleneck. Neither proves that 50% capture is achievable economically.

One correction matters: for stoquastic \(F\), the exact energy excess is
\[
\frac{\frac12\sum_{xy}|F_{xy}|\phi_x\phi_y
(e^{\delta_x}-e^{\delta_y})^2}
{\sum_x\phi_x^2e^{2\delta_x}}.
\]
Your \(Q\) is its quadratic expansion, assuming normalized \(\phi\). Small global RMS does not ensure accuracy in the tails; validate using the exact frozen-\(F\) energy.

1. **Test learnability with controlled learning curves.** Split by symmetry orbit, withholding all incident training edges from test orbits. Sweep data volume and capacity independently; report training/test \(Q\), resolved by amplitude stratum. Compare against targets shuffled within amplitude and weighted-degree bins, rescaled to the same \(Q\). Generalization substantially above this control demonstrates exploitable structure. Memorization without generalization suggests poor compression at the tested scales—not mathematical randomness.

   Separately, solve a regularized **global tangent-space least-squares problem** using streamed edge batches and independently validated damping. This measures accessible linearized improvement without confusing per-batch Gauss–Newton interpolation with representation failure.

2. **My first ≥50% attempt:** freeze the ViT and train a larger positive multiplicative residual with 2–4 softly gated experts, gated by guide log-amplitude. Supply configuration features **and Hamiltonian-neighborhood information**: frozen-guide amplitude ratios, FN diagonal, and pooled connected-configuration embeddings. These expose the mechanism generating the correction.

   Sample edges, not just configurations: start with \(q_{xy}\propto w_{xy}=|F_{xy}|\phi_x\phi_y\), then mix in estimated gradient-norm priorities with full-support coverage. Reweight by \(w/q\). Use a large persistent replay pool, accumulated gradients, and validation-selected damping; retain cross-gate edges. Accept only when exact energy recovers ≥50% of the FN gain.

   Pair/Jastrow residuals are useful cheap baselines; Pfaffian/backflow replacement has no demonstrated advantage for this particular correction. Lanczos distillation deserves a parallel benchmark, but check sign changes before treating it as amplitude-only learning.

3. **Closest literature:** Misery et al., [arXiv:2507.05352](https://arxiv.org/abs/2507.05352), directly addresses gradient-targeted importance sampling, including frustrated spins and projected-state fitting. Westerhout et al., [arXiv:1907.08186](https://arxiv.org/abs/1907.08186), separates generalization from expressibility in frustrated NQS. Luo–Clark, [arXiv:1807.10770](https://arxiv.org/abs/1807.10770), motivates neural backflow, but does not establish efficient compression of FN tail corrections.
