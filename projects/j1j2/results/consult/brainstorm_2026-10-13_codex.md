# Codex (gpt-6-astra, high) brainstorm on OBSTACLE_2026-10-13.md
Consultant opinion, not instructions.

**I would first test storing an upper-bound Hamiltonian instead of a wavefunction.** The shared assumption worth dropping is that every useful improvement must survive deterministic compression into an independently evaluable amplitude function. Facts 1–5 establish failures of particular compressions, not an impossibility theorem.

Ranked below by my judgment of practical promise. These are proposals, not validated results; I take the measurements in [the obstacle](obstacle_2026-10-13.md) as given.

1. **Store the fixed-node wall directly, without requiring globally consistent amplitude ratios.**

   **Idea:** Fix a learned sign pattern and replace each positive off-diagonal edge \(h>0\), in that sign gauge, by diagonal penalties \(hr\) and \(h/r\), with \(r>0\) predicted by a fixed-size edge network. Optimize this effective Hamiltonian using its projected ground-state occupations, rather than distilling its ground-state amplitudes.

   **Why it could work:** On each edge,
   \[
   H_{\mathrm{eff}}-H
   =h\begin{pmatrix}r&-1\\-1&1/r\end{pmatrix}\succeq0.
   \]
   Thus the ground energy remains an upper bound even when the edge ratios do not integrate to any wavefunction. This is a derived extension of the [lattice FN upper-bound construction, cond-mat/9412037](https://arxiv.org/abs/cond-mat/9412037): it targets fact 4 directly, uses fact 5 for signs, and avoids the amplitude storage required by facts 1–3.

   **Risk:** Useful edge penalties may be just as difficult to represent; accurate ground-state occupations also cost projection time.

   **Cheapest exact kill:** Solve the existing 6×6 FN Hamiltonian for \(\Phi\). Fit a cheap edge head to minimize
   \(\sum_{\mathrm{bad}}h(r\Phi_x^2+\Phi_y^2/r)\), then recompute its lowest eigenvalue. Reject that head if it cannot capture appreciable improvement even with exact occupations.

2. **Store signs; let a walker population carry the amplitudes.**

   **Idea:** Iterate only the sign network while retaining a fixed, evaluable amplitude guide. For each sign update, project the corresponding FN Hamiltonian with a bounded walker population; never distil the projected amplitudes.

   **Why it could work:** Fact 5 already demonstrates useful sign compression, while facts 2–3 concern the harder amplitude problem. FN projection sums arbitrarily many imaginary-time steps with fixed work per propagation step; it does not require recursively evaluating earlier networks.

   **Risk:** Fact 4 remains fully operative: frozen guide amplitudes produce a biased wall. This could yield useful improvement yet plateau well above the desired accuracy; population and equilibration costs must also be counted.

   **Cheapest exact kill:** On 6×6, build and diagonalize FN Hamiltonians for the first few sign iterates with identical frozen amplitudes. If the sign improvements do not lower the exact FN energies, reject the proposed loop before implementing walkers; an exact-sign arm diagnoses how much amplitude bias remains.

3. **Change the compression objective from accurate ratios to retained energy descent.**

   **Idea:** Fit the signed additive increment in amplitude space, sampling both the current state and its residual \(r=(H-E)\psi\). Select updates by their Rayleigh quotient, using a bounded training budget and a line search, rather than by log-ratio accuracy or average target agreement.

   **Why it could work:** Fact 2 localizes lost improvement in tails and shows that supplying those tails restores 91%; this leaves an objective/coverage failure plausible. Fact 3 concerns logarithms near nodes, which can be singular even when signed amplitude corrections are representable; fact 4 demands sensitivity to precisely those errors.

   **Risk:** The networks may genuinely lack capacity. This is related to [supervised wavefunction optimization, arXiv:1811.12423](https://arxiv.org/abs/1811.12423), and could simply become expensive ordinary NQS training.

   **Cheapest exact kill:** Repeat only the failed second iteration with exact residual-enriched training data and the same architecture. Evaluate the full-table Rayleigh quotient after an exact one-dimensional line search; reject the fixed-budget proposal if retained gain remains poor despite removing sampling coverage errors.

4. **Keep a bounded dictionary of explicit amplitude exceptions.**

   **Idea:** Represent the state as a compact network plus signed corrections on at most \(M\) symmetry orbits. Refresh that dictionary using residual importance, retaining inherited exceptions rather than regenerating every old tail through a wrapper.

   **Why it could work:** The 91% exact-tail splice in fact 2 is unusually direct evidence for this route. Orbit corrections accommodate arbitrary nonlocality from fact 3 and repair the ratios relevant to fact 4 without additional Hamiltonian depth.

   **Risk:** Eight percent of probability weight need not mean few configurations. The necessary dictionary could grow with iterations or exponentially with system size.

   **Cheapest exact kill:** Rank candidate exceptions using the exact failed-step residual, then solve the generalized Ritz problem in the span of the base state and selected orbit vectors for several affordable \(M\). This gives the optimal energy for each chosen support; reject the support strategy if even that optimum fails to retain the gain or requires a rapidly increasing \(M\).

5. **Store a small variational subspace, not one compressed iterate.**

   **Idea:** Maintain a fixed bank of independently evaluable states and optimize their linear combination through a small generalized eigenproblem. Replace one bank member per iteration with a trained residual direction, keeping the other members intact.

   **Why it could work:** Earlier tails survive in retained basis states, addressing the inheritance failure in fact 2. Destructive interference between components can create complicated nodes and nonlocal log-ratios without requiring each component to reproduce them; fact 3 therefore does not rule this out.

   **Risk:** The correction sequence may have high effective rank. A growing bank or basis members that reference previous states would merely hide the original cost growth.

   **Cheapest exact kill:** Form exact overlap and Hamiltonian matrices for a small bank of existing evaluable candidates. Compare its optimal Ritz energy with the exact-step target; if useful gains appear only when uncompressed historical targets are added, this affordable dictionary has failed.

6. **Replace deterministic compression by unbiased stochastic sparsification.**

   **Idea:** Apply linear imaginary-time propagation and randomly compress its signed correction to a fixed number of entries, using the guide as a control variate. Preserve amplitudes in expectation instead of preserving every tail in each realization.

   **Why it could work:** This drops the deterministic storage demand behind facts 2–3; [fast randomized iteration, arXiv:2005.00654](https://arxiv.org/abs/2005.00654) supplies a relevant precedent. Fact 5 suggests a learned sign guide could reduce, although not eliminate, cancellation.

   **Risk:** Fixed work per step does not imply fixed statistical accuracy. Sign cancellation may overwhelm the gain; unbiased amplitudes do not make energy ratios unbiased.

   **Cheapest exact kill:** Randomly compress the exact second-step correction at affordable populations and evaluate each resulting state’s exact Rayleigh quotient. Reject if compression noise consumes the energy gain; these snapshot energies are variational, whereas projected estimators require separate bias control.

For every route, hold representation size fixed and count training, projection, sampling and final evaluation against both explicit \(p\)-step Lanczos and equal-cost NQS optimization. The proposed kills reject concrete budgets and representations, not entire method classes. Literature references above were checked at abstract level.
