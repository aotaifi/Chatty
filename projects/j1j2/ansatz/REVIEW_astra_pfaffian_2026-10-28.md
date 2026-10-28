# Second opinion on the Pfaffian angle: Codex gpt-6-astra, xhigh, read-only (job cj-d28d6046530c, 2026-10-28)

Asked by the Pfaffian angle lead at the PI's request: expressibility vs learnability of the complex PP + 3x3 backflow copy. Prompt: /scratch/a/A.Otaifi/codex_jobs/cj-d28d6046530c/. Spot-checked by the angle lead: conn.json (one component, 15,804,956 orbits) and cx_c1fit.py bonds_of (bond partners y not restricted to training orbits) confirmed.

[Derived] The storage problem is preserving amplitude ratios on blocked configuration-space edges, to the accuracy required by FN projection.  
[Derived] Expressibility concerns the best possible parameters; training records reveal only parameters actually found.  
[Opinion] The decisive evidence would be a successful same-class copy or a certified approximation lower bound.

[Opinion] **The remaining error is unresolved; I see moderate evidence for a learnability component, but no established expressibility floor.**

[Checked] The three complex copies have tax₀ = \(1.107,\ 0.912,\ 0.786\times10^{-5}\)/site; all beat the recorded real reference \(1.80\times10^{-5}\). However, C doubles the real parameter count, its real reruns did not converge, and its mechanism criterion saturated. Improvement survives; attribution does not. Sources: [C results](/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs), [pre-registration](/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/PLAN_pfaffian.md:77).

- [Checked] **Perron–Frobenius survives for this sector:** [conn.json](/scratch/a/A.Otaifi/ansatz_runs/pfaffian_g/conn.json) records one connected kept-graph component containing all 15,804,956 orbits. [Derived] Connectivity plus nonpositive off-diagonals gives a strictly positive ground state for finite guide ratios. [Checked] Finite stored logarithms alone would not establish this: the numerical code floors amplitudes.

- [Derived] **The nodal argument survives only as a heuristic.** On a finite configuration set, generic real and complex parameters both avoid exact zeros. Opposite signs at adjacent configurations do not require either endpoint to be small: \(A(\pm1)=\pm1\) gives the exact positive modulus \(1\). Codimension counting requires a continuous extension and regularity assumptions; it supplies no approximation lower bound. [Checked] Thus q’s \(0.424\) versus \(0.069\) establishes association, not necessity or causation ([q JSON](/scratch/a/A.Otaifi/ansatz_runs/pfaffian_q/q_ownnode.json)).

- [Derived] **The pair formula survives, but its interpretation needs care.** With \(e=\log(A/\phi)\),
  \[
  C_\phi=\sum_{\{x,y\}\in B}2|H_{xy}|\phi_x\phi_y[\cosh(e_x-e_y)-1],
  \qquad E_{\rm FN}[A]\le R_s(\phi)+C_\phi .
  \]
  Here \(R_s\) is the physical Rayleigh quotient with signs \(s_2\). [Checked] For this target, \(R_s(\phi)-E_{\rm FN}[\phi]=1.6262\times10^{-6}\)/site ([split JSON](/scratch/a/A.Otaifi/ansatz_runs/pfaffian_d/split/split_s1.json)). [Derived] Consequently, \(C_\phi\) is neither the tax nor its additive decomposition. F’s 35% splice improvement does not establish that precisely 65% “sits in the bulk.”

- [Derived] **The eigen-equation split survives; “one-signed local offset” does not follow.** Writing \(D_g=H_{xx}+V_g\),
  \[
  w_{xz}=\frac{|H_{xz}|\phi_z}{(D_g(x)-E)\phi_x},\quad
  e_x=\log\sum_{z\in K_x}w_{xz}e^{e_z}+\rho_A(x),\quad \sum_z w_{xz}=1 .
  \]
  A small logarithmic neighbour average permits individual errors to cancel. Moreover, \(\|e\|^2=\|I\|^2+\|\rho\|^2+2\langle I,\rho\rangle\); \(s_{\rm II}\) is not an orthogonal explained-variance fraction and depends on normalisation. [Checked] The [M-J JSON](/scratch/a/A.Otaifi/ansatz_runs/pfaffian_mj/mj_split.json) reports weighted means, not evidence that errors share one sign.

- [Checked] **Coverage is deficient, but not absent:** deep configurations and “held-out” orbits can enter training as bond partners \(y\), because [bonds_of](/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/cx_c1fit.py:309) excludes neither. [Derived] R’s train/held-out comparison therefore cannot cleanly diagnose memorisation. Nor does counting the \(e_x^2\) loss term establish wasted effort: it indirectly controls edge differences.

[Opinion] q, C, D, E and F are justified diagnostics, but their causal conclusions exceed their controls. [Derived] T2 measures the subsequently withdrawn two-hop pipeline, not the present tax₀ question. J changes both the function class and—relative to M-J—the operator from \(H_{\rm FN}[g]\) to \(H_{\rm FN}[A]\); it cannot separate expressibility from learnability. R tests an oracle depth correction; success would still require replacing \(\phi\)-depth by available inputs. Sources: [PLAN.md](/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/PLAN.md:132), [J/R definitions](/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/PLAN_pfaffian.md:239).

[Derived] **Test G is useful but insufficient.** It optimises a moving Hamiltonian’s energy, not reproduction of the original \(\phi\). Success proves that the energy error was reducible within the class; failure proves no floor. Two starts can share a basin, particularly when descended from the same real parent. Full sampling support is not adequate gradient coverage. Flat energy is not stationarity, much less global optimality.

[Checked] [G’s implementation](/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/cx_g1fit.py:1) uses sampled inner gradients and exact outer majoriser checks; that supports monotonicity when solves are accurate. [Derived] Pure gauge normalisation is harmless; amplitude floors, active constraints, inaccurate eigensolves and vanishing steps can produce misleading plateaus.

[Opinion] **The cheapest useful separator is a constructive witness:** keep \(\phi\), \(s_2\), and the complex 3×3 class fixed; fit \(C_\phi\) with importance-corrected sampling covering every blocked edge, then evaluate the full-space loss and tax₀. Include a small matched-range teacher–student control whose target comes from known parameters of this same class.

[Opinion] Budget roughly **1.5–2 H100-hours plus one CPU-hour** for a 400-step attempt, teacher control and readout. [Checked] This extrapolates C’s approximately 6.7 seconds/step and 533 seconds/table ([timings](/scratch/a/A.Otaifi/ansatz_runs/pfaffian_c/runs/c1_cplx_s1/cx_c1.json)). [Opinion] My forecast is a partial 20–40% tax reduction, not elimination. [Derived] A witness with \(|\mathrm{tax}_0|<2\times10^{-6}\)/site and \(C_\phi\) reduced tenfold decisively refutes a floor near today’s value. A genuine floor forbids that witness; unsuccessful training remains inconclusive, even if the teacher control passes. Certifying expressibility failure requires a lower bound over the nonlinear class—no proposed test provides one.

[Derived] **The structure demands a positive scalar potential with consistent ratios, not a particular architecture:**
\[
u_x=\log\phi_x
=\operatorname{LSE}_{z\in K_x}(\log|H_{xz}|+u_z)-\log(D_g(x)-E).
\]
This is a global eigenfunction equation; a local exchange does not imply local dependence on spins. Representing \(A=e^u\) enforces positivity and \(\sum_{\rm cycle}(u_y-u_x)=0\).

[Opinion] My implementable candidate is a positive, guide-relative neural Jastrow \(A=g\,e^{R_\theta}\), with global symmetry-tied real RBM features for \(R_\theta\). [Derived] It supplies nonlocal corrections without moving determinant zeros, and \(\partial_\theta\log A=\partial_\theta R_\theta\). This motivates the class, not a claim that a small RBM suffices. [Checked] Factorised residuals already appear in [B10/B11](/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/TRIED_IDEAS.md:64).

[Derived] With \(M=\alpha N\) hidden units, the residual costs \(O(NM)\) per fresh amplitude, \(O(M)\) per cached exchange ratio, and \(O(NM)\) for all FN transition rates, **plus the guide and sign costs**. An explicit Jacobi layer instead requires \(O(N)\) base evaluations per amplitude and generically \(O(N^2)\) per walker step.

[Checked] The same logarithmic eigen-equation is Gispen–Lamacraft, [arXiv:2012.07063](https://proceedings.mlr.press/v145/gispen22a/gispen22a.pdf), equations 16 and 30–32; the RBM positive-product form and cached ratios are Carleo–Troyer, [arXiv:1606.02318](https://arxiv.org/html/1606.02318v1), Appendix C.

[Opinion] Stop calling plateaus floors, interpreting bound shares as tax shares, using tax₂ to justify this pipeline, and selecting larger windows from an error histogram alone.

[Checked] I ran a 20-second read-only NumPy audit: after normalising both amplitudes, only 43–44% of deep orbits had negative errors, by orbit count—not cost weight. I also recomputed the Rayleigh offset above; no files changed.