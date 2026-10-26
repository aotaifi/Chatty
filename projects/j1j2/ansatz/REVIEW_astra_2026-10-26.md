# Second opinion: Codex gpt-6-astra, xhigh, read-only (job cj-baa5ea3230cf, 2026-10-26)

Asked by the ansatz angle lead after the R1 4x6 PASS. Prompt: /scratch/a/A.Otaifi/codex_jobs/cj-baa5ea3230cf/. Its 4x4 numbers are UNVERIFIED until R2 (a) reproduces them.

[Checked] R1 passes its registered two-size rule.  
[Derived] That establishes neither J2-specific physics nor affordable reconstruction inside the FN loop.  
[Opinion] I would test reconstruction and residual-sign handling before spending on another network fit.

1. **Is R1 sound?**

   [Checked] The [fine-grid files](/scratch/a/A.Otaifi/ansatz_runs/recheck/lu_4x6_fine.json) and logs confirm \(t^*\in(0.14,0.16]\) at both sizes, hence a ratio below 1.143; the LM-overlap gate also passes.

   [Checked] I ran additional exact 4×4 checks, using under three CPU-seconds. At matched squared overlap with \(\psi_0\), \(F=0.907323\):

   | Evolution | \(W_M(\Phi)/W_M(\psi_0)\) |
   |---|---:|
   | \(e^{+tH_{J2}}\), \(t=0.16\) | \(8.80\times10^{-4}\) |
   | \(e^{-uH_{J1}}\), \(u=0.18487\) | \(1.92\times10^{-3}\) |
   | \(e^{-uV_{J1}}\) | \(0.0187\) |
   | \(e^{+uV_{J3}}\), unique distance-two bonds | \(0.0871\) |
   | \(e^{+u(V_{J1}+V_{J2})}\) | \(0.283\) |

   [Checked] J1 reaches the \(10^{-3}\) threshold at \(u=0.19172\), versus J2 at \(t=0.15918\). [Derived] J2 specificity survives quantitatively, but the strongest control is only 2.18× worse at matched fidelity. Indeed,
   \[
   H_{J2}\psi_0=(E_0-H_{J1})\psi_0,
   \]
   so the two normalized evolutions have **identical first-order changes**.

   [Derived] Positivity-improving evolution generically removes signs eventually on finite connected sectors with nonzero overlap with the positive extremal state. Size-independent cleaning time is not guaranteed, but finite-size sign removal itself is weak evidence.

   [Derived] \(W_M\) is a probability of a global sign, not an extensive defect count. For independent sign-defect regions of density \(q\),
   \[
   W_M(N)=\frac{1-(1-2q)^N}{2}.
   \]
   Thus \(W_M/N\) eventually vanishes even without improvement. Report both total \(W_M\) and
   \(\gamma_M=-N^{-1}\log|1-2W_M|\), whose extensivity interpretation requires approximately independent regions. A fixed nonzero residual density eventually makes the total-weight suppression ratio approach one.

   [Checked] The metric also changes its sampling measure: at \(t=0.16\), 4×4 has \(W_M(\Phi)=1.12\times10^{-5}\), yet the remaining wrong-sign configurations carry \(2.00\times10^{-3}\) of \(|\psi_0|^2\). [Opinion] Add original-target-weighted violations and FN-relevant edge errors; the current metric can reward hiding troublesome configurations.

   [Derived] Suspicion (c) has the spectral direction reversed: \(e^{+tH_{J2}}\) selects **maximal** J2 energy—ferromagnetic order within each Néel sublattice. In the singlet sector its endpoint is LM. [Checked] My 4×4 check gives \(\langle H_{J2}\rangle/N:0.07670\to0.15265\), toward the maximum \(0.25\), and normalized \(\langle S_A^2\rangle:0.5125\to0.7028\). [Derived] Falling global LM overlap with size does not exclude growing local order. [Opinion] Track these intensive observables and \(-\log F_{\rm LM}/N\), not only \(F_{\rm LM}<0.7\).

   [Derived] Both samples retain width four; moreover each 4×4 J2 component is the unusually symmetric graph \(K_{4,4}\). They are poor evidence for a two-dimensional size limit.

   [Checked] [local_undo.py](/project/theorie/a/A.Otaifi/Chatty/organize/projects/j1j2/ansatz/code/local_undo.py) has consistent Marshall signs, \(+tH_{J2}\), and \(J_2=0.5\) included. Reversing \(t\) worsens violations 7.03×; full-\(H\) evolution leaves them unchanged. [Derived] Gate inverses are individually proportional to exponentials of the same bond operators, so that arm is closely related evidence; its fixed ordering also needs symmetry/order checks.

2. **Novelty**

   [Checked] Relevant precedents exist: [Sorella’s generalized Lanczos construction](https://arxiv.org/abs/cond-mat/0009149), and [Dobrautz–Luo–Alavi’s nonunitary Gutzwiller transformation](https://arxiv.org/abs/1811.03607), which produces compact transformed Hubbard eigenvectors. These claims were checked against abstracts and the repository’s literature audit.

   [Checked] [Hastings–Wen quasi-adiabatic continuation](https://doi.org/10.1103/PhysRevB.72.045141) requires a gap or suitable low-energy-density conditions. [Derived] Neither it nor a Trotter decomposition guarantees a Marshall-positive preimage under this particular generator.

   [Opinion] I am **not sure** an earlier paper states the precise size-uniform J2-only proposition; the targeted search did not establish one. The potentially novel result is a quantitatively useful, size-uniform positive preimage—not applying imaginary-time evolution to a variational state.

3. **Scaling**

   [Derived] Write \(\Phi_+(x)=s_N(x)f(x)\), \(f\ge0\), and \(T=e^{-tS_XH_{J2}S_X}\ge0\). Then
   \[
   \psi(y)=s_X(y)\sum_xT_{yx}s_X(x)s_N(x)f(x).
   \]
   The propagator is positive, but its boundary vector carries alternating signs.

   [Derived] Define the endpoint cancellation ratio
   \[
   \eta_y=\frac{|\psi(y)|}{\sum_xT_{yx}f(x)}.
   \]
   Even with ideal importance sampling, relative error \(\epsilon\) requires approximately \(\epsilon^{-2}\eta_y^{-2}\) independent samples. With \(C_f\) the network-evaluation cost, an optimistic continuous-time cost is
   \[
   O\!\left((Nt+C_f)\epsilon^{-2}\eta_y^{-2}\right).
   \]
   If \(\eta_y\sim e^{-cNt}\), this is exponential; \(c\) is unmeasured, not a theorem about this state.

   [Checked] At 4×4, \(t=0.16\), the \(|\psi_0|^2\)-weighted median \(\eta\) is 0.694 overall but 0.0412 for configurations with \(|\psi_0(x)|^2<10^{-7}\). [Derived] That tail median implies roughly six million independent samples for 1% relative amplitude accuracy.

   [Derived] Trotterization does not automatically solve contraction. A generic callable positive network leaves exponentially many endpoints; with a locally factorized tensor representation, straightforward exact boundary contraction costs \(\operatorname{poly}(N,m)e^{O(Lm)}\) for \(m\) layers. A local-observable light cone does not bound a global amplitude’s relative error.

   [Derived] A convergent connected-cluster expansion could cost \(O(Nb^kC_f)\), with \(k=O(\log(N/\epsilon))\) under uniform geometric convergence. No such convergence bound has been established here, especially near cancellation zeros.

4. **Does the next step meet the PI’s bar?**

   [Opinion] The 6×6 R1 run tests finite-size robustness; C2-L addresses the actual objective. My strongest bet is that cleaning survives while reconstruction cost or refit amplification defeats the loop.

   [Derived] From the recorded spectra,
   \[
   \kappa(e^{-0.16H_{J2}})=4.95,\ 10.37,\ 28.71
   \]
   at 4×4, 4×6 and the existing 6×6 symmetry sector. Fixed local time therefore does not mean bounded global conditioning; this is a worst-case warning, not measured fit amplification.

   [Derived] “The loop carries the residual sign” needs an explicit algorithm:
   \[
   s_{\Phi}=\operatorname{sign}(e^{tH_{J2}}\psi_{\rm FN})
   \]
   generally differs from the physical FN sign. Reading \(s_\Phi\) from the exact table would conceal the unresolved handoff.

   [Opinion] **Cheapest next kill test:** a 4×4 loop round with exact preimage magnitudes but the proposed **oracle-free residual-sign rule**, followed by reconstruction, two hops and FN. Estimated cost: under one CPU-minute. Failure even without magnitude-fitting error kills that implementation; it does not kill every positive-preimage hypothesis.

   [Opinion] Before 6×6 output, preserve R1’s historical PASS but amend further claims: require a matched-fidelity J1 control, intensive sign/order diagnostics, and reconstruction-error checks. Loop success must retain \(\mathrm{tax}_2<0.82\times10^{-6}\), with matched total cost including exponentials and sign evaluation.

5. **Dead ends**

   [Checked] C1’s raw data give \(R/R_{\rm null}=0.99015\). [Derived] That establishes failure of those parity classes to predict violations; it **does not establish spatial locality**, a premise R1 repeatedly borrows.

   [Checked] The revised records retain C4’s missing second size, C5’s inconclusive ratios, and restricted parent/similarity families for C6/C6b. [Opinion] Their broad obituary is unjustified; these are limited tests, not exclusions of entire mechanisms.

   [Checked] Old C2-L’s positive-RBM tax was \(5.66\times10^{-5}\). [Opinion] That implementation deserved rejection, but it did not test the newly proposed residual-sign loop. Likewise, one-shot sign failures alone do not settle the PI’s revised bar.

**Top three risks, ranked**

1. **[Opinion] Reconstruction remains exponentially expensive.** Test \(\eta_y\) on FN-relevant edges versus size; add the absolute-kernel exponential to the planned 6×6 calculation.
2. **[Opinion] Residual-sign handling or refit amplification consumes the gain.** Run the exact-magnitude 4×4 loop test first, then matched fits measuring actual tax.
3. **[Opinion] R1 measures generic filtering on narrow tori.** Run matched-fidelity J1 controls at 6×6 alongside \(\gamma_M\), sublattice order and target-weighted violations.
---
# Debate round 1 (Codex gpt-6-astra xhigh, job cj-f4d4b2023b16, same Codex session), 2026-10-26

[Checked] R2 supports its registered classifications; it has not established a benign, learnable stored object.  
[Derived] R3 is a nonlinear product approximation whose leading error includes correlations between disjoint bonds.  
[Opinion] Test it, but neither its current PASS nor its failure would settle the stored-preimage route.

**R-a — agree conditionally; disagree with the locality inference.**

[Derived] Under your independent-defect model, \(Nq\ll1\) gives \(W_M\simeq Nq\), so the overlay supports a size-independent suppression factor. But independence and spatial locality are assumptions, not consequences of that overlay.

[Derived] A nonlocal alternative is a rare collective sign sector whose weight is suppressed by an \(O(1)\) spectral separation:
\[
W_N(t)\simeq W_N(0)e^{-2\Delta t}.
\]
Its ratio also overlays without independently identifiable local defects. Two systems sharing width four cannot distinguish these mechanisms.

[Opinion] Call the overlay evidence for an intensive suppression scale. Establish locality through connected double-swap responses versus separation.

**R-b — agree for the oracle round trip; disagree for a genuine loop test.**

[Checked] R2 confirms exact reconstruction at 4×4 for \(t\ge0.18\). [Derived] Starting from exact \(\psi_0\), retaining exact magnitudes and applying the exact inverse then makes that test tautological.

[Opinion] My previous recommendation insufficiently distinguished this from testing the actual FN target, callable evaluator and fitted magnitudes. Use 4×6 for the residual-sign question; 4×4 can still decisively expose evaluator or fitting failures.

1. **What does the independent-bond product approximate?**

   [Derived] Let \(f=\widetilde\Phi\), \(e_b=(h_bf)/f\), and
   \[
   \Gamma_{bc}(y)=
   \frac{[(h_bh_c+h_ch_b)f](y)}{2f(y)}-e_b(y)e_c(y).
   \]
   At fixed \(f\),
   \[
   \frac{\psi_L-e^{-tH_{J2}}f}{f}
   =-t^2\sum_{b<c}\Gamma_{bc}+O(t^3).
   \]
   Thus it is first-order correct, but its error is **not merely a Trotter commutator**. Even disjoint, commuting bonds contribute when
   \[
   \frac{f(y^{bc})}{f(y)}
   \ne
   \frac{f(y^b)f(y^c)}{f(y)^2}.
   \]

   [Derived] Extensive \(t\|H\|\) alone does not invalidate a local expansion; absence of bounds on these connected terms does. The ground-state identity makes J1/J2 undoing identical to first order, but supplies no second-order factorization.

   [Checked] I independently implemented R3 at 4×4, without reading worker outputs. Two implementations—sparse one-bond exponentials and analytic factors accumulated logarithmically—agree. Total checks used about nine CPU-seconds.

   | \(t\) | \((E_{\rm var}(\psi_L)-E_0)/N\) | Squared ground-state overlap |
   |---|---:|---:|
   | 0.16 | 0.653620 | \(2.29\times10^{-6}\) |
   | 0.20 | 0.00315230 | 0.978613 |

   [Checked] The one-bond round-trip residual was \(2.6\times10^{-16}\); reversing the sign gave 0.0599. FN solves were numerically unreliable, so these numbers do **not** establish R3’s FN verdict.

   [Derived] The instability has a concrete mechanism: for \(M\) antiparallel bonds,
   \[
   \psi_L\propto f^{\,1-M}\prod_b(af+bf_b).
   \]
   Small \(f\) can produce enormous artificial enhancements. The exact linear propagator has no such denominators.

   [Opinion] The cheapest improvement to test is exact two-bond cluster factors, beginning with overlapping three-site clusters: \(O(N)\) constant-size contractions and base-ratio calls. But inspect omitted disjoint \(\Gamma_{bc}\) first; correcting **all** pairs costs \(O(N^2)\). An overlapping-only correction is uncontrolled if distant conditional correlations remain large.

2. **Is the PASS rule appropriate?**

   [Opinion] It is a permissive screening rule, not the PI’s loop bar.

   [Derived] At 4×6, \(0.1\) times Marshall-FN allows \(6.95\times10^{-4}\)/site—847 times the numerical value of the 6×6 tax gate. These are different diagnostics, so the comparison is illustrative, but it exposes the precision gap.

   [Opinion] Keep the registered classification, but require the next stage to use a common frozen \(t\), an actual FN target, and the same two-hop/FN readout as G-tax. Separate decoder error from fitting error, and match total computational cost.

   [Derived] “\(O(N)\)” means \(2N\) base-network ratio evaluations **per amplitude**. It does not include the network’s own cost, neighboring amplitudes, subsequent hops or FN work. Also, each ratio may depend on every spin; a local exchange is not necessarily a local decision rule.

3. **Shortest path after PASS, and interpretation after failure**

   [Opinion] After PASS, use the existing 6×6 G-tax FN target. First reconstruct from exact preimage magnitudes with the proposed sign rule and local decoder, then apply \(K_2+\mathrm{FN}\). This isolates decoder error before training.

   [Opinion] If that survives, run the matched positive-network fit and measure
   \[
   \mathrm{tax}_2=
   E_{\rm FN}[K_2D_t(f_{\rm fit})]
   -E_{\rm FN}[K_2\phi_{\rm FN}],
   \]
   retaining the \(0.82\times10^{-6}\) gate and \(1.657\times10^{-6}\) baseline. Report exact-store decoder bias separately; fitting could otherwise conceal it through accidental cancellation.

   [Opinion] Most likely failure: correlated ratio errors move many factors across zero simultaneously. Near \(g_b=0\),
   \[
   \delta\log|g_b|\simeq\frac{bR_b}{a+bR_b}\,\delta\log R_b,
   \]
   so the product decoder can amplify errors that were harmless under the exact exponential.

   [Derived] After failure, “this independent-bond decoder fails” is justified. “Reconstruction is nonlocal and the route costs \(e^{0.067N}\)” is not.

   [Checked] At \(t=0.16\), the raw costs are \(C_{16}=3.376\), \(C_{24}=4.981\). [Derived] They also fit \(C\propto N^{0.959}\); \(C/N\) differs by only 1.6%. The registered exponential-signal rule fires, but two points identify neither an asymptotic exponent nor a lower bound for other estimators.

4. **Where the step-back overreaches**

   [Derived] Three statements need narrowing:

   - **“The stored preimage is benign.”** Five smooth Jastrow perturbations probe a tiny error family and variational-energy response; they establish neither learnability nor FN refit tax.
   - **“J2 has a positive reconstruction kernel whereas J1 does not.”** This is false: \(e^{+uH_{J1}}=e^{-u(-H_{J1})}\) is entrywise nonnegative in the original spin basis because \(-H_{J1}\) is stoquastic. Both constructions have signed boundary vectors.
   - **“Taylor truncation: no convergence.”** Finite-dimensional exponentials converge. Fixed-order global truncation lacks uniform size control; that is the actual objection.

   [Derived] Changing the decoder also changes the optimal stored target: \(e^{tH_{J2}}\psi\) is the inverse image of the exponential, not of your nonlinear product.

**[Opinion] The single point where you are most likely wrong:** treating the ratio-curve overlay as justification for independently multiplying local sign events. The omitted correlated exchanges are precisely the assumption R3 must test.